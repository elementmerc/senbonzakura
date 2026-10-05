# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Replies out of a GGUF, through the pinned llama.cpp server, for the same scorers.

WHY THIS EXISTS

`score` loads a model through transformers, and transformers cannot read a GGUF. GGUF is how
most people actually run these models on their own machines, and several of the models worth
measuring are published in that format first. So a measurement tool that cannot open one cannot
measure the artefact people are arguing about. Decision Q-88 commissioned this.

WHAT IT IS, AND WHAT IT DELIBERATELY IS NOT

This is a GENERATOR and not a second scorer. It hands back reply strings, and `score.score`
applies exactly the same `metrics` detectors to them that the transformers path uses. One
instrument, two ways of getting the replies, which is the only arrangement under which two
numbers from the two paths mean the same thing.

It is NOT a fallback. Nothing here is reached when the transformers path fails, and nothing here
quietly substitutes for it. A run either asked for a GGUF or it did not.

WHY A SERVER SUBPROCESS AND NOT A LIBRARY BINDING

The pinned llama.cpp release carries `libllama.so`, whose C API exports `llama_get_logits_ith`,
so a ctypes binding would reach further than this does. It would also mean writing struct
layouts for `llama_model_params`, `llama_context_params` and `llama_batch` by hand against a
header the release archive does not contain, and getting one field wrong is memory corruption
rather than an exception. In a tool whose rule is fail loud, a binding whose failure mode is
silent corruption is the wrong trade, so the subprocess boundary is deliberate: it costs a port
and a process, and everything that can go wrong comes back as an error with a sentence on it.

WHY THE NUMBERS ARE REPRODUCIBLE, WHICH TAKES THREE SEPARATE DECISIONS

1. **Greedy, explicitly.** Temperature 0, top-k 1, and the sampler chain cut to temperature
   alone, so none of llama.cpp's default penalties (repeat, dry, xtc, min-p) can touch the
   output. The defaults are not greedy, and a scorer that inherited them would be measuring a
   sampler.
2. **Threads pinned, not inherited.** llama.cpp's CPU result depends on how the work is split,
   and the default thread count is the machine's core count. That would make the same command on
   two machines a different measurement, so the thread count is fixed here, recorded in the
   provenance, and `AGREEMENT_FIELDS` refuses to compare two runs that used different values.
3. **No prompt cache.** `cache_prompt` is off, so a prompt's reply cannot depend on which prompt
   ran before it.

WHAT IT CANNOT DO TODAY

The compass (`margin.py`) needs the next-token logit vector, and the server CAN return the whole
of it: `/completion` with `n_probs` set to the vocabulary size returns a log probability for
every token id, and the compass's statistic is a DIFFERENCE of two logits at one position, which
is unchanged by the constant that separates a log probability from a logit. So the compass is
reachable this way and is not built here, because it is a separate instrument with its own
validity checks and it deserves its own decision rather than arriving as a side effect of this
module. What this module does today is refusal scoring, and it says so rather than implying
more.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

from . import metrics, vendored

#: The four bytes every GGUF starts with, imported from the reader that owns the format rather
#: than spelled a second time here.
from .gguf_io import MAGIC as GGUF_MAGIC

#: The binary out of the pinned llama.cpp release. It is NOT in `vendor_llama.py`'s `WANT_BINS`
#: today, so an ordinary install does not carry it and the refusal below says what to do.
SERVER_BIN = "llama-server"

#: How long to wait for the server to answer `/health` before giving up. A model has to be read
#: off disk first, so this is generous; it is a deadline rather than a loop without one, because
#: baseline section 2.1 has no exception for a subprocess that never comes up.
HEALTH_DEADLINE_S = 300

#: Per request. One prompt at 64 new tokens is well inside this on a CPU; a full vocabulary of
#: log probabilities on a large model is the slow case this leaves room for.
REQUEST_TIMEOUT_S = 600

#: How long the process gets to exit after being asked politely, before it is killed.
SHUTDOWN_GRACE_S = 10

#: Fixed rather than taken from the machine. See point 2 in the module docstring.
DEFAULT_THREADS = 4

#: Enough for a rendered chat prompt plus a 64-token reply on every model measured here, and
#: small enough that the server allocates its KV cache in a second rather than a minute.
DEFAULT_CONTEXT = 2048

#: The same budget `score` uses, so the two paths truncate a long reply at the same place.
DEFAULT_MAX_NEW = 64

#: File types that carry the weights without loss, so a reply difference between the two paths
#: cannot be explained by quantisation. The per-prompt floor below applies only to these.
LOSSLESS_FILE_TYPES = ("F32", "F16", "BF16")

#: For a LOSSLESS GGUF, the share of prompts on which the two paths must reach the same refusal
#: verdict. Not 1.0: the two stacks use different kernels and different arithmetic order, so a
#: reply near the boundary can land either side of it, and demanding exactness would be demanding
#: that two implementations of floating point agree. 0.95 is the bar because at that point a
#: disagreement is an outlier rather than a pattern, and a pattern is what would mean the two
#: paths are not running the same model. For a QUANTISED file no floor is applied at all, because
#: quantisation genuinely changes replies and a threshold there would be a number nobody measured.
MIN_LOSSLESS_AGREEMENT = 0.95

#: Conditions under which two refusal rates from the two paths are the same measurement. Each is
#: a field in the provenance, and a difference in any of them is reported by name rather than
#: averaged away, which is `baseline.comparability`'s shape and is here for its reason: a
#: comparison across a mismatch is how a drifted prompt renderer got a published table.
AGREEMENT_FIELDS = (
    ("prompt", "the rendered prompt text, which is what the model actually read"),
    ("max_new", "the reply budget, because a refusal can arrive after a short budget ends"),
)


class GgufRunError(Exception):
    """Anything that stopped a GGUF from being scored, phrased for the person who ran it."""


def _sha256(path, *, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def server_identity(exe, source, *, log=print):
    """Which binary ran, what it says its own build is, and whether that is the pinned one.

    `quantise.build_info` asks `llama-quantize` by dry-running a quantisation; the server answers
    `--version` directly. The regex and the pin reading are imported from there rather than
    copied, because two spellings of "what build is this" drift, and the one that drifts is
    whichever is read less often.
    """
    from .quantise import _BUILD_RE, pinned_tag

    info = None
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                           timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        r = None
    if r is not None:
        text = (getattr(r, "stdout", "") or "") + (getattr(r, "stderr", "") or "")
        m = _BUILD_RE.search(text if isinstance(text, str) else "")
        if m:
            build, commit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            info = {"build": int(build), "commit": commit}
    tag = pinned_tag()
    ident = {"tool": SERVER_BIN, "source": source, "path": str(exe),
             "pinned_tag": tag, "reported_build": info, "sha256": None}
    try:
        ident["sha256"] = _sha256(exe)
    except OSError:
        pass
    import re
    if info and tag and re.fullmatch(r"b\d+", str(tag)) and int(str(tag)[1:]) != info["build"]:
        ident["pin_mismatch"] = True
        log(f"  WARNING: the pin says {tag} and {SERVER_BIN} reports build {info['build']}. "
            f"The runner that produced this number is not the one this install claims to "
            f"vendor, so record both before comparing it with anything.")
    return ident


def model_identity(path):
    """What was loaded: its hash, its architecture, and whether its weights are lossless.

    The hash is the field that makes a figure checkable. Two GGUFs of one model from two
    publishers are different files with different quantisers behind them, and a result naming
    only the model has not said which one it measured.
    """
    from . import gguf_io

    # ABSENCE IS CHECKED FIRST, AND THE REASON IS THE MESSAGE RATHER THAN THE CONTROL FLOW.
    # `read_header` turns a missing file into its own GGUFError, so without this a path that is
    # simply not there was reported as "not a GGUF this tool can read", which accuses the file of
    # being malformed when the fault is a typo in a path. `gguf_io`'s own tests call that class of
    # defect the wrong accusation, and this is the same one a layer up.
    if not os.path.exists(path):
        raise GgufRunError(
            f"there is no file at {path}, so it could not be read. Nothing was run.")
    try:
        header = gguf_io.read_header(path)
    # No `except OSError` beside it: `gguf_io` catches its own read failures and reports them as
    # a GGUFError carrying the errno, so a second handler here would be a branch nothing can
    # reach, and an unreachable error path reads as a guard while guarding nothing.
    except gguf_io.GGUFError as e:
        raise GgufRunError(
            f"{path} is not a GGUF this tool can read, so nothing was run:\n    {e}") from e
    file_type = header.get("file_type")
    try:
        census = gguf_io.type_census(path)
    except (gguf_io.GGUFError, OSError):
        # The header parsed and the tensor table did not. The census is a diagnostic rather than
        # a precondition, so it is left absent and the run goes on.
        census = None
    # The hash is read after the header, so a file that disappears in between is a real sequence
    # rather than an invented one, and it must not arrive as a bare OSError from inside a
    # measurement.
    try:
        digest, size = _sha256(path), os.path.getsize(path)
    except OSError as e:
        raise GgufRunError(
            f"{path} parsed as a GGUF and then could not be read to the end: {e}. Nothing was "
            f"run, because a figure that cannot name the file it measured is not reproducible."
        ) from e
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": size,
        "architecture": header.get("architecture"),
        "file_type": file_type,
        "tensor_count": header.get("tensor_count"),
        "tensor_types": census,
        "lossless": file_type in LOSSLESS_FILE_TYPES,
    }


def _free_port():
    """A loopback port the kernel says is free, which is the only authority on that question."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GgufRunner:
    """A started llama.cpp server, bound to loopback, that hands back greedy replies.

    Use it as a context manager. The process is stopped on the way out whatever happened, because
    a server left running holds a model's worth of memory and the next run binds a different port
    and looks fine.
    """

    def __init__(self, gguf, *, binary=None, threads=DEFAULT_THREADS, context=DEFAULT_CONTEXT,
                 max_new=DEFAULT_MAX_NEW, port=None, log=print, health_deadline=HEALTH_DEADLINE_S,
                 request_timeout=REQUEST_TIMEOUT_S):
        self.gguf = str(gguf)
        self.threads = int(threads)
        self.context = int(context)
        self.max_new = int(max_new)
        self.log = log
        self.health_deadline = float(health_deadline)
        self.request_timeout = float(request_timeout)
        self._proc = None
        self._log_path = None
        self._port = port
        if self.threads < 1:
            raise GgufRunError(f"threads must be at least 1, and is {threads!r}")
        if self.max_new < 1:
            raise GgufRunError(f"the reply budget must be at least 1 token, and is {max_new!r}")
        self.model = model_identity(self.gguf)
        if binary is None:
            exe, source = self._locate()
        else:
            exe, source = str(binary), "explicit"
            if not os.path.exists(exe):
                raise GgufRunError(f"no {SERVER_BIN} at {exe}, which was named explicitly")
        self.binary = exe
        self.identity = server_identity(exe, source, log=self.log)

    def _locate(self):
        """The pinned binary, or a refusal that says how to get one.

        `vendored.find_binary` already implements the order this wants (the copy inside this
        install, then PATH, then a refusal naming both), so this adds only the part it cannot
        know: that `llama-server` is not in the vendoring tool's binary list today, which is the
        actual reason an otherwise complete install does not have it.
        """
        try:
            return vendored.find_binary(SERVER_BIN, log=self.log)
        except vendored.VendorError as e:
            raise GgufRunError(
                f"a GGUF needs {SERVER_BIN} to run it, and this install does not have one.\n"
                f"    {e}\n"
                f"  A platform wheel carries it, and a source checkout gets it from "
                f"`python tools/packaging/vendor_llama.py`, which fetches the pinned llama.cpp "
                f"release and verifies its hash before extracting anything. A build of that same "
                f"release on PATH also works, and the provenance records which one answered so a "
                f"figure says what produced it.") from e

    # ── the process ──────────────────────────────────────────────────────────────
    def start(self):
        if self._proc is not None:
            raise GgufRunError("this runner is already started")
        port = self._port or _free_port()
        self._port = port
        argv = [self.binary, "--model", self.gguf, "--host", "127.0.0.1", "--port", str(port),
                "--ctx-size", str(self.context), "--threads", str(self.threads),
                "--no-warmup", "--log-disable"]
        import tempfile
        handle, self._log_path = tempfile.mkstemp(prefix="senbonzakura-llama-server-", suffix=".log")
        self._log_file = os.fdopen(handle, "w+")
        self._keep_log = False
        try:
            self._proc = subprocess.Popen(argv, stdout=self._log_file, stderr=subprocess.STDOUT)
        except OSError as e:
            self._log_file.close()
            self._discard_log()
            raise GgufRunError(f"{SERVER_BIN} would not start: {e}") from e
        self.log(f"  {SERVER_BIN} on 127.0.0.1:{port}, {self.threads} threads, "
                 f"context {self.context}")
        self._await_health()
        return self

    def _discard_log(self):
        """Remove the server's log, which only a clean run is allowed to do.

        A temporary file nobody deletes is the `.part` left behind that baseline section 2.1
        forbids, and a diagnostic nobody can read is the observability it asks for. Both are
        satisfied by deciding on the way out: a run that failed keeps its log and says where it
        is, and a run that worked leaves nothing.
        """
        path, self._log_path = getattr(self, "_log_path", None), None
        if path is None:
            return
        try:
            os.unlink(path)
        except OSError:
            # It is a temporary file either way, and failing to tidy one is not a reason to turn
            # a finished measurement into an error.
            self.log(f"  note: the server's log at {path} could not be removed")

    def _log_tail(self, lines=12):
        """What the server said, for a message about why it did not come up."""
        self._keep_log = True
        try:
            with open(self._log_path, encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-lines:]).strip()
        except OSError:
            return "(its log could not be read)"

    def _await_health(self):
        deadline = time.monotonic() + self.health_deadline
        last = ""
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                # `stop` BEFORE RAISING, for the same reason the deadline path below does it: it
                # is what closes the log handle. Raising straight out of here left an open file
                # object and the temp file beside it, which the suite's ResourceWarning gate
                # caught as an unclosed `rb+` handle on a test that was otherwise passing.
                status, tail = self._proc.returncode, self._log_tail()
                self.stop()
                raise GgufRunError(
                    f"{SERVER_BIN} exited with status {status} before it was ready, so nothing "
                    f"was measured. Its last words:\n{tail}")
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last = f"{type(e).__name__}: {e}"
            time.sleep(0.25)
        # THE TAIL IS READ BEFORE THE PROCESS IS STOPPED, and the order is load bearing. `stop`
        # decides what happens to the log, and on a clean run it deletes it; reading the tail
        # afterwards found a `None` path and raised TypeError, so the deadline path crashed
        # instead of producing the one message that explains a server that never came up. Found
        # by the test for the deadline rather than by a run.
        tail = self._log_tail()
        self.stop()
        raise GgufRunError(
            f"{SERVER_BIN} did not answer /health within {self.health_deadline:.0f}s, so the run "
            f"was abandoned rather than left waiting. Last attempt: {last}\n{tail}")

    def stop(self):
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=SHUTDOWN_GRACE_S)
            except subprocess.TimeoutExpired:
                self.log(f"  {SERVER_BIN} ignored terminate, killing it")
                proc.kill()
                proc.wait(timeout=SHUTDOWN_GRACE_S)
        handle = getattr(self, "_log_file", None)
        if handle is not None and not handle.closed:
            handle.close()
        if getattr(self, "_keep_log", False):
            self.log(f"  the server's log is kept at {self._log_path}")
        else:
            self._discard_log()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.stop()
        return False

    @property
    def base(self):
        return f"http://127.0.0.1:{self._port}"

    # ── talking to it ────────────────────────────────────────────────────────────
    def request(self, path, payload):
        """One JSON request to the server, with every failure turned into a sentence.

        Public because the compass helpers below are collaborators rather than callers from
        outside: they need the same error handling and the same timeout, and a second copy of
        either would be a second set of failure messages for one failure.
        """
        if self._proc is None:
            raise GgufRunError("the runner is not started, so there is nothing to ask")
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode("utf-8", "replace")
            raise GgufRunError(
                f"{SERVER_BIN} refused {path}: HTTP {e.code}. {detail}") from e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            alive = self._proc.poll() is None
            raise GgufRunError(
                f"{SERVER_BIN} did not answer {path} ({type(e).__name__}: {e}). The process is "
                f"{'still running' if alive else 'gone'}.\n{self._log_tail()}") from e
        try:
            return json.loads(raw)
        except ValueError as e:
            raise GgufRunError(
                f"{SERVER_BIN} answered {path} with something that is not JSON, so nothing can "
                f"be read from it: {raw[:200]!r}") from e

    def rendered(self, prompt):
        """The prompt as the model will read it, rendered by the template inside the GGUF.

        Asked of the server rather than guessed at, for two reasons. A GGUF user has no
        transformers tokenizer to render with, and the agreement control needs this string in
        order to compare it against the one the transformers path builds. A template difference
        between the two paths would make every number afterwards a comparison of templates.
        """
        out = self.request("/apply-template", {"messages": [{"role": "user", "content": prompt}]})
        text = out.get("prompt")
        if not isinstance(text, str) or not text:
            raise GgufRunError(
                f"{SERVER_BIN} returned no rendered prompt for this model, so there is no way to "
                f"know what it would have read. A GGUF with no chat template cannot be scored "
                f"comparably with a chat model loaded through transformers.")
        return text

    def reply(self, rendered_prompt):
        """One greedy reply to an already rendered prompt."""
        out = self.request("/completion", {
            "prompt": rendered_prompt,
            "n_predict": self.max_new,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 0,
            "cache_prompt": False,
            # The default chain carries repeat, dry, xtc and min-p. None of them belongs in a
            # measurement, and leaving them in place would have measured llama.cpp's defaults.
            "samplers": ["temperature"],
        })
        text = out.get("content")
        if not isinstance(text, str):
            raise GgufRunError(
                f"{SERVER_BIN} returned no reply text for a prompt (keys: {sorted(out)}), and a "
                f"missing reply scored as an empty one would count as compliance.")
        return text

    def replies(self, prompts, *, render=True, heartbeat=30.0):
        """A reply per prompt, in order, with a heartbeat on a long run.

        Sequential on purpose. The server has several slots and could take these in parallel, and
        a reply would then depend on what shared the batch with it, which is the reproducibility
        the module docstring spends three paragraphs on.
        """
        if not prompts:
            raise GgufRunError("no prompts were given, so there is nothing to score")
        out, last = [], time.monotonic()
        for i, p in enumerate(prompts):
            out.append(self.reply(self.rendered(p) if render else p))
            if time.monotonic() - last >= heartbeat:
                self.log(f"  {i + 1}/{len(prompts)} prompts")
                last = time.monotonic()
        return out

    def provenance(self):
        """Everything a reader needs to tell this number apart from another one."""
        return {
            "runner": "llama.cpp server",
            "binary": self.identity,
            "model": self.model,
            "settings": {
                "threads": self.threads, "context": self.context, "max_new": self.max_new,
                "decoding": "greedy", "temperature": 0.0, "top_k": 1, "seed": 0,
                "samplers": ["temperature"], "cache_prompt": False, "parallel_requests": 1,
            },
        }


# ── the agreement control, which is the condition Q-88 attaches to all of the above ──
def comparability(left, right):
    """Every condition the two paths disagree about, as (field, left, right, why).

    Named one by one rather than reduced to a boolean, because the useful output of a failed
    comparison is which axis failed.
    """
    bad = []
    for field, why in AGREEMENT_FIELDS:
        a, b = left.get(field), right.get(field)
        if a != b:
            bad.append((field, a, b, why))
    return bad


def agreement(hf_replies, gguf_replies, *, lossless, prompts=None, seed=0,
              resamples=metrics.DEFAULT_RESAMPLES, floor=metrics.MIN_REPORTABLE_N,
              min_lossless_agreement=MIN_LOSSLESS_AGREEMENT):
    """Do the two paths measure the same refusal rate on the same prompts?

    THE BAR, AND WHY IT IS NOT EQUALITY. Identical rates would be the wrong test: the two stacks
    run different kernels in a different arithmetic order, and a quantised GGUF is not even the
    same weights. So the question asked here is the one the project asks everywhere else, whether
    the difference is distinguishable from nothing at this sample size. The estimator is
    `metrics.paired_rate_bootstrap` on the SAME prompts, which is what makes the pairing
    legitimate, and the verdict is whether its interval on the difference contains zero.

    Two further conditions sit alongside it:

      * below `metrics.MIN_REPORTABLE_N` prompts nothing is reported at all, which is the floor
        the rest of this project already refuses under;
      * for a LOSSLESS file, the per-prompt verdicts must agree on at least
        `min_lossless_agreement` of prompts. Equal rates with scattered per-prompt disagreement
        is a real and separate failure, and on a file whose weights are unchanged it cannot be
        blamed on quantisation. For a quantised file this floor is not applied, because
        quantisation legitimately changes individual replies and a threshold there would be a
        number nobody measured.

    Returns the finding either way. A refusal to call the two comparable IS the result, not an
    error, and `verdict` carries it.
    """
    if len(hf_replies) != len(gguf_replies):
        raise GgufRunError(
            f"the two paths produced {len(hf_replies)} and {len(gguf_replies)} replies. An "
            f"agreement figure needs one reply per prompt from each, aligned by prompt.")
    n = len(hf_replies)
    hf_flags = [metrics.is_refusal(t) for t in hf_replies]
    gg_flags = [metrics.is_refusal(t) for t in gguf_replies]
    same = [a == b for a, b in zip(hf_flags, gg_flags, strict=True)]
    per_prompt = (sum(same) / n) if n else None
    hf_rate = (sum(hf_flags) / n) if n else None
    gg_rate = (sum(gg_flags) / n) if n else None
    paired = None
    if n >= 2:
        paired = metrics.paired_rate_bootstrap(
            {"hf": hf_flags, "gguf": gg_flags},
            lambda r: r["gguf"] - r["hf"], seed=seed, resamples=resamples)
    reasons = []
    if n < floor:
        reasons.append(
            f"{n} prompts is below the {floor} this project will state a rate on at all, so this "
            f"run cannot say whether the two paths agree")
    if paired is not None and paired.get("crosses_zero") is False:
        reasons.append(
            f"the paired interval on the difference is {list(paired['ci'])} and does not contain "
            f"zero, so the two paths measure different refusal rates on these prompts")
    if lossless and per_prompt is not None and per_prompt < min_lossless_agreement:
        reasons.append(
            f"the per-prompt verdicts agree on {per_prompt:.1%} of prompts, below the "
            f"{min_lossless_agreement:.0%} expected of a file whose weights are unchanged, so the "
            f"two paths are not running the same model")
    result = {
        "n": n,
        "lossless": bool(lossless),
        "transformers_refusal": None if hf_rate is None else round(hf_rate, 4),
        "gguf_refusal": None if gg_rate is None else round(gg_rate, 4),
        "difference": None if (hf_rate is None or gg_rate is None) else round(gg_rate - hf_rate, 4),
        "paired_difference": paired,
        "per_prompt_agreement": None if per_prompt is None else round(per_prompt, 4),
        # MEASURED 2026-10-06, AND IT IS WHY THERE IS NO BAR ON THIS. The same weights through the
        # same stack, float32 against bfloat16, agreed on only 17 of 40 replies, while the refusal
        # rate (0.050) and the keyword rate (0.075) were identical in both. So this number moves
        # with the dtype either arm happens to load in, and the rates do not. A threshold on it
        # would be a threshold on numerical precision wearing the name of a runner comparison.
        #
        # A DIAGNOSTIC AND NOT A BAR, and it earns its place when the refusal rate cannot speak.
        # Two paths that both refuse nothing produce equal rates whatever they are doing, so on a
        # model with no refusals the rate agreement is satisfied by a model that is broken in both
        # paths. The share of byte-identical greedy replies has no such blind spot: it is a much
        # harder question than the rate, nobody should expect it to be 1.0 across two arithmetic
        # orders, and no threshold is applied to it for exactly that reason.
        "identical_replies": (None if not n else
                              round(sum(1 for a, b in zip(hf_replies, gguf_replies, strict=True)
                                        if a == b) / n, 4)),
        "per_prompt_floor_applied": bool(lossless),
        "disagreeing_prompts": [i for i, s in enumerate(same) if not s],
        "comparable": not reasons,
        "why_not": reasons,
    }
    if prompts is not None and len(prompts) == n:
        result["examples"] = [
            {"prompt": prompts[i], "transformers": hf_replies[i], "gguf": gguf_replies[i]}
            for i in result["disagreeing_prompts"][:5]]
    return result


def refuse_if_incomparable(finding):
    """Raise unless the agreement run earned the right to call the two paths one measurement.

    The same shape as `baseline.refuse_if_incomparable`, and for the same reason: a figure from
    this runner quoted beside a transformers figure is a claim that the two are the same
    measurement, and that claim has to be earned by a check rather than assumed because both
    numbers came out of the same tool.
    """
    if finding.get("comparable"):
        return
    lines = "\n".join(f"  * {why}" for why in finding.get("why_not") or ["no reason recorded"])
    raise GgufRunError(
        "this GGUF run is NOT comparable with a transformers run of the same model, so it must "
        "not be quoted beside one:\n" + lines
        + "\n\nThe figure itself still stands as a measurement of the GGUF on its own terms. "
          "What it cannot do is stand in a table beside a number produced the other way.")


def describe(finding):
    """The agreement finding in the lines a reader of a result needs."""
    lines = [(f"agreement on {finding['n']} prompts, "
              f"{'lossless' if finding['lossless'] else 'quantised'} file")]
    lines.append(f"  transformers refusal: {finding['transformers_refusal']}")
    lines.append(f"  GGUF refusal:         {finding['gguf_refusal']}")
    paired = finding.get("paired_difference") or {}
    if paired.get("ci"):
        lines.append(f"  difference:           {finding['difference']} "
                     f"(paired 95% interval {list(paired['ci'])}, "
                     f"{'contains' if paired.get('crosses_zero') else 'excludes'} zero)")
    lines.append(f"  same verdict on:      {finding['per_prompt_agreement']} of prompts")
    lines.append(f"  identical replies:    {finding['identical_replies']} of prompts "
                 f"(a diagnostic, with no bar on it)")
    if finding["comparable"]:
        lines.append("  COMPARABLE: a figure from this runner may be quoted beside a "
                     "transformers figure for this model.")
    else:
        lines.append("  NOT COMPARABLE, and this is the finding rather than a failure:")
        lines.extend(f"    {why}" for why in finding["why_not"])
    return lines


def looks_like_gguf(path):
    """Is this input a GGUF file on disk?

    BY ITS MAGIC BYTES AND NOT BY ITS NAME. `--model` already takes a Hub id, a directory of
    safetensors and a path, so a fourth kind of input is a detection rather than a flag: two
    flags that each mean "the model" is the two-instruments problem this module was written to
    avoid, one level up. A file called `model.gguf` that is not one is refused by
    `model_identity` with a sentence about the file, which is the honest failure; a GGUF called
    something else still works.
    """
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


#: How many token log probabilities to ask for on the first try. The verdict tokens on a judging
#: prompt are usually near the top, so most prompts are answered by a request of this size, and
#: the full vocabulary is asked for only when they are not. Measured in `verdict_logprobs`'s
#: docstring; the fallback is what makes the cheap request safe rather than lucky.
FIRST_TRY_PROBS = 1000


class VerdictReadout:
    """One position's verdict log probabilities, and how they were obtained.

    `escalated` is not a detail. A prompt answered from the cheap request and one that needed the
    whole vocabulary are the same number, and a reader asking why a run took twenty minutes wants
    to know how often the cheap path missed.
    """

    __slots__ = ("benign", "escalated", "harmful", "returned", "top")

    def __init__(self, harmful, benign, *, top, returned, escalated):
        self.harmful = harmful
        self.benign = benign
        self.top = top
        self.returned = returned
        self.escalated = escalated

    @property
    def verdict_mass(self):
        """How much probability the two verdict sets hold between them at this position.

        `margin.READOUT_SUSPECT_MASS` is the floor this feeds, and it is a WARNING rather than a
        gate in that module for a reason written down there: a constant that quietly decides an
        outcome is a defect this project has already paid for. The same applies here, so this
        reports the number and decides nothing.
        """
        import math

        return sum(math.exp(v) for v in (*self.harmful.values(), *self.benign.values()))

    @property
    def margin(self):
        """The max over the HARMFUL spellings minus the max over the BENIGN ones, or None.

        THE SAME QUANTITY `margin.margins` COMPUTES, and the reason log probabilities are allowed
        to stand in for logits is that this is a DIFFERENCE at one position. A log probability is
        its logit minus that position's log-sum-exp, the same constant for every token there, and
        the constant cancels in the subtraction. `margin.py`'s own note at the call site says the
        statistic is a difference of two logits at one position, which is what makes the
        substitution exact rather than approximate.
        """
        if not self.harmful or not self.benign:
            return None
        return max(self.harmful.values()) - max(self.benign.values())


class GgufCompassError(GgufRunError):
    """The verdict position could not be read, with the reason."""


def verdict_logprobs(runner, rendered_prompt, harmful_ids, benign_ids, *,
                     first_try=FIRST_TRY_PROBS):
    """The log probability of every verdict token at the position the prompt ends on.

    THE PAYLOAD PROBLEM, AND WHAT WAS ACTUALLY TESTED. The whole next-token distribution is
    available (`n_probs` set to the vocabulary size returns every token) and it is expensive:
    4.5 MiB and 0.23 s per prompt on a 49,152-token vocabulary, so a 200 plus 200 compass pass
    moves about 1.8 GB. Three cheaper requests were tried against the pinned build before this
    settled on escalation:

      * `logit_bias` on the verdict tokens, to force them into a small top-N. **Measured and it
        does not work.** With `post_sampling_probs` false the returned list is the RAW
        distribution and the bias does not move it, so the verdict tokens were absent from the
        top 8 exactly as before. With it true the list is post-sampling, and at temperature 0 the
        chain collapses to one token at probability 1.0, which is not a margin at all.
      * `top_k` and friends: same answer, and for the same reason. They are samplers, and the
        reported distribution is read before the sampler chain.
      * asking for the two token ids directly: the server has no such parameter at b11046.

    So the request that works is `n_probs`, and the saving available is to ask for a MODERATE
    number first and fall back to the whole vocabulary only when a verdict token is missing from
    it. On a judging prompt the verdict words are what the model is about to say, so the cheap
    request usually contains them, and `escalated` records every time it did not. The fallback is
    what makes this exact: a margin is never computed from a partial view.
    """
    def read(n_probs):
        out = runner.request("/completion", {
            "prompt": rendered_prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 0,
            "cache_prompt": False,
            "samplers": ["temperature"],
            "n_probs": n_probs,
        })
        rows = (out.get("completion_probabilities") or [{}])[0]
        entries = rows.get("top_logprobs") or []
        if not entries:
            raise GgufCompassError(
                f"{SERVER_BIN} returned no token probabilities for this prompt, so there is no "
                f"verdict position to read. Keys present: {sorted(rows)}.")
        found = {e["id"]: e["logprob"] for e in entries if isinstance(e.get("id"), int)}
        return (entries,
                {i: found[i] for i in harmful_ids if i in found},
                {i: found[i] for i in benign_ids if i in found})

    # Two explicit attempts rather than a loop, so there is no branch that cannot be reached.
    # The loop this replaces ended in a raise that nothing could ever execute, which reads as a
    # guard and guards nothing.
    entries, harmful, benign = read(first_try)
    if harmful and benign:
        return VerdictReadout(harmful, benign, top=entries[0].get("token"),
                              returned=len(entries), escalated=False)
    entries, harmful, benign = read(_vocabulary_ceiling(runner))
    return VerdictReadout(harmful, benign, top=entries[0].get("token"),
                          returned=len(entries), escalated=True)


def _vocabulary_ceiling(runner):
    """A request large enough to mean "every token", whatever this model's vocabulary is.

    The server clamps `n_probs` to the vocabulary, measured at b11046: asking for 60,000 on a
    49,152-token model returned exactly 49,152. So a ceiling above any real vocabulary is a way
    of saying "all of them" without having to read the vocabulary size out of the file first, and
    the response says how many actually came back.
    """
    return int(getattr(runner, "vocabulary_ceiling", 1_000_000))


def verdict_token_ids(runner, word):
    """First-token ids for a verdict word, through the GGUF's own tokeniser.

    `margin.label_token_ids` does this with a transformers tokeniser and the same six spellings.
    The ids are NOT interchangeable between the two: a GGUF carries its own vocabulary, and using
    a transformers id against it would read a different token and call it a verdict. So the two
    paths each ask their own tokeniser, which is also why the compass agreement control compares
    MARGINS rather than ids.
    """
    ids = set()
    for variant in (word, " " + word, word.capitalize(), " " + word.capitalize(),
                    word.lower(), " " + word.lower()):
        out = runner.request("/tokenize", {"content": variant, "add_special": False})
        tokens = out.get("tokens") or []
        if tokens and isinstance(tokens[0], int):
            ids.add(tokens[0])
    if not ids:
        raise GgufCompassError(
            f"the GGUF's tokeniser returned no tokens for any spelling of {word!r}, so there is "
            f"no verdict token to read a margin between. Nothing was measured.")
    return sorted(ids)


def compass_margins(runner, rendered_prompts, harmful_ids, benign_ids, *, heartbeat=30.0,
                    log=None):
    """A margin per prompt, in the rows `margin.py` already reports, plus how they were read.

    Sequential and uncached for the reasons in this module's docstring: a verdict position that
    depends on which prompt shared a batch is not a measurement of this prompt.
    """
    say = log or runner.log
    rows, escalated, last = [], 0, time.monotonic()
    for i, rendered in enumerate(rendered_prompts):
        readout = verdict_logprobs(runner, rendered, harmful_ids, benign_ids)
        escalated += int(readout.escalated)
        rows.append({"margin": readout.margin, "top": readout.top,
                     "verdict_mass": readout.verdict_mass,
                     "escalated": readout.escalated})
        if time.monotonic() - last >= heartbeat:
            say(f"  {i + 1}/{len(rendered_prompts)} verdict positions read")
            last = time.monotonic()
    return rows, {"escalated": escalated, "n": len(rows),
                  "first_try_probs": FIRST_TRY_PROBS}


def compass_agreement(hf_harmful, hf_benign, gguf_harmful, gguf_benign, *, seed=0):
    """Do the two paths' compasses agree, on the same prompts and the same statistic?

    WHY THE COMPARISON IS THE AUC AND NOT THE MARGINS THEMSELVES. A margin is a difference of two
    logits, and the two stacks do not share a scale: a quantised file's logits are not the
    original's, and even a lossless conversion runs them through different kernels. What the
    compass REPORTS is the AUC, which depends only on the ORDER of the margins, so that is the
    quantity two runners can be held to. Comparing raw margins would manufacture a disagreement
    out of a scale difference nobody claims is the same.

    The estimator is `margin.paired_bootstrap_delta_ci` on the same prompts, which is what the
    compass already uses to say whether an AUC moved, so this asks the existing instrument a new
    question rather than inventing a second one.
    """
    from . import margin as margin_mod

    for name, rows in (("transformers harmful", hf_harmful), ("transformers harmless", hf_benign),
                       ("gguf harmful", gguf_harmful), ("gguf harmless", gguf_benign)):
        if any(m is None for m in rows):
            raise GgufCompassError(
                f"the {name} arm carries a prompt with no readable verdict, so an agreement "
                f"figure over these rows would be computed on a different set per arm. Drop the "
                f"unreadable prompts from both arms, or report neither.")
    if len(hf_harmful) != len(gguf_harmful) or len(hf_benign) != len(gguf_benign):
        raise GgufCompassError(
            f"the two arms hold {len(hf_harmful)}/{len(hf_benign)} and "
            f"{len(gguf_harmful)}/{len(gguf_benign)} prompts. A paired comparison needs the same "
            f"prompts in the same order in both.")
    hf_auc = margin_mod.auc(hf_harmful, hf_benign)
    gg_auc = margin_mod.auc(gguf_harmful, gguf_benign)
    delta = margin_mod.paired_bootstrap_delta_ci(
        (hf_harmful, hf_benign), (gguf_harmful, gguf_benign), seed=seed)
    n = len(hf_harmful) + len(hf_benign)
    reasons = []
    if n < metrics.MIN_REPORTABLE_N:
        reasons.append(
            f"{n} prompts is below the {metrics.MIN_REPORTABLE_N} this project will state a rate "
            f"on at all, so this run cannot say whether the two compasses agree")
    # `delta_crosses_zero` and `delta_ci` are the keys `margin.paired_bootstrap_delta_ci`
    # actually returns. Reading a key that does not exist is how a control comes to pass
    # everything: the first version of this looked for `ci`, found nothing, and called two
    # compasses that ordered the prompts in opposite directions comparable. Found by the test
    # below rather than by a run.
    if delta and delta.get("delta_crosses_zero") is False:
        reasons.append(
            f"the paired interval on the AUC difference is {list(delta['delta_ci'])} and does "
            f"not contain zero, so the two paths order these prompts differently")
    return {
        "n": n,
        "transformers_auc": None if hf_auc is None else round(hf_auc, 4),
        "gguf_auc": None if gg_auc is None else round(gg_auc, 4),
        "auc_difference": (None if None in (hf_auc, gg_auc) else round(gg_auc - hf_auc, 4)),
        "paired_delta": delta,
        "comparable": not reasons,
        "why_not": reasons,
    }
