// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

//! The compiled half of senbonzakura.
//!
//! WHAT BELONGS HERE, AND WHAT DOES NOT. Rust is here for two things and should not spread past
//! them. The first is parsers over input an attacker chose, where a bounded, memory-safe reader is
//! the requirement rather than the optimisation: section 2.1 asks for hard caps on recursion,
//! decompressed size and per-item memory, and those are cheaper to state and keep in a type system
//! that enforces them. The second is hot walks over large files, where Python's per-item overhead
//! is the cost rather than the I/O.
//!
//! It is NOT here for network code, for orchestration, or for anything whose time is spent waiting.
//! The rented-pod driver is Python for that reason: it is HTTP and JSON, and a compiled extension
//! would add build complexity to a path that spends its life blocked on a socket.
//!
//! WHY THE EXTENSION IS REQUIRED RATHER THAN OPTIONAL. Operator decision, 2026-10-09: senbonzakura
//! is one package that a user installs once and then has everything on their own machine, with
//! paid tiers gating ACCESS rather than gating code. A fallback implementation in Python would
//! mean two copies of one logic, and three copies of prompt rendering once drifted far enough to
//! put the compass's reading on the wrong token. One implementation, compiled, is the safer shape
//! even though it costs a wheel per platform.
//!
//! `abi3-py310` is what keeps that cost flat: one wheel per platform covers 3.10 and everything
//! after, which is the same cardinality `setup.py` already produces for the vendored binaries.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub mod timeline;

/// The signature catalogue, as a list of dicts. Callable without evaluating anything, so a caller
/// can show what is checked for without having a model in hand.
#[pyfunction]
fn signatures(py: Python<'_>) -> PyResult<Py<PyList>> {
    let out = PyList::empty(py);
    for (id, severity, description) in timeline::signatures() {
        let d = PyDict::new(py);
        d.set_item("id", id)?;
        d.set_item("severity", severity.as_str())?;
        d.set_item("description", description)?;
        out.append(d)?;
    }
    Ok(out.unbind())
}

/// RFC 3339 to seconds since the epoch, or None. Exposed because the Python side needs the same
/// answer the signatures were computed with, and two parsers would be two answers.
#[pyfunction]
fn parse_timestamp(text: &str) -> Option<i64> {
    timeline::parse_rfc3339(text)
}

/// Every hijacking signature that fires on a namespace's commit timeline.
///
/// `commits` is a list of dicts, oldest first, each with `date` (RFC 3339), optional `author`, and
/// `extensions` (a list of file extensions present in the tree at that commit, without the dot).
/// `truncated` says whether the list is a bounded slice of a longer history, and it is REQUIRED
/// rather than defaulted: a caller who does not know whether their view is complete must say so,
/// because every "first ever" claim depends on the answer and defaulting it to false would turn an
/// unknown into a confident assertion.
#[pyfunction]
#[pyo3(signature = (namespace, commits, truncated))]
fn evaluate_timeline(
    py: Python<'_>,
    namespace: &str,
    commits: &Bound<'_, PyList>,
    truncated: bool,
) -> PyResult<Py<PyList>> {
    let mut parsed = Vec::with_capacity(commits.len());
    for (i, item) in commits.iter().enumerate() {
        let d = item.cast::<PyDict>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "commits[{i}] is not a dict. Each commit is {{'date': ..., 'author': ..., \
                 'extensions': [...]}}"
            ))
        })?;
        let date: String = match d.get_item("date")? {
            Some(v) => v.extract()?,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "commits[{i}] has no 'date'. A commit with no date cannot be placed in the \
                     timeline, and guessing one would invent a dormancy gap or hide one"
                )))
            }
        };
        let author: Option<String> = match d.get_item("author")? {
            Some(v) if !v.is_none() => Some(v.extract()?),
            _ => None,
        };
        let extensions: Vec<String> = match d.get_item("extensions")? {
            Some(v) if !v.is_none() => v.extract()?,
            _ => Vec::new(),
        };
        parsed.push(timeline::Commit { date, author, extensions });
    }

    let snap = timeline::Snapshot {
        namespace: namespace.to_string(),
        commits: parsed,
        truncated,
    };
    let out = PyList::empty(py);
    for obs in timeline::evaluate(&snap) {
        let d = PyDict::new(py);
        d.set_item("signature", obs.signature)?;
        d.set_item("severity", obs.severity.as_str())?;
        d.set_item("observed_at", obs.observed_at)?;
        d.set_item("basis", obs.basis)?;
        d.set_item("namespace", &snap.namespace)?;
        out.append(d)?;
    }
    Ok(out.unbind())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Private by name and by statement. A Python-facing wrapper is where the refusals and the
    // plain-language errors belong; until it exists this says so rather than naming a module that
    // is not there, because a docstring pointing at nothing is worse than one pointing at itself.
    m.add("__doc__", "The compiled half of senbonzakura. Private: the leading underscore is the \
                      interface statement, and nothing outside the package should import it \
                      directly.")?;
    m.add_function(wrap_pyfunction!(signatures, m)?)?;
    m.add_function(wrap_pyfunction!(parse_timestamp, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_timeline, m)?)?;
    Ok(())
}
