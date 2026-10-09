// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

//! Hijacking signatures over a model namespace's commit timeline.
//!
//! WHAT THIS IS FOR. A compromised uploader account is the cheap way to put a hostile checkpoint
//! somewhere people already trust, and it leaves traces in the commit history that cost nothing to
//! read: a namespace that only ever shipped safetensors suddenly pushing a pickle, an account
//! dormant for two years waking up, a different author on the newest commit. None of these is
//! proof of anything. Together they are a reason to look harder before downloading several
//! gigabytes of weights, which is the whole point: the verdict has to arrive BEFORE the download,
//! or it has not saved anybody anything.
//!
//! PORTED, not invented, from a sister project's private crate on 2026-10-09, where the same seven
//! signatures were written and calibrated. Two differences, both deliberate:
//!
//!   1. **All seven fire here. Two of them never fired there.** `format-tier-regression` and
//!      `first-ever-onnx-after-pytorch-only` were declared in that crate's signature list and
//!      never emitted by its evaluator, so a reader of the list believed in two checks that did
//!      not exist. Both are implemented below from data the evaluator already had in hand.
//!   2. **No calendar dependency.** The requirement is the number of days between two RFC 3339
//!      timestamps, and that is exact civil-date arithmetic rather than a library.
//!
//! TRUNCATION DISCIPLINE, which is the part that matters most and is carried across unchanged.
//! When the timeline is a bounded slice rather than the whole history, every "first ever X"
//! signature is suppressed, because a partial view cannot honestly claim a first. The signatures
//! that survive truncation are the ones observable from any slice: a gap between two kept
//! commits, a change of author between them, a burst within a day. Reporting "first ever pickle"
//! off a window that starts after the first pickle is exactly the shape of wrong this project
//! keeps withdrawing numbers over.

use std::collections::BTreeSet;

/// How much a signature firing should worry a reader. Ordered, so a caller can take the worst.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Severity {
    Low,
    Medium,
    High,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Low => "low",
            Severity::Medium => "medium",
            Severity::High => "high",
        }
    }
}

/// One commit in a namespace's history, as much of it as the caller could see.
pub struct Commit {
    /// RFC 3339. Kept as text because that is how it arrives and because a timestamp this cannot
    /// parse must be skipped rather than guessed at.
    pub date: String,
    /// The uploader on this commit, when the host says.
    pub author: Option<String>,
    /// Every file extension present in the tree at this commit, lowercase and without the dot.
    pub extensions: Vec<String>,
}

/// A namespace's timeline, and whether it is all of it.
pub struct Snapshot {
    pub namespace: String,
    pub commits: Vec<Commit>,
    /// True when `commits` is a bounded slice of a longer history. See the truncation note above.
    pub truncated: bool,
}

/// One signature that fired, with the evidence that made it fire.
pub struct Observation {
    pub signature: &'static str,
    pub severity: Severity,
    pub observed_at: String,
    /// WHY it fired, in a sentence a user could check. Never an accusation about a person: it
    /// describes what the commit history shows, which is a record rather than a character.
    pub basis: String,
}

/// The seven signatures, as a catalogue a caller can show without evaluating anything.
pub fn signatures() -> &'static [(&'static str, Severity, &'static str)] {
    &[
        ("first-ever-pickle", Severity::High,
         "the namespace only ever shipped safetensors-tier formats, and the newest commit adds a \
          pickle-family file (.bin, .pt, .pth, .pkl)"),
        ("long-dormant-then-active", Severity::Medium,
         "the namespace had no commits for over 18 months and has just pushed one"),
        ("format-tier-regression", Severity::Medium,
         "the namespace only ever shipped .safetensors and is now pushing a legacy format"),
        ("first-ever-h5", Severity::Low,
         "the namespace's first ever .h5 (legacy Keras) push"),
        ("first-ever-onnx-after-pytorch-only", Severity::Low,
         "the namespace only ever shipped PyTorch, and the newest commit introduces .onnx"),
        ("author-username-change", Severity::Low,
         "the namespace had a single author until the newest commit, which has a different one"),
        ("rapid-fire-after-dormancy", Severity::Low,
         "more than five commits in 24 hours on a namespace whose typical gap is months"),
    ]
}

const PICKLE_FAMILY: [&str; 4] = ["bin", "pt", "pth", "pkl"];
const LEGACY_FORMATS: [&str; 3] = ["bin", "pt", "h5"];

/// Days from the civil date 1970-01-01. Hinnant's algorithm, exact for every proleptic Gregorian
/// date, which a subtraction of year numbers is not: it has to get leap years and centuries right
/// or an 18-month threshold drifts by days near a century boundary.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Seconds since the epoch from an RFC 3339 timestamp, or None when it cannot be read.
///
/// None rather than a default, deliberately. A commit whose date this cannot parse is dropped
/// from the date-based signatures rather than being treated as the epoch, which would place it
/// fifty years before everything else and make a dormancy gap out of a formatting problem.
pub fn parse_rfc3339(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let num = |from: usize, to: usize| -> Option<i64> {
        let part = s.get(from..to)?;
        if !part.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        part.parse().ok()
    };
    if b[4] != b'-' || b[7] != b'-' || (b[10] != b'T' && b[10] != b't' && b[10] != b' ') {
        return None;
    }
    if b[13] != b':' || b[16] != b':' {
        return None;
    }
    let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    let (h, mi, sec) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || sec > 60 {
        return None;
    }
    let mut epoch = days_from_civil(y, mo, d) * 86_400 + h * 3_600 + mi * 60 + sec;
    // The offset, when there is one. A timestamp that says +05:30 and is read as UTC is wrong by
    // five and a half hours, which matters to a 24 hour window and to nothing else here.
    let rest = &s[19..];
    let tz = rest.trim_start_matches(|c: char| c == '.' || c.is_ascii_digit());
    if let Some(sign) = tz.as_bytes().first() {
        if *sign == b'+' || *sign == b'-' {
            let oh: i64 = tz.get(1..3)?.parse().ok()?;
            let om: i64 = tz.get(4..6).unwrap_or("00").parse().unwrap_or(0);
            let delta = oh * 3_600 + om * 60;
            epoch += if *sign == b'+' { -delta } else { delta };
        }
    }
    Some(epoch)
}

/// Every signature that fires on this snapshot, in catalogue order.
pub fn evaluate(snap: &Snapshot) -> Vec<Observation> {
    let mut out = Vec::new();
    if snap.commits.len() < 2 {
        // One commit has no history to compare against, and zero has nothing at all. Returning
        // early rather than letting the loops below read an empty baseline as "never shipped a
        // pickle", which would fire `first-ever-pickle` on every brand new namespace.
        return out;
    }
    let last = snap.commits.last().expect("len >= 2");
    let baseline = &snap.commits[..snap.commits.len() - 1];

    let ext_set = |cs: &[Commit]| -> BTreeSet<String> {
        cs.iter()
            .flat_map(|c| c.extensions.iter().map(|e| e.to_ascii_lowercase()))
            .collect()
    };
    let before = ext_set(baseline);
    let now: BTreeSet<String> = last.extensions.iter().map(|e| e.to_ascii_lowercase()).collect();
    let newly = |e: &str| now.contains(e) && !before.contains(e);

    // ── the "first ever" family, all suppressed under truncation ──────────────────────
    if !snap.truncated {
        let new_pickles: Vec<&str> = PICKLE_FAMILY.iter().copied().filter(|e| newly(e)).collect();
        if !new_pickles.is_empty() {
            out.push(Observation {
                signature: "first-ever-pickle",
                severity: Severity::High,
                observed_at: last.date.clone(),
                basis: format!(
                    "the newest commit is the first on this namespace to carry a pickle-family \
                     extension ({})",
                    new_pickles.join(", ")
                ),
            });
        }
        if newly("h5") {
            out.push(Observation {
                signature: "first-ever-h5",
                severity: Severity::Low,
                observed_at: last.date.clone(),
                basis: "the newest commit is the first on this namespace to carry .h5".to_string(),
            });
        }
        // IMPLEMENTED HERE AND DEAD IN THE ORIGINAL. The namespace shipped PyTorch weights and
        // nothing else, and the newest commit adds ONNX. On its own that is a perfectly ordinary
        // thing to do, which is why it is Low: it is only interesting beside the others.
        let pytorch_only = before.iter().any(|e| e == "bin" || e == "pt" || e == "pth")
            && !before.contains("onnx")
            && !before.contains("safetensors");
        if pytorch_only && newly("onnx") {
            out.push(Observation {
                signature: "first-ever-onnx-after-pytorch-only",
                severity: Severity::Low,
                observed_at: last.date.clone(),
                basis: "the namespace had shipped only PyTorch weights, and the newest commit \
                        introduces .onnx"
                    .to_string(),
            });
        }
        // ALSO IMPLEMENTED HERE AND DEAD IN THE ORIGINAL. Distinct from first-ever-pickle: this
        // one is about a namespace that had moved to safetensors exclusively and has gone back,
        // which is a direction of travel rather than a first appearance.
        if before.contains("safetensors") && before.len() == 1 {
            let regressed: Vec<&str> =
                LEGACY_FORMATS.iter().copied().filter(|e| now.contains(*e)).collect();
            if !regressed.is_empty() {
                out.push(Observation {
                    signature: "format-tier-regression",
                    severity: Severity::Medium,
                    observed_at: last.date.clone(),
                    basis: format!(
                        "the namespace had shipped only .safetensors, and the newest commit \
                         carries {}",
                        regressed.join(", ")
                    ),
                });
            }
        }
    }

    // ── the signatures that survive truncation ────────────────────────────────────────
    if let (Some(prev), Some(prev_t), Some(last_t)) = (
        baseline.last(),
        baseline.last().and_then(|c| parse_rfc3339(&c.date)),
        parse_rfc3339(&last.date),
    ) {
        let _ = prev;
        let gap_days = (last_t - prev_t) / 86_400;
        if gap_days > 18 * 30 {
            out.push(Observation {
                signature: "long-dormant-then-active",
                severity: Severity::Medium,
                observed_at: last.date.clone(),
                basis: format!(
                    "{gap_days} days between the last two commits, which is over 18 months"
                ),
            });
        }
    }

    if baseline.len() >= 2 {
        let authors: BTreeSet<&str> =
            baseline.iter().filter_map(|c| c.author.as_deref()).collect();
        if authors.len() == 1 {
            let was = authors.into_iter().next().expect("len == 1");
            if let Some(now_author) = last.author.as_deref() {
                if now_author != was {
                    out.push(Observation {
                        signature: "author-username-change",
                        severity: Severity::Low,
                        observed_at: last.date.clone(),
                        // The two names are the evidence and they are the namespace's own public
                        // commit metadata, not a judgement about either person.
                        basis: format!(
                            "every earlier commit here is by '{was}'; the newest is by \
                             '{now_author}'"
                        ),
                    });
                }
            }
        }
    }

    if snap.commits.len() >= 6 {
        if let Some(last_t) = parse_rfc3339(&last.date) {
            let window = last_t - 24 * 3_600;
            let recent = snap
                .commits
                .iter()
                .filter_map(|c| parse_rfc3339(&c.date))
                .filter(|t| *t >= window)
                .count();
            if recent > 5 {
                // THE MEDIAN IS TAKEN OVER THE HISTORY BEFORE THE BURST, AND THIS IS A FIX.
                //
                // The original computed it over every commit including the burst, which cannot
                // work: six commits inside a day contribute five gaps of zero days, and five
                // zeroes outvote three gaps of a year. So the median came out at zero, the
                // "previously low frequency" test compared zero against ninety days and failed,
                // and the signature could not fire on the exact shape it was written for. Found
                // 2026-10-09 by porting it and writing the test the description implies.
                //
                // The word the implementation dropped is "previously". A burst is only
                // interesting against the cadence that came before it, so the gaps are taken
                // between commits that both predate the window. Fewer than two such commits
                // means there is no prior cadence to compare against, and the signature stays
                // quiet rather than guessing one.
                let before: Vec<i64> = snap
                    .commits
                    .iter()
                    .filter_map(|c| parse_rfc3339(&c.date))
                    .filter(|t| *t < window)
                    .collect();
                let mut gaps: Vec<i64> =
                    before.windows(2).map(|w| (w[1] - w[0]) / 86_400).collect();
                if !gaps.is_empty() {
                    gaps.sort_unstable();
                    let median = gaps[gaps.len() / 2];
                    if median > 90 {
                        out.push(Observation {
                            signature: "rapid-fire-after-dormancy",
                            severity: Severity::Low,
                            observed_at: last.date.clone(),
                            basis: format!(
                                "{recent} commits in the last 24 hours, against a typical gap of \
                                 {median} days"
                            ),
                        });
                    }
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(date: &str, author: Option<&str>, exts: &[&str]) -> Commit {
        Commit {
            date: date.to_string(),
            author: author.map(str::to_string),
            extensions: exts.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn snap(commits: Vec<Commit>, truncated: bool) -> Snapshot {
        Snapshot { namespace: "org/model".to_string(), commits, truncated }
    }

    fn fired(obs: &[Observation]) -> Vec<&str> {
        obs.iter().map(|o| o.signature).collect()
    }

    #[test]
    fn days_from_civil_is_exact_on_known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        // 2000 is a leap year and 1900 is not, which is the case a naive formula gets wrong.
        assert_eq!(days_from_civil(2000, 3, 1) - days_from_civil(2000, 2, 28), 2);
        assert_eq!(days_from_civil(1900, 3, 1) - days_from_civil(1900, 2, 28), 1);
    }

    #[test]
    fn rfc3339_parses_the_shapes_a_host_actually_sends() {
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_rfc3339("1970-01-02T00:00:00Z"), Some(86_400));
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00.123Z"), Some(0));
        // An offset is applied, not ignored: +05:30 means this instant is earlier in UTC.
        assert_eq!(parse_rfc3339("1970-01-01T05:30:00+05:30"), Some(0));
        assert_eq!(parse_rfc3339("1969-12-31T18:30:00-05:30"), Some(0));
    }

    #[test]
    fn an_unparseable_date_is_none_rather_than_the_epoch() {
        // The failure this prevents: a malformed date read as 1970 turns a formatting problem
        // into a fifty year dormancy gap.
        for bad in ["", "not a date", "2026-13-01T00:00:00Z", "2026-10-09", "2026-10-09T99:00:00Z"]
        {
            assert_eq!(parse_rfc3339(bad), None, "{bad:?} parsed when it should not");
        }
    }

    #[test]
    fn a_namespace_with_one_commit_fires_nothing() {
        // The bug this guards: an empty baseline reads as "never shipped a pickle", so every new
        // namespace that opens with a .bin would be reported as a first-ever-pickle event.
        let s = snap(vec![c("2026-10-09T00:00:00Z", Some("a"), &["bin"])], false);
        assert!(evaluate(&s).is_empty());
    }

    #[test]
    fn a_first_pickle_after_safetensors_only_is_high_severity() {
        let s = snap(
            vec![
                c("2026-01-01T00:00:00Z", Some("a"), &["safetensors"]),
                c("2026-10-09T00:00:00Z", Some("a"), &["safetensors", "bin"]),
            ],
            false,
        );
        let obs = evaluate(&s);
        assert!(fired(&obs).contains(&"first-ever-pickle"));
        let hit = obs.iter().find(|o| o.signature == "first-ever-pickle").unwrap();
        assert_eq!(hit.severity, Severity::High);
        assert!(hit.basis.contains("bin"));
    }

    #[test]
    fn truncation_suppresses_every_first_ever_claim() {
        // THE DISCIPLINE THAT MATTERS MOST. A bounded window cannot see a first, and claiming one
        // off a slice that starts after the real first is a statement about the window.
        let commits = vec![
            c("2026-01-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2026-10-09T00:00:00Z", Some("a"), &["safetensors", "bin", "h5", "onnx"]),
        ];
        let whole = evaluate(&snap(commits, false));
        assert!(fired(&whole).iter().any(|s| s.starts_with("first-ever")));

        let commits = vec![
            c("2026-01-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2026-10-09T00:00:00Z", Some("a"), &["safetensors", "bin", "h5", "onnx"]),
        ];
        let sliced = evaluate(&snap(commits, true));
        assert!(
            !fired(&sliced).iter().any(|s| s.starts_with("first-ever")),
            "a truncated timeline claimed a first: {:?}",
            fired(&sliced)
        );
    }

    #[test]
    fn dormancy_survives_truncation_because_the_kept_dates_are_ground_truth() {
        let commits = vec![
            c("2023-01-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2026-10-09T00:00:00Z", Some("a"), &["safetensors"]),
        ];
        assert!(fired(&evaluate(&snap(commits, true))).contains(&"long-dormant-then-active"));
    }

    #[test]
    fn a_gap_just_under_the_threshold_does_not_fire() {
        let commits = vec![
            c("2026-01-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2026-06-01T00:00:00Z", Some("a"), &["safetensors"]),
        ];
        assert!(!fired(&evaluate(&snap(commits, false))).contains(&"long-dormant-then-active"));
    }

    #[test]
    fn the_two_signatures_that_never_fired_upstream_fire_here() {
        // format-tier-regression: safetensors only, then a legacy format.
        let commits = vec![
            c("2026-01-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2026-02-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2026-03-01T00:00:00Z", Some("a"), &["safetensors", "h5"]),
        ];
        assert!(fired(&evaluate(&snap(commits, false))).contains(&"format-tier-regression"));

        // first-ever-onnx-after-pytorch-only: pytorch only, then onnx.
        let commits = vec![
            c("2026-01-01T00:00:00Z", Some("a"), &["bin"]),
            c("2026-02-01T00:00:00Z", Some("a"), &["bin", "onnx"]),
        ];
        assert!(fired(&evaluate(&snap(commits, false)))
            .contains(&"first-ever-onnx-after-pytorch-only"));
    }

    #[test]
    fn an_author_change_needs_a_settled_single_author_before_it() {
        let commits = vec![
            c("2026-01-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2026-02-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2026-03-01T00:00:00Z", Some("b"), &["safetensors"]),
        ];
        assert!(fired(&evaluate(&snap(commits, false))).contains(&"author-username-change"));

        // Two authors already in the history is a shared namespace, not a takeover.
        let commits = vec![
            c("2026-01-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2026-02-01T00:00:00Z", Some("b"), &["safetensors"]),
            c("2026-03-01T00:00:00Z", Some("c"), &["safetensors"]),
        ];
        assert!(!fired(&evaluate(&snap(commits, false))).contains(&"author-username-change"));
    }

    #[test]
    fn a_burst_only_counts_against_a_namespace_that_was_slow() {
        let mut commits = vec![
            c("2020-01-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2021-01-01T00:00:00Z", Some("a"), &["safetensors"]),
            c("2022-01-01T00:00:00Z", Some("a"), &["safetensors"]),
        ];
        for h in 0..6 {
            commits.push(c(&format!("2026-10-09T0{h}:00:00Z"), Some("a"), &["safetensors"]));
        }
        assert!(fired(&evaluate(&snap(commits, false))).contains(&"rapid-fire-after-dormancy"));

        // A busy namespace pushing six times in a day is just a busy namespace.
        let mut commits = Vec::new();
        for d in 1..=6 {
            commits.push(c(&format!("2026-10-0{d}T00:00:00Z"), Some("a"), &["safetensors"]));
        }
        for h in 0..6 {
            commits.push(c(&format!("2026-10-09T0{h}:00:00Z"), Some("a"), &["safetensors"]));
        }
        assert!(!fired(&evaluate(&snap(commits, false))).contains(&"rapid-fire-after-dormancy"));
    }

    #[test]
    fn every_catalogued_signature_can_actually_fire() {
        // THE TEST THAT EXISTS BECAUSE THE ORIGINAL FAILED IT. Two of its seven signatures were
        // declared and never emitted, so the catalogue promised checks that did not run. This
        // asserts the catalogue and the evaluator agree, by driving a case for each.
        let mut seen: BTreeSet<&str> = BTreeSet::new();

        let cases: Vec<Snapshot> = vec![
            snap(
                vec![
                    c("2026-01-01T00:00:00Z", Some("a"), &["safetensors"]),
                    c("2026-02-01T00:00:00Z", Some("a"), &["safetensors"]),
                    c("2026-03-01T00:00:00Z", Some("a"), &["safetensors", "bin", "h5"]),
                ],
                false,
            ),
            snap(
                vec![
                    c("2026-01-01T00:00:00Z", Some("a"), &["bin"]),
                    c("2026-02-01T00:00:00Z", Some("a"), &["bin", "onnx"]),
                ],
                false,
            ),
            snap(
                vec![
                    c("2023-01-01T00:00:00Z", Some("a"), &["safetensors"]),
                    c("2026-10-09T00:00:00Z", Some("a"), &["safetensors"]),
                ],
                false,
            ),
            snap(
                vec![
                    c("2026-01-01T00:00:00Z", Some("a"), &["safetensors"]),
                    c("2026-02-01T00:00:00Z", Some("a"), &["safetensors"]),
                    c("2026-03-01T00:00:00Z", Some("b"), &["safetensors"]),
                ],
                false,
            ),
            {
                let mut cs = vec![
                    c("2020-01-01T00:00:00Z", Some("a"), &["safetensors"]),
                    c("2021-01-01T00:00:00Z", Some("a"), &["safetensors"]),
                    c("2022-01-01T00:00:00Z", Some("a"), &["safetensors"]),
                ];
                for h in 0..6 {
                    cs.push(c(&format!("2026-10-09T0{h}:00:00Z"), Some("a"), &["safetensors"]));
                }
                snap(cs, false)
            },
        ];
        for s in &cases {
            for o in evaluate(s) {
                seen.insert(o.signature);
            }
        }
        for (id, _, _) in signatures() {
            assert!(seen.contains(id), "catalogued signature {id} never fired in any case");
        }
    }
}
