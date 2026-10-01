#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King
#
# Report that a check DID NOT RUN, and decide from one flag whether that is yellow or red.
#
# THE PROBLEM THIS EXISTS FOR
#
# A check has three outcomes, never two: PASS, FAIL, and DID NOT RUN. GitHub gives a plain
# workflow job four conclusions (success, failure, cancelled, skipped) and no way to emit a
# neutral one, so the third outcome has to borrow one of the other two. Today it borrows success
# and says so loudly: a `::warning::` annotation and a block in the run summary, on a green job.
# That is the right trade while the only thing reading the conclusion is a person, because a gate
# that goes red when the network hiccups teaches everybody to re-run until it is green, and a red
# build that means nothing is worse than no build.
#
# It stops being the right trade the day something other than a person reads the conclusion. A
# required status check on a branch reads green as "this was verified", and a network blip would
# then let a merge through on a check that verified nothing. That day is a settings change on the
# operator's GitHub account, not a change in this tree, so Q-71 put the switch here and left it
# off: one flag, flipped once, at the moment branch protection lands.
#
# USAGE
#
#   tools/ci/did_not_run.sh <tool> <what was not checked>
#
#   if ! tools/ci/did_not_run.sh actionlint "the workflows were NOT linted"; then exit 1; fi
#
# Exit 0 means "reported, carry on"; exit 1 means "reported, and the caller must fail".
#
# WHY AN UNSET FLAG IS AN ERROR RATHER THAN A DEFAULT
#
# Defaulting a safety switch to off when nobody set it is how a switch stops being one: a caller
# that forgets to pass the environment through gets the permissive behaviour and no indication
# that it chose it. So an unset or unrecognised value is refused, loudly, and the refusal names
# the variable. A wiring mistake then looks like a wiring mistake rather than like a pass.

set -uo pipefail

FLAG_NAME="REQUIRE_EVERY_CHECK_TO_RUN"

if [ "$#" -ne 2 ]; then
  echo "::error::did_not_run.sh takes exactly two arguments, the tool and what was not checked; got $#." >&2
  exit 1
fi

tool="$1"
what="$2"

if [ -z "${tool}" ] || [ -z "${what}" ]; then
  echo "::error::did_not_run.sh was given an empty tool name or an empty description, so the annotation it produced would not tell a reader what failed to run." >&2
  exit 1
fi

required="${!FLAG_NAME-}"
case "${required}" in
  true|false) ;;
  "")
    echo "::error::${FLAG_NAME} is not set, so this job cannot tell whether a check that did not run should fail the build. Set it in the workflow's top-level env. It is not defaulted here on purpose: a safety switch that quietly defaults to off is not a switch." >&2
    exit 1
    ;;
  *)
    echo "::error::${FLAG_NAME} is '${required}', which is neither 'true' nor 'false'. Refusing to guess which was meant." >&2
    exit 1
    ;;
esac

# The same words either way, because the FACT is the same fact and only its consequence changes.
# A reader comparing two runs should not have to work out whether the wording shifted or the
# finding did.
sentence="DID NOT RUN: ${tool} did not run, so ${what}. This is not a pass."

if [ "${required}" = "true" ]; then
  echo "::error::${sentence} ${FLAG_NAME} is on, so the job fails rather than reporting green, because something other than a person is reading this conclusion."
else
  echo "::warning::${sentence} The job stays green because ${FLAG_NAME} is off; read the annotation rather than the tick."
fi

if [ -n "${GITHUB_STEP_SUMMARY-}" ]; then
  {
    echo "### DID NOT RUN: ${tool}"
    echo ""
    echo "${what}"
    echo ""
    if [ "${required}" = "true" ]; then
      echo "This **fails** the job: \`${FLAG_NAME}\` is on, so a check that could not run is"
      echo "reported as a failure rather than as a green tick a machine would read as a pass."
    else
      echo "This job is **green and did not check anything**. \`${FLAG_NAME}\` is off, which is"
      echo "correct while a person reads this run; it becomes a hole the day a required status"
      echo "check reads the conclusion instead."
    fi
  } >> "${GITHUB_STEP_SUMMARY}"
fi

[ "${required}" = "true" ] && exit 1
exit 0
