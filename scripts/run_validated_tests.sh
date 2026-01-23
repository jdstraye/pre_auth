#!/usr/bin/env bash
set -euo pipefail

# Runs validated ground-truth tests for a preset user list, then runs the rest of
# the pytest suite (excluding the ground-truth param test) and summarizes results.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VALIDATED_USERS=(1140 584 582 692 705 1314)
GT_TEST="tests/test_pdf_extraction_ground_truth.py"

echo "== Test run started: $(date) =="

gt_failures=0
for id in "${VALIDATED_USERS[@]}"; do
  echo
  echo "--- Running GT for user_$id ---"
  if pytest -q "$GT_TEST" --user_id="$id"; then
    echo "user_$id: PASS"
  else
    echo "user_$id: FAIL"
    gt_failures=$((gt_failures+1))
  fi
done

echo
if [ "$gt_failures" -gt 0 ]; then
  echo "Ground-truth failures: $gt_failures"
else
  echo "All validated ground-truth tests passed."
fi

echo
echo "--- Running non-GT pytest suite ---"
# Run remaining test suite excluding the ground-truth test
if pytest -q -k "not test_pdf_extraction_vs_ground_truth" --maxfail=1; then
  echo "Non-GT tests: PASS"
  non_gt_failures=0
else
  echo "Non-GT tests: FAIL (see pytest output for details)"
  non_gt_failures=1
fi

echo
# Final summary
if [ "$gt_failures" -gt 0 ] || [ "$non_gt_failures" -ne 0 ]; then
  echo "== Test run completed: FAIL =="
  echo "GT failures: $gt_failures, Non-GT failures: $non_gt_failures"
  exit 1
else
  echo "== Test run completed: ALL PASS =="
  exit 0
fi
