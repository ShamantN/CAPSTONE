---
name: Bug Report
about: Report a reproducible bug on a DEV split or setup failure
title: "[BUG] "
labels: bug
---

## Environment
- OS & GPU (`python tools/verify_setup.py` summary):
- Git Commit / Branch:

## Steps to Reproduce (`DEV` Splits Only)
*(Confirm you did NOT set `ALLOW_TEST2=1` or use reserved seeds `{606, 707, 808}`.)*
1.
2.

## Expected vs Observed Behavior

## Verification Gate Checks
- [ ] `python tools/check_freeze.py` exits `0`
- [ ] `python tools/audit_test2_isolation.py` exits `0`

