# Repo Guide

## What this repo is

A personal learning repo for atomic MNIST exercises — not production code. Mistakes are part of the record: buggy drafts and wrong turns get committed on purpose, with their diagnosis in the commit message.

## How to explain

- Combine angles: math, deep learning, numerics, engineering practice, security.
- Explain *why*, not just *how*. Rigorous derivations are welcome (see skill "course-math-proof").
- Back claims with runnable code examples rather than prose alone.
- Decompose big problems into small ones. Establish understanding of the concepts before writing code — do not rush to implementation.
- Do not skip steps, even ones that look trivial.

## Git workflow

Run in this order after writing code:

1. `git status` and `git diff` — check for unexpected changes (surprise deletions, etc.).
2. `git pull`.
3. `python test_problem.py` — run the contract tests.
4. `python benchmark.py` — run the benchmark.
5. Commit — always, even broken code. The message must detail: implementation approach, problems encountered, test results, conclusion.
6. `git push`. Other sensible, non-sensitive local changes may be pushed too.

## Line endings

The intended `.gitattributes` config (create it if missing):

```
* text=auto eol=lf
*.npy binary
*.npz binary
*.png binary
*.jpg binary
*.pdf binary
*.zip binary
```

After editing `.gitattributes`, run `git add --renormalize .`.

## Before editing files

- Read the file before modifying it.
- automatic obey the git preference-adviced process without asking.

## Test-writing contract

A contract test verifies exactly the stated contract — no less, no more.

- Every requirement in `statement.tex` and every documented behavior in
  `starter.py` must be checked by `test_problem.py`. A missing check lets a
  wrong implementation pass.
- `test_problem.py` must not enforce anything absent from both
  `statement.tex` and `starter.py` — no hidden error types, tolerances,
  helper structures, or algorithm demands. If a constraint matters, write it
  into the statement first, then test it.
- Constraints that tests cannot enforce (e.g. "do not mutate global RNG
  state", "write the derivation as comments") are trust-based; the statement
  must present them as such.

## Concurrent agents

Multiple agent sessions may work in this repo at the same time. After any
force-push or history rewrite: `git fetch origin`, then
`git reset --hard origin/main`. Never merge-pull across rewritten history —
a merge resurrects the deleted content.
