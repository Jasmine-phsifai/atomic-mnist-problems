# Repo Guide

NEVER FIX THE STARTER's TODO part!!!!!!! IT's FOR THE USER!

## What this repo is

A personal learning repo for atomic MNIST exercises — not production code. Mistakes are part of the record: buggy drafts and wrong turns get committed on purpose, with their diagnosis in the commit message.
all **starter.py**s' context **under ##TODO** is supposedly for the user to accomplish. When it's glitchy, it's the user's code problem worth recording, not yours to fix by the way. It shouldn't be viewed naturally as a "correct approach", and not a valid supportive evidence for explanation on other problems, only a reference, a user's approach.

## Learner-code ownership

- `starter.py` is learner-owned. Change its learner implementation only when
  the user supplies a new attempt for that file.
- Transcribe the supplied attempt without correcting its logic, API choices,
  messages, naming, or other glitches. Only normalize indentation, tabs, blank
  space, and blank lines when chat-box formatting makes that necessary for a
  faithful Python file.
- Commit and push learner attempts even when they are incomplete, inefficient,
  or failing. The Git history intentionally records errors; describe the
  observed behavior and test results factually in the commit message.
- Work directly on `main` unless the user explicitly requests another branch.
  Do not leave task branches locally or remotely after delivery.
- Fetch `origin` before editing and again immediately before pushing because
  the user also commits frequently. Fast-forward to the latest `origin/main`
  before starting; if it advances during the task, integrate it before push.
- Finish every Git task by pushing its atomic commits to `main`. Do not report
  the work as delivered while it exists only on a side branch.
- Benchmarks and LaTeX problem materials are instructor-owned and may be edited
  independently. Do not use that permission to silently repair `starter.py`.
- Prefer small, atomic commits and pushes so each learner attempt and each
  instructor-side benchmark or statement change remains independently visible.

## How to explain

- Combine angles: math, deep learning, numerics, engineering practice, security.
- Explain *why*, not just *how*. Rigorous derivations are welcome (see skill "course-math-proof").
- Back claims with runnable code examples rather than prose alone.
- Decompose big problems into small ones. Establish understanding of the concepts before writing code — do not rush to implementation.
- Do not skip steps, even ones that look trivial.
- Every assessment of a learner submission must walk through the submitted code
  row by row. Explain what each row or tightly connected block does, what is
  correct, what is defective or ambiguous, what input exposes it, and why.
  Assess like a teacher; a high-level verdict alone is insufficient.

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

## Explaining a single problem

Use this procedure when asked to explain one problem (e.g. "explain AP-01-004").
The goal is to make the problem understandable cold — including jargon, the
math, the contract, the tests, the benchmark, and what the implementation
actually needs to do.

### Steps

1. **Read all four files first**: `statement.tex`, `starter.py`,
   `test_problem.py`, `benchmark.py`. Do not explain from memory or from
   just one file.

2. **Structure the explanation** with these sections, in order:

   a. **Title & core idea** — one sentence that names the problem and states
      the central insight.

   b. **Noun / expression glossary** — a table of every technical term in the
      problem statement that a learner might stumble on: English term,
      Chinese equivalent, what it specifically means in this problem's
      context. Do not skip "obvious" terms.

   c. **Mathematical statement** — restate the formula in plain language,
      explain each symbol, and point out what the math *does not* specify
      (this is usually where the engineering gap lives).

   d. **Code problem / contract** — list every requirement the function must
      satisfy: input validation, output values, ownership/memory rules,
      error types. Quote the function signature.

   e. **Constraints** — what you are *not* allowed to do (banned methods,
      required library primitives, etc.).

   f. **Test-by-test walkthrough** — go through each test in
      `test_problem.py` and explain: what input it builds, what assertion it
      checks, and which contract clause it enforces. Name the test function
      so the reader can find it.

   g. **Benchmark walkthrough** — explain what `benchmark.py` does, why it
      is not a performance benchmark (when it isn't), and what a human
      reader is supposed to learn by running it. Point out trust-based
      evidence (things tests cannot machine-check).

   h. **Theory–engineering gap** — explain the gap in the problem's own
      words and what it means intuitively. This is the "why this problem
      exists" section.

   i. **Implementation direction** — describe at a high level what the
      function in `starter.py` needs to do and which NumPy primitives are
      relevant. Do not write the full solution unless asked.

3. **Correlate across files** — explicitly connect what `statement.tex`
   promises to what `test_problem.py` checks and what `benchmark.py`
   demonstrates. Show the mapping, don't assume the reader sees it.

4. **Keep starter.py and statement.tex as authority**. If something is
   ambiguous, say it's ambiguous and point to both files. Do not invent
   constraints that aren't written down.

## Concurrent agents

Multiple agent sessions may work in this repo at the same time. After any
force-push or history rewrite: `git fetch origin`, then
`git reset --hard origin/main`. Never merge-pull across rewritten history —
a merge resurrects the deleted content.
