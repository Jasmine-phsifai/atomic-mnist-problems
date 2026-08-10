# Atomic MNIST Problems

This is a deliberately small, solution-free curriculum repository. It contains
only the first two chapters of a longer study path: **12 + 16 = 28 atomic code
problems**. The target is not merely to reproduce formulas. Each problem forces
an observable encounter with a gap between a mathematical object and its
engineering realization.

The learner is assumed to know linear algebra, numerical linear algebra,
mathematical analysis, probability, convex and non-convex optimization. No
prior knowledge of NumPy engineering, PyTorch, TensorFlow, deep learning APIs,
or training-loop conventions is assumed.

## Current scope

| Chapter | Problems | Theme |
|---|---:|---|
| 01 | 12 | Data contracts, memory semantics, splits, batching, and framework boundaries |
| 02 | 16 | Floating-point evidence, stable probability computations, gradients, and loss semantics |

This is **not** the future 128-problem bank and it does not attempt to cover an
entire MNIST training system.

## Repository contract

- One problem lives in one folder.
- `statement.tex` is the canonical statement.
- `problem.tex` inputs that statement and builds its individual PDF.
- The main book inputs the same `statement.tex`; the two statements therefore
  cannot drift.
- `starter.py` contains signatures, bilingual English/Chinese teaching comments,
  API hints, and `TODO` bodies. It contains no reference solution.
- `test_problem.py` is an executable contract test.
- `benchmark.py` prints or plots diagnostic evidence; it is not a speed contest
  unless the statement explicitly says so.
- Dataset files remain outside the repository. Tests create synthetic fixtures.
- No C or Cython is used in these chapters. The targeted work is either array
  semantics or short numerical kernels; a lower-level extension would obscure
  the lesson without a demonstrated bottleneck.

## Quick start

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/core-cpu.txt
make validate
make pdfs
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\core-cpu.txt
python tools\validate_scaffold.py
python tools\build_pdfs.py
```

`make validate` and `make pdfs` are thin wrappers around the scripts in
`tools/`. GNU make is optional: on Windows, call the scripts directly as
shown above. If make is installed, `make PYTHON=python validate` works too.

For a framework boundary problem, install one additional environment file:

```bash
python -m pip install -r requirements/pytorch-cpu.txt
# or
python -m pip install -r requirements/tensorflow-cpu.txt
```

Run one problem from its folder:

```bash
cd chapters/chapter_01_data_contracts/AP-01-001_npz_loader_contract
python test_problem.py
python benchmark.py
```

Before implementation, the test reports `UNIMPLEMENTED` and exits nonzero. That
is intentional. `make validate` checks the scaffold itself without pretending
the learner has solved the exercises.

## External Kannada-MNIST data

The exercises are dataset-adapter friendly and use Kannada-MNIST as the concrete
case. Keep the downloaded data elsewhere and expose it explicitly:

```bash
export MNIST_DATA_DIR=/absolute/path/to/Kannada_MNIST
```

```powershell
$env:MNIST_DATA_DIR = "C:\absolute\path\to\Kannada_MNIST"
```

No downloader is run implicitly. See `references/DATASET.md` for provenance and
the expected manifest boundary.

## Build outputs

`make pdfs` creates:

- `output/pdf/atomic-mnist-problems.pdf`
- one identically sourced PDF beside the catalog entry under
  `output/pdf/problems/`

Generated files are intentionally not tracked. The source tree remains the
maintainable artifact.

## Licensing

Original code scaffolds and tests are licensed under MIT (`LICENSE-CODE`).
Problem prose is licensed under CC BY-SA 4.0 (`LICENSE-CONTENT`). External
sources retain their own licenses; the statements paraphrase and cite them.
