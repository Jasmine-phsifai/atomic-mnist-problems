# Reference source manifest

Snapshot date: 2026-08-06

## PyTorch official tutorials

- Repository: <https://github.com/pytorch/tutorials>
- Commit: `4d66ab3f9226b7e712b725f9cf0f03fe5b0f1531`
- Online install selector: <https://pytorch.org/get-started/locally/>
- Selected path: `beginner_source/basics/`
- License file is copied beside the snapshot.

## TensorFlow official docs

- Repository: <https://github.com/tensorflow/docs>
- Commit: `35e0922e059d7bc6d515a83e03a7494f0640c314`
- Online installation guide: <https://www.tensorflow.org/install/pip>
- Selected documents: `tensor.ipynb`, `autodiff.ipynb`, `data.ipynb`,
  `quickstart-beginner.ipynb`, and `basic_training_loops.ipynb`.
- License file is copied beside the snapshot.

## Dive into Deep Learning

- Official repository: <https://github.com/d2l-ai/d2l-en>
- Repository commit used for licenses/version metadata:
  `23d7a5aecceee57d1292c56e90cce307f183bb0a`
- Official book site: <https://d2l.ai/>
- PDF: <https://d2l.ai/d2l-en.pdf>
- Release shown by the upstream book configuration: 1.0.3
- Book license: CC BY-SA 4.0
- Sample/reference code license: modified MIT (upstream `LICENSE-SAMPLECODE`)
- Local SHA-256: `b3129f44c4a26b534176bb0f83d85e1ed26a6cdf93ebebacb20104eab2d6bc00`
  (also recorded in `d2l/dive-into-deep-learning.pdf.sha256`)

The PDF is intentionally ignored by Git but included in the distributed working
copy. This avoids storing the same large binary once in the working tree and a
second time in `.git/objects`.
