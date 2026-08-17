# Kannada-MNIST provenance and external-data boundary

Concrete examples use Kannada-MNIST, introduced by Vinay Uday Prabhu as a
drop-in handwritten-digit dataset with 60,000 training images, 10,000 in-domain
test images, and a separate 10,000-image Dig-MNIST distribution-shift set. Each
image is 28 x 28 grayscale and labels are in `{0, ..., 9}`.

Primary sources:

- Paper: <https://arxiv.org/abs/1908.01242>
- Upstream repository: <https://github.com/vinayprabhu/Kannada_MNIST>

The upstream repository has historically exposed more than one serialization
layout. This curriculum therefore teaches an explicit manifest/adapter boundary
instead of hiding filenames inside model code.

Expected local boundary:

```text
$MNIST_DATA_DIR/
  manifest.toml
  ... files named by the manifest ...
```

The official NPZ serializations are bundled in the repository at
`kannada-mnist-study/data/Kannada_MNIST_npz/` (Kannada-MNIST train/test and
Dig-MNIST), so no network download is required. The `$MNIST_DATA_DIR`
manifest layout above remains the taught boundary for data-loading problems:
contract tests still run on synthetic arrays that exercise the same shape,
dtype, label, and split contracts, independent of the bundled files.
