# Primary source map

The problems are paraphrased, separated, and made atomic from the following
primary materials. A problem's own statement lists its nearest source lineage.

## Textbook

- Dive into Deep Learning 1.0.3, data manipulation:
  <https://d2l.ai/chapter_preliminaries/ndarray.html>
- Dive into Deep Learning 1.0.3, probability and statistics:
  <https://d2l.ai/chapter_preliminaries/probability.html>
- Dive into Deep Learning 1.0.3, softmax regression:
  <https://d2l.ai/chapter_linear-classification/softmax-regression.html>
- Dive into Deep Learning 1.0.3, from-scratch softmax:
  <https://d2l.ai/chapter_linear-classification/softmax-regression-scratch.html>

Reference commit: `d2l-ai/d2l-en@23d7a5aecceee57d1292c56e90cce307f183bb0a`.

## Official numerical-library documentation

- NumPy broadcasting: <https://numpy.org/doc/stable/user/basics.broadcasting.html>
- NumPy copies and views: <https://numpy.org/doc/stable/user/basics.copies.html>
- NumPy NPY/NPZ format: <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>
- SciPy `logsumexp`: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html>
- SciPy `softmax`: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html>

## Official framework tutorials and APIs

- PyTorch Learn the Basics: <https://docs.pytorch.org/tutorials/beginner/basics/intro.html>
- PyTorch tensors: <https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html>
- PyTorch `torch.from_numpy`: <https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html>
- TensorFlow basic training loops: <https://www.tensorflow.org/guide/basic_training_loops>
- TensorFlow `tf.convert_to_tensor`: <https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor>

Reference commits:

- `pytorch/tutorials@4d66ab3f9226b7e712b725f9cf0f03fe5b0f1531`
- `tensorflow/docs@35e0922e059d7bc6d515a83e03a7494f0640c314`

## Stanford course material

- CS231n linear classification: <https://cs231n.github.io/linear-classify/>
- CS231n optimization and gradient checking: <https://cs231n.github.io/optimization-1/>
- CS231n assignments: <https://cs231n.stanford.edu/assignments.html>

The repository does not copy assignment solutions. It atomizes concepts such as
stable softmax, analytic/numerical gradient agreement, and batch computation.
