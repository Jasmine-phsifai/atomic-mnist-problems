# Dive into Deep Learning

`dive-into-deep-learning.pdf` is the complete official English PDF, release 1.0.3,
downloaded from <https://d2l.ai/d2l-en.pdf>.

The book is licensed under CC BY-SA 4.0. Sample/reference code has the separate
modified MIT license supplied upstream. Both license texts are stored in this
directory.

The PDF is deliberately ignored by Git to keep future commits small. To restore or
refresh it in a fresh working tree:

```bash
curl -L https://d2l.ai/d2l-en.pdf \
  -o references/d2l/dive-into-deep-learning.pdf
sha256sum -c references/d2l/dive-into-deep-learning.pdf.sha256
```
