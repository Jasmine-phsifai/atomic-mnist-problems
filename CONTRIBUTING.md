# Contributing

Keep additions atomic and evidence-driven.

1. One primary learning objective per problem.
2. At most two tightly coupled subtasks.
3. State the mathematical object and the engineering contract separately.
4. Name every explicit gap and every implicit gap exposed by the diagnostic.
5. Prefer exact invariants. If no conventional threshold exists, define a
   diagnostic variable whose interpretation is justified in the statement.
6. Do not add a reference solution to learner-facing files.
7. Keep Python comments and docstrings bilingual (English first, Chinese second).
8. Keep prose, test output, metadata, and LaTeX in English.
9. Add C/Cython only after a benchmark identifies a standalone bottleneck and
   the problem is specifically about that boundary.
10. Add a primary-source URL and an adaptation note.

Run `make validate` and `make pdfs` before committing.
