# Build the Docs
`sphinx` provides a Makefile, so to build the `html` documentation, simply type:
```
make html
```

Or, alternatively
```
SPHINXBUILD="python3 -m sphinx.cmd.build" make html
```

To build docs without running python files (tutorials) use the `html-noplot` rule.

You can browse the docs by opening the generated html files in the `docs/build` directory
with a browser of your choice.