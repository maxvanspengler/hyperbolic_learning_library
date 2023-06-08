# Build the Docs
`sphinx` provides a Makefile, so to build the `html` documentation, simply type:
```
PLOT_GALLERY=1 make html
```

Or, alternatively
```
SPHINXBUILD="python3 -m sphinx.cmd.build" PLOT_GALLERY=1 make html
```

To build docs without running python files (tutorials) use:
```
make html
```

You can browse the docs by opening the generated html files in the `docs/build` directory
with a browser of your choice.