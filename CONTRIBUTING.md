# Contributing to HypLL

Thank you for considering contributing to HypLL! We are always looking for new contributors to help us implement the constantly improving hyperbolic learning methodology and to maintain what is already here, so we are happy to have you!

We are looking for all sorts of contributions such as
<ul>
  <li>Bugfixes</li>
  <li>Documentation additions and improvements</li>
  <li>New features</li>
  <li>New tutorials</li>
</ul>

# Getting started
### Git workflow
We want any new changes to HypLL to be linked to GitHub issues. If you have something new in mind that is not yet mentioned in an issue, please create an issue detailing the intended change first. Once you have an issue in mind that you would like to contribute to, [fork the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

After you have finished your contribution, [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) on GitHub to have your contribution merged into our `main` branch. Note that your pull request will automatically be checked to see if it matches some of our standards. To avoid running into errors during your pull request, consider performing these tests locally first. A description on how to run the tests is given down below.

### Local setup
In order to get started locally with HypLL development, first clone the repository and make sure to install the repository with the correct optional dependencies from the `pyproject.toml` file. For example, if you only need the base development dependencies, navigate to the root directory (so the one containing this file) of your local version of this repository and install using:
```
pip install -e .[dev]
```
We recommend using a virtual environment tool such as `conda` or `venv`.

For further instructions, please read the section below that corresponds to the type of contribution that you intend to make.


# Contributing to development
**Optional dependencies:** If you are making development contributions, then you will at least need the `dev` optional dependencies. If your contribution warrants additional changes to the documentation, then the `docs` optional dependencies will likely also be of use for testing the documentation build process.

**Formatting:** For formatting we use [black](https://black.readthedocs.io) and [isort](https://pycqa.github.io/isort/). If you are unfamiliar with these, feel free to check out their documentations. However, it should not be a large problem if you are unfamiliar with their style guides, since they automatically format your code for you (in most cases). `black` is a style formatter that ensures uniformity of the coding style throughout the project, which helps with readability. To use `black`, simply run (inside the root directory)
```
black .
```
`isort` is a utility that automatically sorts and separates imports to also improve readability. To use `isort`, simply run (inside the root directory)
```
isort .
```

**Testing:** When your contribution is a simple bugfix it can be sufficient to use the existing tests. However, in most cases you will have to add additional tests to check your new feature or to ensure that whatever bug you are fixing no longer occurs. These tests can be added to the `/tests` directory. We use [pytest](https://docs.pytest.org) for testing, so if you are unfamiliar with this, please check their documentation or use the other tests as an example. If you think your contribution is ready, you can test it by running (inside the root directory)
```
pytest
```
If you made any changes to the documentation then you will also need to test the build process of the docs. You can read how to do this down below underneath the "Contributing to documentation" header.


# Contributing to documentation
**Optional dependencies:** When making documentation contributions, you will most likely need the `docs` optional dependencies to test the build process of the docs and the `dev` optional dependencies for formatting.

**Contributing:** Our documentation is built using the documentation generator [sphinx](https://www.sphinx-doc.org). Most of the documentation is generated automatically from the docstrings, so if you want to contribute to the API documentation, a simple change to the docstring of the relevant part of the code should suffice. For more complicated changes, please take a look at the `sphinx` documentation.

**Formatting:** For formatting new documentation, please use the existing documentation as examples. Once you are happy with your contribution, use `black` as described under the "Contributing to development" header to ensure that the code satisfies our formatting style.

**Testing:** To test the build process of the documentation, please follow the instructions within the `README` inside the `docs` directory.


# Contributing to new tutorials
**Optional dependencies:** When contributing to the tutorials, you will need the `dev` optional dependencies.

**Contributing:** We have two types of tutorials:
<ol>
    <li>Basic tutorials for learning about HypLL</li>
    <li>Tutorials showcasing how to implement peer-reviewed papers using HypLL</li>
</ol>
If your intended contribution falls under the first category, first make sure that there is a need for your specific tutorial. In other words, make sure that there is an issue asking for your tutorial and make sure that its inclusion is already agreed upon. If it falls under the second category, make sure that the paper that you are implementing is a peer-reviewed publication and, again, that there is an issue asking for this implementation.

**Formatting:** The formatting for tutorials is not very strict, but please use the existing tutorials as a guideline and ensure that the tutorials are self-contained.

**Testing:** Once you are ready with your tutorial, please test it by ensuring that it runs to completion. 


# Code review process
Pull requests have to be reviewed by at least one of the HypLL collaborators. 
