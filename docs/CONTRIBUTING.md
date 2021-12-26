# Contributor Guidelines

First of all, thank you for considering contributing to this project!
While we have specific guidelines below, _we also encourage you to
contribute to the best of your ability and not let these guidelines
hinder your productivity and creativity_. We are open to any contribution,
and we are always looking for new ways to improve the project.
We are also open to any feedback, however small or large,
and in any area (from documentation to calculation to presentation).

---

The cycle of contribution goes something like this:

1. See if there is an issue open that you can help with.
If there is not one, please open one.

2. Create a personal fork of this repository;
and in it, create a branch (from `uncscode:main`)
with the `issue000` in the name of the branch
(e.g. `username/issue000` or `issue000`),
where `000` is the number of the issue from step 1.

3. Set up an appropriate environment:
    - an easy option is just to use the `.devcontainer` in root
    - another option is to either `pip install` or `conda install`
    the packages required for development in `requirements.txt` in root.

4. Write your code in the branch. This usually includes the following.

    a. Code to be implemented.

    b. Documentation associated with added code in a.

    c. Tests associated with added code in a.

    d. Ideally, you'd also add a Jupyter notebook to
    showcase your work (if applicable).

5. _Optionally_, you can run standard linting and testing calls
on your code _locally_ to make sure it works as expected.
This can be done in several ways,
for example the `pylint`, `flake8`, and `pytest` calls below.
These calls will be run once you submit your pull request.

6. Submit a pull request to the `main` branch of this repository.
Upon submission, standard automated tests will be run on your code.

7. If you don't hear back from maintainers,
feel free to mention one of us directly in the comments of the PR.
Expect to have speedy feedback and help from us to finalize the PR.

```bash
pylint $(find particula/ -name "*.py" | xargs)
```

```bash
flake8 particula/ --count
```

```bash
pytest particula/
```

---

More information about contributing to this project can be found below.
We are excited and looking forward to your contribution!

---

## GitHub

We use GitHub to develop `particula` completely in the open. Our repository is available here: [https://uncscode.github.io/particula/](https://uncscode.github.io/particula/).
There are several ways to use GitHub: through the command line via `git` and/or `gh`, through the web interface and/or the GitHub web editor, or through an IDE like PyCharm or a code editor like Visual Studio Code.
In general, we recommend that you fork our repository, that you work with VS Code, and that submit a pull request based on an issue.
If any of these sound unfamiliar or if you need help, please see more information below and feel free to contact us directly to discuss options.
We look forward to getting you started and up to speed on this project with us!

Links: [https://docs.github.com/en/get-started](https://docs.github.com/en/get-started)

## VS Code

Visual Studio Code is a free and open-source code editor for writing code and it has a rich ecosystem of extensions that allow you to write code in a variety of languages with a lot of helpful features and tools.

Links: [https://code.visualstudio.com/](https://code.visualstudio.com/)

## `.devcontainer`

In the root of our repository, we have a `.devcontainer` folder that contains the environment variables required for development.
This is a convenient way to set up the environment for development.
It requires Docker to be installed and running.
And the first time it runs it may take a few minutes to install the Docker image (a minimal operating system with Python 3.9).

Links: [https://code.visualstudio.com/docs/remote/containers](https://code.visualstudio.com/docs/remote/containers)

## Python code style

We follow the Google's Python style guide.
We encourage you to follow it too, but we also encourage you to contribute to the best of your ability and not let these guidelines hinder your productivity and creativity.

Links: [https://google.github.io/styleguide/pyguide.html](https://google.github.io/styleguide/pyguide.html)

## Running `particula` locally

Once you are in the root directory, you will be able to import `particula` as a package/model and thus all documentation on website applies.
You must be in the root directory.

## Writing tests

It is essential that every piece of code has an associated test.
This is a good way to ensure that the code is working as intended.
It also ensures that the code is not broken and that the code is not too complex.
However small or big, a test is always required.

## Running testing/linting locally

We use `pytest`, `pylint`, and `flake8` to run tests and linting.
The command below can be run in the root directory like you'd run the package above.

```bash
pylint $(find particula/ -name "*.py" | xargs)
```

```bash
flake8 particula/ --count
```

```bash
pytest particula/
```

## Building `particula` locally

To build `particula` locally, you must be in the root directory.
You have two options, depending on your usage case.

1. You can use `python -m build` to build the package wheels locally (note: you will need to install `build` too, via `pip install build`).
2. You can build the conda recipe available at [https://github.com/conda-forge/particula-feedstock](https://github.com/conda-forge/particula-feedstock) either via `python build-locally.py` in the root of `particula-feedstock` or via `conda build recipe` (equivalently, but faster, `mamba build recipe`). For the latter, you will need to have `conda-build` installed (for `conda build` to work) or `boa` (for `mamba build` to work). In either case, you can install package with conda via, `conda install conda-build` or `mamba install boa`.

Links: [https://packaging.python.org/en/latest/tutorials/packaging-projects/](https://packaging.python.org/en/latest/tutorials/packaging-projects/) and [https://docs.conda.io/projects/conda-build/en/latest/user-guide/index.html](https://docs.conda.io/projects/conda-build/en/latest/user-guide/index.html)

## Documentation writing

We prefer that the tutorials are written in the form of Jupyter notebooks after the package is released and published.
A convenient option is using Google's Colaboratory to write the notebooks.

Links: [https://colab.research.google.com/](https://colab.research.google.com/)

## More information

We will update this regularly with more information, but in the meanwhile, please feel free to contact us directly on GitHub.
