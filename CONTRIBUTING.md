# Contributing guidelines

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

More information about contributing to this project can be found in docs/ and on our website [https://uncscode.github.io/particula](https://uncscode.github.io/particula).
We are excited and looking forward to your contribution!