# *PARTICULA*

Particula is a simple, fast, and powerful particle simulator, or at least two of the three, we hope. It is a simple particle system that is designed to be easy to use and easy to extend. The goal is to have a robust aerosol (gas phase *and* particle phase) simulation system that can be used to answer scientific questions that arise for experiments and research discussions.

(*DISCLAIMER* [GitHub Copilot](https://copilot.github.com/) is heavily involved in the development of this project and especially in the writing of these paragraphs! *DISCLAIMER*)

(*WARNING* `particula` is in early and messy development. It is not ready for production use yet, but you are welcome to use it anyway --- even better, you're welcome to contribute! *WARNING*)

## Goals & conduct

The main goal is to develop an aerosol particle model that is usable, efficient, and productive. In this process, we all will learn developing models in Python and associated packages. Let us all be friendly, respectful, and nice to each other. Any code added to this repository is automatically owned by all. Please speak up if something (even if trivial) bothers you. Talking through things always helps. This is an open-source project, so feel free to contribute, however small or big your contribution may be.

We follow the Google Python style guide [here](https://google.github.io/styleguide/pyguide.html). We have contribution guidelines [here](https://github.com/uncscode/particula/blob/main/docs/CONTRIBUTING.md) and a code of conduct [here](https://github.com/uncscode/particula/blob/main/docs/CODE_OF_CONDUCT.md) as well.

## Usage & development

The development of this package will be illustrated through Jupyter notebooks ([here](https://github.com/uncscode/particula/blob/main/docs)) that will be put together in the form of a Jupyter book on our [website](https://uncscode.github.io/particula). To use it, you can install `particula` from PyPI or conda-forge with `pip install particula` or `conda install -c conda-forge particula`, respectively.

For development, you can fork this repository and then install `particula` in an editable (`-e`) mode --- this is achieved by `pip install -e ".[dev]"` in the root of this repository where `setup.cfg` exists. Invoking `pip install -e ".[dev]"` will install `particula`, its runtime requirements, and the development and test requirements. The editable mode is useful because it will allow to see the manifestation of your code edits globally through the `particula` package in your environment (in a way, with the `-e` mode, `particula` self-updates to account for the latest edits). Note, there is a known bug with `-e` mode --- if you face problems installing it due to permissions, try `pip install --prefix=~/.local -e ".[dev]"` instead where `--prefix` is somewhere in the user space.
