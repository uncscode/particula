# Home

The documentation for `particula` is organized into source code and examples in Jupyter notebooks.
To learn more about Jupyter notebooks, visit [jupyter.org](https://jupyter.org/) and [jupyterbook.org](https://jupyterbook.org/).


## Getting Started with Python

If you are new to Python, it's highly recommended to go through an introductory course to build a solid foundation. "Python for Everybody" is an excellent free resource that covers the basics and beyond:

- Access the course and materials at [Python for Everybody](https://www.py4e.com/).

## Gorkowski/PARTICULA Setup Instructions

Welcome to the development fork of `Particula`. This repository is actively developed, so you may encounter frequent updates and some instability as new features are integrated.

### Setting Up Your Development Environment

**Step 1: Install Visual Studio Code (VSCode)**

To edit and manage your code efficiently, download and install Visual Studio Code, a popular code editor that supports Python and many other languages.

- Visit the [Visual Studio Code website](https://code.visualstudio.com/) to download the installer for Windows.
- Follow the installation prompts to install VSCode on your machine.
- Once installed, launch VSCode to configure it for Python development.

**Step 2: Install Miniconda**  

Install Miniconda, which includes Conda, a powerful package and environment manager. This tool will help you manage different project dependencies separately and efficiently.

- Download Miniconda for Windows from [Miniconda's website](https://docs.conda.io/en/latest/miniconda.html).
- Follow the installation instructions to install Miniconda on your system.

**Step 3: Setup Proxy**

If you are behind a proxy, you may need to configure your proxy settings to allow Conda, Pip, and VScode to access the internet.


**Step 4: Create a New Python Environment**  

Avoid conflicts with other development projects by creating an isolated Python environment. Here’s how:

- Open VSCode, then open the integrated terminal (`Terminal > New Terminal`).
  - Be sure to select `cmd` for command prompt.
- Use the following Conda command to create an environment named `analysisV1` with Python 3.11:

  ```bash
  conda create --name analysisV1 python=3.11
  ```

**Step 5: Activate the Environment**  

Ensure you’re working within the context of your new environment:
- In the VSCode terminal, activate your environment by running:

  ```bash
  conda activate analysisV1
  ```

### Installing the Project

**Step 6: Install the Project**

Now, install the `Particula` fork directly using pip in your activated environment:

```bash
pip install git+https://github.com/Gorkowski/particula.git
```


<!-- ## Installing `particula`

You can install `particula` from PyPI using the following command:

```bash
python -m pip install particula
```

Or from conda-forge using the following command:

```bash
conda install -c conda-forge particula
```

Alternative, you could fork the repository (or copy it locally) and install it using the following command:

```bash
git clone https://github.com/uncscode/particula.git
cd particula
python -m pip install particula
``` -->

## Contributing to `particula`

We are open to and we welcome contributions from anyone who wants to contribute to this project.
We have a short [contributing document](contribute) in the root of the repository, which you can read.
However, feel free to reach out with any questions or comments!