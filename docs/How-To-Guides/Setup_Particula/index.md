# Index: Setup Particula

## Getting Started with Python

If you are new to Python, it's highly recommended to go through an introductory course to build a solid foundation. "Python for Everybody" is an excellent free resource that covers the basics and beyond:

- Access the course and materials at [Python for Everybody](https://www.py4e.com/).


## Setting Up Your Development Environment

**Step 1: Install Visual Studio Code (VSCode)**

To edit and manage your code efficiently, download and install Visual Studio Code, a popular code editor that supports Python and many other languages.

- Visit the [Visual Studio Code website](https://code.visualstudio.com/) to download the installer for Windows.
- Follow the installation prompts to install VSCode on your machine.
- Once installed, launch VSCode to configure it for Python development.

**Step 2: Install Miniconda**

Install Miniconda, which includes Conda, a powerful package and environment manager. This tool will help you manage different project dependencies separately and efficiently.

- Download Miniconda for Windows from [Miniconda's website](https://docs.conda.io/en/latest/miniconda.html).
- Follow the installation instructions to install Miniconda on your system.

**Step 3: Install Git**

Install Git to manage your code repositories effectively. Git is a version control system that lets you track changes, revert to previous stages, and collaborate on projects with others.

- Download Git from [Git's official website](https://git-scm.com/download/win).
- Run the downloaded file to start the installation.
- During the installation, choose your preferred editor for Git, and make sure to select "Git from the command line and also from 3rd-party software" to ensure it integrates well with your system's command prompt.
- Complete the installation by following the on-screen instructions.

Once installed, you can verify the installation by opening a command prompt or terminal and typing `git --version`, which should display the installed version of Git.

**Step 4: Setup Proxy**

If you are behind a proxy, you may need to configure your proxy settings to allow Conda, Pip, git, and VScode to access the internet.


**Step 5: Create a New Python Environment**  

Avoid conflicts with other development projects by creating an isolated Python environment. Here’s how:

- Open VSCode, then open the integrated terminal (`Terminal > New Terminal`).
  - Be sure to select `cmd` for command prompt.
- Use the following Conda command to create an environment named `analysisV1` with Python 3.11:

```bash
conda create --name analysisV1 python=3.11
```

**Step 6: Activate the Environment**  

Ensure you’re working within the context of your new environment:
- In the VSCode terminal, activate your environment by running:

```bash
conda activate analysisV1
```

## Installing the Project

**Step 7: Install the Project**

Now, install the `Particula` using pip in your activated environment, use one of the following methods:

### Install the pip package

```bash
pip install particula
```

### or Install the main repository

```bash
pip install git+https://github.com/uncscode/particula.git
```

### or Install the *Gorkowski* fork

```bash
pip install git+https://github.com/Gorkowski/particula.git
```

