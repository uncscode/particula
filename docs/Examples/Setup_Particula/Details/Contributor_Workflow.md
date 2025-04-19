## Contributing to Particula (Developer Contributors)

Interested in contributing to the Particula project? This section explains how to set up a development environment and the workflow for contributing changes. We assume you have a GitHub account and Git installed (see the [Beginner Setup](#beginner-environment-setup-for-users-new-to-python) if not). Following these steps will allow you to run the latest development version of Particula and prepare your contributions for a pull request.

### Fork the Repository (on GitHub)

First, **fork the Particula repository** to your own GitHub account. Forking creates your personal copy of the project:

- Visit the Particula GitHub repo: **<https://github.com/uncscode/particula>**.  
- Click the **“Fork”** button in the top-right corner of the page.  
- GitHub will create a fork under your account (e.g., `github.com/<your-username>/particula`).

*(If you’re new to forking, see GitHub’s guide on how to fork a repository for more details.)*

### Clone Your Fork Locally

Next, clone the forked repository to your local machine so you can work on the code:

1. Open a terminal on your development machine and navigate to a folder where you want to place the project.
2. Run the clone command (replace `<your-username>` with your GitHub username):  
   ```bash
   git clone https://github.com/<your-username>/particula.git
   ```  
   This creates a directory `particula` with the project files.
3. Change into the project directory:  
   ```bash
   cd particula
   ```

Now you have the code locally. Before writing code, we need to set up a Python environment for development.

### Set Up a Development Environment (`.venv`)

It’s good practice to use a virtual environment (named `.venv`) for development. This keeps the project’s dependencies isolated. You have two options to create a `.venv` in the project:

- **Using Python’s built-in venv:**  
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate  # activate on Linux/Mac
  .\.venv\Scripts\activate   # activate on Windows (PowerShell/CMD)
  ```  
  This creates a folder `.venv` in the project and activates it. After activation, your prompt will show `(.venv)`.
- **Using uv:** If you prefer uv, you can create and activate the env in one step:  
  ```bash
  uv venv .venv
  ```  
  This will create `.venv` and automatically make it active for subsequent `uv` commands.

With the virtual environment active (by either method), upgrade pip and install Particula in **development mode** with the required dev dependencies:

```bash
pip install -U pip setuptools wheel
pip install -e ".[dev,extra]"
```

The `pip install -e ".[dev,extra]"` command tells pip to install the package in editable mode (`-e`) from the current directory (`.`) including the `[dev,extra]` optional dependencies (which include development and extra tools). This will pull in things like testing frameworks, linters, etc., as defined by Particula’s `pyproject.toml`. If using uv, run `uv pip install -e ".[dev,extra]"` equivalently.

**Tip:** The `.[dev,extra]` syntax installs all standard and extra dependencies needed for development (such as documentation or additional features). You can inspect `pyproject.toml` for the exact extras defined.

### Development Workflow: Branch, Code, Commit, PR

You are now ready to make changes. Head over to the [contributor](/docs/contribute/index.md) page for more information on the development workflow.
