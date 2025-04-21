## Contributing to **Particula**

Thanks for thinking about contributing! Our goal is to make your first pull request (PR) smooth—whether you are brand‑new to open‑source or a seasoned developer.

---

### 1. Choose the Right Path

| You are… | Start here |
|----------|-----------|
| **Experienced with Git + Python** | Jump straight to [Github workflow](#3-github-flow) below. |
| **New to GitHub or virtual‑envs** | Follow [setup for new contributors](#2-setup-for-new-contributors) and the [Contributor Workflow guide](../Examples/Setup_Particula/Details/Contributor_Workflow.md). |
| **Planning to contribute major code or new features** | Follow [Feature Workflow](Feature_Workflow/index.md) and [Code Specifications](Code_Specifications/index.md) before you begin. |

---

### 2. Setup for New Contributors

For the complete, click‑through tutorial (fork → clone → virtual‑env →
editable install) see the
[Contributor Workflow guide](../Examples/Setup_Particula/Details/Contributor_Workflow.md).

> _If you already have Python, Git, and a preferred editor, skim this section._

1. **Install Git & VS Code**  
   * Git: <https://git‑scm.com/downloads>  
   * VS Code: <https://code.visualstudio.com/> (recommended extensions: _Python_, _GitHub Pull Requests & Issues_).

2. **Fork the repository**  
   Click **Fork** on <https://github.com/uncscode/particula>. This creates `yourname/particula`.

3. **Clone your fork**  
   ```bash
   git clone https://github.com/<your‑username>/particula.git
   cd particula
   ```

4. **Create a virtual environment** (pick **one**)

   | Tool | Command | Instructions |
   |------|----------|---------|
   | **uv (fast, minimal)** | `uv venv .venv`<br>`source .venv/bin/activate`<br>`uv pip install -e ".[dev,extra]"` | [uv setup guide](/particula/Examples/Setup_Particula/Details/Setup_UV/) |
   | **pip + venv** | `python -m venv .venv`<br>`source .venv/bin/activate`<br>`pip install -e ".[dev,extra]"` | [pip setup guide](/particula/Examples/Setup_Particula/Details/Setup_Pip/) |
   | **conda** | `conda create -n particula-dev`<br>`conda activate particula-dev`<br>`pip install -e ".[dev,extra]"` | [conda setup guide](/particula/Examples/Setup_Particula/Details/Setup_Conda/) |

5. **Verify installation**  
   ```bash
   pytest -q    # all tests should pass
   particula --help
   ```

---

### 3. GitHub Flow

| Step | What you do | Why |
|------|-------------|-----|
| 1 | **Fork → Clone → Set up `.venv` → `pip install -e ".[dev,extra]"`** | Gives you a local, editable checkout with all dev tools. |
| 2 | **Sync with upstream**<br>(keeps your `main` current) | Avoids merge conflicts later. |
| 3 | **Create a feature branch** | Isolates your work and makes the PR easy to review. |
| 4 | **Theory → Code → Test → Examples** | Adds value without breaking existing functionality. |
| 5 | **Commit + Push** to your fork | Publishes your branch to GitHub. |
| 6 | **Open a Pull Request** to `uncscode/particula` | Starts the review & CI pipeline. |
| 7 | **Discuss & Iterate** with reviewers | Polishes the contribution. |
| 8 | **Merge** (done by maintainer) | Your code lands in `main`. |
| 9 | **Delete branch** (optional) | Keeps your fork tidy. |
| 10 | **Celebrate** 🎉 | You just helped improve **Particula**! |

#### Expanded GitHub Flow

**Step 1 – Fork → Clone → Environment**

```bash
# (on GitHub) click Fork, then:
git clone https://github.com/<your‑username>/particula.git
cd particula

# create and activate a virtual env (pick one method)
python -m venv .venv           # or: uv venv .venv   |   conda create -n particula-dev python=3.11
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# install in editable (dev) mode
pip install -e ".[dev,extra]"
```

**Step 2 – Sync your fork with upstream (repeat as needed)**

```bash
git remote add upstream https://github.com/uncscode/particula.git   # one‑time
git checkout main
git pull upstream main
git push origin main
```

**Step 3 – Create a feature branch**

```bash
git checkout -b issue123-fix-simulation
```


**Step 4 – Theory → Code → Test → Examples**

For a step‑by‑step breakdown see the
[Feature Workflow guide](Feature_Workflow/index.md) and the
[Code Specifications](Code_Specifications/index.md).

```bash
# run unit tests with warnings treated as errors
pytest -Werror
```

* Add/adjust tests in `*/tests/`.  
* Update docstrings & MkDocs pages for any behavior change.


**Step 5 – Commit & push**

```bash
git add .
git commit -m "FIX #123: timestep rollover in simulation"
git push -u origin issue123-fix-simulation
```


**Step 6 – Open a Pull Request**

* Go to your fork → **Compare & pull request**.  
* Fill in the PR template; mark as **Draft** for early feedback if desired.


**Step 7 – Discuss & iterate**

* GitHub Actions runs tests (`pytest -Werror`), linters (`flake8`, `pylint`), and docs build automatically.  
* Push additional commits to the same branch—CI re‑runs and the PR updates.


**Step 8 – Merge**  
When CI is green and reviews are approved, a maintainer will merge your PR into `main`.


**Step 9 – Delete your branch** (Optional)

```bash
# delete remote branch
git push origin --delete issue123-fix-simulation
# delete local branch
git branch -d issue123-fix-simulation
```


**Step 10 – Celebrate!** 🎉  
Your contribution is now part of **Particula**—thank you!

---

### 4. Coding Standards & Review Expectations

| Topic | Rule |
|-------|------|
| **Style** | Detailed rules: [Code Specifications](Code_Specifications/index.md). |
| **Docstrings** | Follow the templates in [Function docstring format](Code_Specifications/Function_docstring_format.md) and [Class docstring format](Code_Specifications/Class_docstring_format.md). One‑line summary + details + sections (`Arguments`, `Returns`, `Raises`, `Examples`, `References`). |
| **Typing** | Use `typing` annotations. Omit types in docstrings. |
| **Tests** | Every public function/class must have at least one `pytest` test. Aim for coverage ≥ 90 %.  See [Add Unit Tests](Feature_Workflow/Add_Unit_Test.md). |
| **Commit messages** | Imperative mood, ≤ 72 chars summary + context body if needed. |
| **Large changes** | Open a **discussion** first and discuss design before implementation. |

---

### 5. Common Commands

*one‑liners you can copy‑paste*

| Purpose | Command |
|---------|---------|
| **Quick unit tests** | `pytest -q -Werror` |
| **Run tests in parallel** | `pytest -Werror` |
| **Static type‑checking (pytype, Mac/Linux)** | `pytype particula` |
| **Black auto‑format (79 cols)** | `black . --line-length 79` |
| **Flake8 lint** | `flake8 . --config .github/.flake8` |
| **Pylint lint** | `pylint particula` |

> **CI note:** Every pull request triggers **GitHub Actions** (Ubuntu / macOS / Windows).  
> The workflow runs `pytest -n auto -Werror`, `flake8`, `pylint`, `pytype`, builds the docs, and checks coverage.  
> Any warning promoted to an error (via `-Werror`) or other failure marks the PR ❌. Click **“Details →”** beside the failing job to view logs, fix locally, push again, and the checks will re‑run automatically.

---

### 6. Need Help?

* **Questions:** open a “Discussion” or tag a maintainer in your PR.  
* **Stuck on Git?** Try `git status`, `git restore`, or ask for pairing in the chat.  
* **Broken tests on CI?** Click “Details” next to the failing job; logs usually point to the exact line.

We appreciate every contribution—code, docs, tests, or ideas. Welcome to the **Particula** community! 🎉
