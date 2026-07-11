## Contributing to **Particula**

Thanks for thinking about contributing! Our goal is to make your first pull request (PR) smooth—whether you are brand‑new to open‑source or a seasoned developer.

---

### 1. Choose the Right Path

| You are… | Start here |
|----------|-----------|
| **Experienced with Git + Python** | Jump straight to [Github workflow](#3-github-flow) below. |
| **New to GitHub or virtual‑envs** | Follow [setup for new contributors](#2-setup-for-new-contributors) and the [Contributor Setup guide](../Examples/Setup_Particula/Details/Contributor_Setup.md). |
| **Planning to contribute major code or new features** | Follow [Feature Workflow](Feature_Workflow/index.md) and [Code Specifications](Code_Specifications/index.md) before you begin. |

---

### 2. Setup for New Contributors

For the complete, click‑through tutorial (fork → clone → virtual‑env →
editable install) see the
[Contributor Setup guide](../Examples/Setup_Particula/Details/Contributor_Setup.md).

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
   | **uv (fast, minimal)** | `uv venv .venv`<br>`source .venv/bin/activate`<br>`uv pip install -e ".[dev,extra]"` | [uv setup guide](../Examples/Setup_Particula/Details/Setup_UV.md) |
   | **pip + venv** | `python -m venv .venv`<br>`source .venv/bin/activate`<br>`pip install -e ".[dev,extra]"` | [pip setup guide](../Examples/Setup_Particula/Details/Setup_PIP.md) |
   | **conda** | `conda create -n particula-dev`<br>`conda activate particula-dev`<br>`pip install -e ".[dev,extra]"` | [conda setup guide](../Examples/Setup_Particula/Details/Setup_Conda.md) |

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
| 3 | **Discuss your feature** | Helps us understand your goals and avoid duplication of effort. |
| 4 | **Theory → Code → Test → Examples** | Adds value without breaking existing functionality. |
| 5 | **Commit + Push** to your fork | Publishes your branch to GitHub. |
| 6 | **Open a Pull Request** to `uncscode/particula` | Starts the review & CI pipeline. |
| 7 | **Discuss & Iterate** with reviewers | Polishes the contribution. |
| 8 | **Rebase** (done by maintainer) | Your code lands in `main`. |
| 9 | **Celebrate** 🎉 | You just helped improve **Particula**! |

#### Expanded GitHub Flow

**Step 1 – Fork → Clone → Environment**

See setup guides above for details.

**Step 2 – Sync your fork with upstream (repeat as needed)**

```bash
git remote add upstream https://github.com/uncscode/particula.git   # one‑time
git checkout main
git pull upstream main
git push origin main
```

**Step 3 – Discuss your feature**

If you are planning a large feature or change, please open a **Discussion** first.
This helps us understand your goals and avoid duplication of effort.
Or if you are new to the project, ask for guidance in the [GitHub Discussions](https://github.com/uncscode/particula/discussions).

If you have experience with the codebase and it is a small change, you can skip this step.
Pure bug fixes or small features or new examples are usually fine to skip this step too.

**Step 4 – Theory → Code → Test → Examples**

For a full step‑by‑step breakdown see the
[Feature Workflow guide](Feature_Workflow/index.md) and the
[Code Specifications](Code_Specifications/index.md).

- Create feature branch (e.g., `issue123-fix-simulation`).
- Add/adjust new code or documentation.
- Add/adjust tests.
- Add/adjust docstrings & markdown pages.

**Step 5 – Commit & push**

For an overview of git standards we use, see [Linear Git Repository](../contribute/Code_Specifications/Details/Linear_Git_Repository.md).

```bash
git add .
git commit -m "FIX #123: timestep rollover in simulation"
git push -u origin issue123-fix-simulation
```

**Step 6 – Open a Pull Request**

* Go to your fork → **Compare & pull request**.  
* Fill in the PR template; mark as **Draft** for early feedback if desired.


**Step 7 – Discuss & iterate**

* GitHub Actions runs tests (`pytest -Werror`), Ruff validation, type checks, and docs build automatically.  
* Push additional commits to the same branch—CI re‑runs and the PR updates.


**Step 8 – Rebase**  
When CI is green and reviews are approved, a maintainer will merge-rebase your PR into `main`.

**Step 9 – Celebrate!** 🎉  
Your contribution is now part of **Particula**—thank you!

---

### 4. Coding Standards & Review Expectations

| Topic | Rule |
|-------|------|
| **Style** | Detailed rules: [Code Specifications](Code_Specifications/index.md). |
| **Docstrings** | Follow the templates in [Function docstring format](../contribute/Code_Specifications/Details/Function_docstring_format.md) and [Class docstring format](../contribute/Code_Specifications/Details/Class_docstring_format.md). One‑line summary + details + sections (`Arguments`, `Returns`, `Raises`, `Examples`, `References`). |
| **Typing** | Use `typing` annotations. Omit types in docstrings. |
| **Tests** | Every public function/class must have at least one `pytest` test. Add regression coverage for fixes and keep change-scope coverage strong. See [Add Unit Tests](../contribute/Feature_Workflow/Details/Add_Unit_Test.md) and the [testing guide](../../.opencode/guides/testing_guide.md). |
| **Commit messages** | Imperative mood, ≤ 72 chars summary + context body if needed. |
| **Large changes** | Open a **discussion** first and discuss design before implementation. |

---

### 5. Common Commands

*one‑liners you can copy‑paste*

| Purpose | Command |
|---------|---------|
| **Quick unit tests** | `pytest -q -Werror` |
| **Coverage check** | `pytest --cov=particula --cov-report=term-missing` |
| **Ruff auto-fix lint** | `ruff check particula/ --fix` |
| **Ruff format** | `ruff format particula/` |
| **Final Ruff check** | `ruff check particula/` |
| **Static type-checking** | `mypy particula/ --ignore-missing-imports` |

> **CI note:** Every pull request triggers **GitHub Actions** (Ubuntu / macOS / Windows).  
> The local validation path should mirror the current repository workflow: run
> `ruff check particula/ --fix`, `ruff format particula/`,
> `ruff check particula/`, `mypy particula/ --ignore-missing-imports`, and the
> relevant `pytest` command for your change scope (plus coverage when needed).  
> Any warning promoted to an error (via `-Werror`) or other failure marks the PR ❌. Click **“Details →”** beside the failing job to view logs, fix locally, push again, and the checks will re‑run automatically.
>
> **GPU validation note:** Use the testing guide and GPU roadmap as the
> canonical policy for Warp-targeted validation. Focused Warp CPU runs are the
> default supported path, CUDA runs are optional/local/manual only, and
> missing-Warp or missing-CUDA cases should surface as expected skips rather
> than CI failures. For targeted local runs, use marker selection such as
> `-m "warp and gpu_parity"` or `-m "warp and cuda"`.

---

### 6. Need Help?

* **Questions:** open a “Discussion” or tag a maintainer in your PR.  
* **Stuck on Git?** Try `git status`, `git restore`, or ask for pairing in the chat.  
* **Broken tests on CI?** Click “Details” next to the failing job; logs usually point to the exact line.

We appreciate every contribution—code, docs, tests, or ideas. Welcome to the **Particula** community! 🎉
