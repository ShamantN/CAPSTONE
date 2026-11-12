# Repository Settings & Branch Protection Reference (`ShamantN/CAPSTONE`)

- **Repository Visibility:** `PUBLIC` (`https://github.com/ShamantN/CAPSTONE`)
- **Maintainer / Code Owner:** `@ShamantN`
- **CI Workflow:** [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)

---

## Choosing Between Option A (Solo Maintainer) vs Option B (Team Collaboration)

GitHub enforces a built-in rule: **a Pull Request author cannot approve their own PR**. If `@ShamantN` opens a PR while **both** `Require at least 1 approval` AND `Do not allow bypassing` are enabled on `main` without a second human collaborator approving it, GitHub blocks merging.

| Mode | Required Reviews | Bypass Setting | Code Owner Review | When to Use |
| :--- | :---: | :--- | :---: | :--- |
| **Option A (Solo Owner / Release Prep)** | `0` approvals *(or uncheck `Do not allow bypassing` so admin can merge own PR)* | Admin allowed until collaborator joins **OR** `Required approvals = 0` with CI + manual gate checks required | Keep [`.github/CODEOWNERS`](../../.github/CODEOWNERS) active for collaborator PRs | Whenever `@ShamantN` authors PRs before another maintainer is available to approve |
| **Option B (Active Team / Friend Collaborator)** | `1` approval | **Checked (`Do not allow bypassing`)** | **Checked (`Require review from Code Owners`)** | Once a collaborator with **`Write`** access (not `Admin`) is active and reviews PRs |

See step-by-step GitHub UI setup instructions in [`docs/run/MAINTAINER_SETUP.md`](../run/MAINTAINER_SETUP.md).

