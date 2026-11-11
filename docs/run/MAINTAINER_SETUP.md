# Maintainer GitHub Setup Guide (`@ShamantN`)

- **Purpose:** One-time click-by-click GitHub UI setup to safely add collaborators without giving admin bypass rights or risking accidental changes to `main`, `consistency-v2`, or the frozen verifier.
- **Note:** Only repository admins (`@ShamantN`) can apply these settings in GitHub **Settings**. Do not script or bypass these checks externally.

---

## 1. Add Collaborators with `Write` Role (Not `Admin`)

1. Open **https://github.com/ShamantN/CAPSTONE** on GitHub while signed in as `@ShamantN`.
2. Click the **`Settings`** tab in the top repository navigation bar.
3. In the left sidebar under **`Access`**, click **`Collaborators`** *(or `Collaborators and teams`)*.
4. Click the green **`Add people`** button.
5. Enter your friend's GitHub username or email address and select them.
6. Under **Choose a role**, select **`Write`** — **do NOT select `Maintain` or `Admin`** *(only `Write` prevents collaborators from deleting branches, disabling branch protection, or altering secrets/repo visibility)*.
7. Click **`Add <username> to this repository`**.

---

## 2. Protect `main` and `consistency-v2` (Classic Branch Protection or Rulesets)

Repeat the rule below for **`main`** first, then **`consistency-v2`**:

1. Go to **`Settings`** -> left sidebar **`Code and automation`** -> **`Branches`**.
2. Click **`Add classic branch protection rule`** *(or **`Add branch protection rule`**)*.
3. In **Branch name pattern**, enter **`main`** *(repeat a second rule for **`consistency-v2`**)*.
4. Enable these checkboxes under **Protect matching branches**:
   - [x] **Require a pull request before merging**
     - Set **Required number of approvals before merging** to **`1`**
     - [x] **Dismiss stale pull request approvals when new commits are pushed** *(recommended)*
     - [x] **Require review from Code Owners** *(enforces [`.github/CODEOWNERS`](../../.github/CODEOWNERS) so edits to `consistency_v2/verify/`, `verifier_freeze.json`, `splits.py`, `split_access_log.jsonl`, `holdout_v2.json`, or `numbers.json` require `@ShamantN`)*
   - [x] **Require status checks to pass before merging**
     - [x] **Require branches to be up to date before merging**
     - Search and add the CI check name from [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) once visible in recent runs *(for example `test` / workflow job name)*.
   - [x] **Require conversation resolution before merging** *(recommended)*
   - [x] **Do not allow bypassing the above settings** *(critical — ensures even admins follow the PR + check gates!)*
5. Under **Rules applied to everyone including administrators**:
   - Leave **Allow force pushes** **UNCHECKED** *(blocks history rewrites)*.
   - Leave **Allow deletions** **UNCHECKED** *(blocks accidental branch deletion)*.
6. Click **`Create`** (or **`Save changes`**).

---

## 3. Quick Verification After Setup

1. Ask your collaborator to accept the GitHub email/notification invite and clone fresh following [`00_START_HERE.md`](00_START_HERE.md) & [`CONTRIBUTING.md`](../../CONTRIBUTING.md).
2. Confirm on GitHub **`Settings -> Branches`** that both `main` and `consistency-v2` show active protection badges and **`Require review from Code Owners`** is enabled.
3. Have the collaborator open a test branch (never direct push to `main` — verified blocked!) and verify `.github/pull_request_template.md` auto-populates on PR creation.

