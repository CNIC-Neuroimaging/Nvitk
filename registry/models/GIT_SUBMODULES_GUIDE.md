# Git Submodules: Complete Guide

**A comprehensive tutorial on using git submodules with BioImaging-Models**

---

## 📚 Table of Contents

1. [What are Git Submodules?](#what-are-git-submodules)
2. [Why Use Submodules for Models?](#why-use-submodules-for-models)
3. [Basic Concepts](#basic-concepts)
4. [Getting Started](#getting-started)
5. [Day-to-Day Workflow](#day-to-day-workflow)
6. [Advanced Operations](#advanced-operations)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## 🤔 What are Git Submodules?

Git submodules allow you to **include one Git repository inside another** as a subdirectory. The parent repository tracks a specific commit of the submodule, creating a snapshot of that submodule's state.

### Real-World Analogy

Think of it like this:
- Your **main repository** (`BioImaging`) is like a recipe book
- The **submodule** (`BioImaging-Models`) is like a separate ingredient catalog
- The recipe book **references** specific versions of ingredients from the catalog
- You can update the ingredient catalog independently
- The recipe book can choose when to use updated ingredients

---

## 🎯 Why Use Submodules for Models?

### Benefits

1. **Separation of Concerns**
   - Code and models are versioned independently
   - Models can be large (GBs) and updated frequently
   - Code changes don't require re-downloading models

2. **Selective Cloning**
   - Clone code repository quickly
   - Download models only when needed
   - Save bandwidth and disk space

3. **Version Control**
   - Track exactly which model version was used
   - Reproducibility: code commit + model commit = exact experiment state
   - Easy to rollback to previous model versions

4. **Shared Models**
   - Multiple projects can reference the same model repository
   - Consistent model versions across projects
   - Central model storage and management

5. **Independent Updates**
   - Update models without touching code repository
   - Update code without re-downloading models
   - Choose when to adopt new model versions

---

## 📖 Basic Concepts

### Key Terminology

- **Parent Repository**: The main repository that contains the submodule (e.g., `BioImaging`)
- **Submodule**: The external repository embedded in the parent (e.g., `BioImaging-Models`)
- **Submodule Commit**: A specific commit in the submodule that the parent tracks
- **`.gitmodules`**: Configuration file that defines submodules

### How It Works
```bash
BioImaging/ (Parent repo)
├── .git/
├── .gitmodules ← Submodule configuration
├── src/
│ └── imaging/
├── models/ ← Submodule directory
│ ├── .git ← Points to BioImaging-Models repo
│ ├── vascx/
│ └── retfound/
└── README.md
```


The parent repository stores:
- Path to submodule: `models/`
- Submodule repository URL: `git@gitlab.cnic.es:.../bioimaging-models.git`
- Specific commit SHA: `abc123...` (the exact state of the submodule)

---

## 🚀 Getting Started

### Prerequisites

```bash
# Ensure Git version supports submodules well
git --version
# Should be 2.13 or newer for best experience

# Ensure Git LFS is installed (for model files)
git lfs version
git lfs install
```

---

## 📥 Scenario 1: Adding Submodule to Existing Repository

### Step 1: Navigate to Parent Repository

```bash
cd /home/imarcoss/BioImaging
```

### Step 2: Add the Submodule

```bash
# Add BioImaging-Models as a submodule in the 'models' directory
git submodule add git@gitlab.cnic.es:your-group/bioimaging-models.git models

# What this does:
# 1. Clones BioImaging-Models into models/
# 2. Creates .gitmodules file
# 3. Stages both for commit
```

### Step 3: Review Changes

```bash
# Check what was created
git status

# You'll see:
# new file:   .gitmodules
# new file:   models (special "commit" type)

# View .gitmodules content
cat .gitmodules
```

Output:
```ini
[submodule "models"]
	path = models
	url = git@gitlab.cnic.es:your-group/bioimaging-models.git
```

### Step 4: Commit the Submodule

```bash
git commit -m "Add BioImaging-Models as submodule

- Models are now tracked in separate repository
- Enables independent versioning of code and models
- Uses Git LFS for efficient large file storage"

git push origin main
```

**Important**: After this commit, your repository tracks:
- The submodule location (`models/`)
- The submodule URL
- The current commit of the submodule (e.g., `abc123...`)

---

## 📥 Scenario 2: Cloning Repository with Submodules

When someone else clones `BioImaging`, they need to initialize submodules.

### Method 1: Clone with Submodules (Recommended)

```bash
# Clone parent and submodule in one command
git clone --recursive git@gitlab.cnic.es:your-group/bioimaging.git

cd bioimaging
ls models/  # Models directory is populated
```

### Method 2: Clone Then Initialize

```bash
# Clone parent repository
git clone git@gitlab.cnic.es:your-group/bioimaging.git
cd bioimaging

# At this point, models/ exists but is empty
ls models/  # Empty directory

# Initialize and clone submodules
git submodule init
git submodule update

# Now models/ is populated
ls models/  # Shows vascx/, retfound/, etc.
```

### Method 3: One-Liner for Existing Clone

```bash
# If you already cloned without --recursive
git submodule update --init --recursive
```

---

## 🔄 Day-to-Day Workflow

### Checking Submodule Status

```bash
# From parent repository root
git submodule status

# Output example:
# 9b3c8f1a9e... models (heads/main)
#   ↑ commit hash that parent is tracking
```

### Viewing Submodule Changes

```bash
# See if submodule has uncommitted changes
git status

# Output might show:
# modified:   models (new commits)
# This means the submodule has new commits not yet tracked by parent
```

---

## 📝 Scenario 3: Updating Models (Common Workflow)

### When New Models Are Added to BioImaging-Models

Someone added new models to the `BioImaging-Models` repository. Here's how you update:

```bash
cd /home/imarcoss/BioImaging

# Method 1: Update submodule to latest commit
cd models/
git pull origin main
cd ..

# Now parent knows submodule has changed
git status
# Shows: modified:   models (new commits)

# Update parent to track new submodule commit
git add models
git commit -m "Update models submodule to latest version

- Added new VascX model variants
- Updated TotalSegmentator to v2.1"

git push origin main
```

### Method 2: Update All Submodules from Parent

```bash
# From parent repository root
git submodule update --remote --merge

# This updates all submodules to their remote's default branch
# Then commit the update
git add .gitmodules models
git commit -m "Update models submodule"
git push
```

---

## 🔄 Scenario 4: Working on Models While Developing Code

### Developing Models and Code Simultaneously

```bash
# 1. Make sure submodule is on a branch (not detached HEAD)
cd /home/imarcoss/BioImaging/models
git checkout main
git pull origin main

# 2. Make changes to models
cp /path/to/new_model.pth vascx/new_model.pth
git add vascx/new_model.pth
git commit -m "Add new VascX model variant"
git push origin main

# 3. Return to parent and update submodule reference
cd /home/imarcoss/BioImaging
git add models
git commit -m "Update models submodule with new VascX model"
git push origin main
```

**Visual Representation:**
```bash
┌─────────────────────────────────────┐
│ BioImaging (parent) │
│ Tracks: models @ commit abc123 │
└─────────────────────────────────────┘
↓ references
┌─────────────────────────────────────┐
│ BioImaging-Models (submodule) │
│ Commit abc123: has VascX v1.0 │
│ Commit def456: has VascX v2.0 ← new│
└─────────────────────────────────────┘
After update:
┌─────────────────────────────────────┐
│ BioImaging (parent) │
│ Tracks: models @ commit def456 ✓ │
└─────────────────────────────────────┘
```

---

## 🎯 Scenario 5: Using Specific Model Version

### Pinning to a Specific Commit

Sometimes you want a specific model version, not the latest:

```bash
cd /home/imarcoss/BioImaging/models

# View available versions
git log --oneline

# Checkout specific commit
git checkout abc123def

# Return to parent
cd ..

# Commit this specific submodule version
git add models
git commit -m "Pin models to v1.2.0 (commit abc123)

- Ensures reproducibility for experiments
- v1.3.0 has breaking changes, staying on v1.2.0"

git push origin main
```

### Using Tagged Versions

```bash
cd /home/imarcoss/BioImaging/models

# List available tags
git tag -l

# Checkout tagged version
git checkout vascx-v1.0.0

cd ..
git add models
git commit -m "Pin models to VascX v1.0.0 release"
git push origin main
```

---

## 🔀 Scenario 6: Team Collaboration

### When Teammate Updates Models

Your teammate updated the models submodule. You need to sync:

```bash
cd /home/imarcoss/BioImaging

# Pull parent repository changes
git pull origin main

# Output shows:
# Fetching submodule models
# Submodule path 'models': checked out '...'

# Update submodule to tracked commit
git submodule update --init --recursive

# Now your models/ matches the commit tracked by parent
```

**Common Issue**: Submodule out of sync

```bash
# Symptom: git status shows "modified: models"
git status

# Fix: Update to tracked commit
git submodule update --init --recursive

# Verify
git status
# Should show clean working tree
```

---

## 🛠️ Advanced Operations

### Cloning Without Downloading Models (Fast Clone)

```bash
# Clone parent only
git clone git@gitlab.cnic.es:your-group/bioimaging.git
cd bioimaging

# Initialize submodule config but don't download yet
git submodule init

# Later, when you need models:
git submodule update

# Or with LFS, skip large files initially
GIT_LFS_SKIP_SMUDGE=1 git submodule update
cd models
git lfs pull --include="vascx/*"  # Download only specific models
```

### Removing a Submodule

```bash
# 1. Deinitialize
git submodule deinit -f models

# 2. Remove from git
git rm -f models

# 3. Remove cached data
rm -rf .git/modules/models

# 4. Commit removal
git commit -m "Remove models submodule"
```

### Changing Submodule URL

```bash
# Edit .gitmodules
nano .gitmodules
# Change URL to new location

# Sync the change
git submodule sync

# Update to use new URL
git submodule update --remote

# Commit change
git add .gitmodules
git commit -m "Update models submodule URL"
```

---

## 🐛 Troubleshooting

### Problem 1: "detached HEAD" in Submodule

**Symptom**: When you `cd models/` and run `git status`, it says "HEAD detached at..."

**Cause**: Submodules are checked out at specific commits, not branches.

**Solution**: This is normal! But if you want to make changes:

```bash
cd models/
git checkout main  # Switch to branch
# Make your changes
git commit -m "Changes"
git push origin main

cd ..
git add models  # Update parent to track new commit
git commit -m "Update models"
```

### Problem 2: Submodule Directory is Empty

**Symptom**: `ls models/` shows nothing

**Solution**:

```bash
# Initialize and update
git submodule update --init --recursive
```

### Problem 3: "modified: models" Won't Go Away

**Symptom**: `git status` always shows `modified: models`

**Solution A**: Reset to tracked commit

```bash
git submodule update --init --recursive
```

**Solution B**: Commit the new submodule state

```bash
git add models
git commit -m "Update models submodule"
```

### Problem 4: Merge Conflicts in Submodule

**Symptom**: During merge, conflict in `models` entry

**Solution**:

```bash
# Update to the correct commit (usually "theirs")
git checkout --theirs models
git add models
git commit

# Or update to latest
cd models/
git pull origin main
cd ..
git add models
git commit
```

### Problem 5: LFS Files Not Downloading

**Symptom**: Model files are tiny text pointer files

**Solution**:

```bash
cd models/
git lfs pull

# Or for specific files
git lfs pull --include="vascx/*"
```

---

## ✅ Best Practices

### 1. Always Keep Submodule on a Branch When Making Changes

```bash
# Before editing models
cd models/
git checkout main  # Not detached HEAD
```

### 2. Document Submodule Version in Commit Messages

```bash
git commit -m "Update analysis pipeline

Uses models submodule commit abc123:
- VascX v1.2.0
- TotalSegmentator v2.1"
```

### 3. Use Tags for Model Versions

In `BioImaging-Models`:

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

In `BioImaging`:

```bash
cd models/
git checkout v1.0.0
cd ..
git add models
git commit -m "Pin models to v1.0.0 release"
```

### 4. Update Regularly

```bash
# Weekly sync
cd models/
git pull origin main
cd ..
git add models
git commit -m "Update models (weekly sync)"
```

### 5. CI/CD Considerations

In `.gitlab-ci.yml`:

```yaml
before_script:
  # Initialize submodules
  - git submodule update --init --recursive
  
  # For faster CI, skip LFS large files if not needed
  - GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

---

## 📊 Workflow Summary

### Quick Reference

```bash
# Initial setup (once)
git submodule add <url> models

# Clone with submodules
git clone --recursive <url>

# Update submodules after clone
git submodule update --init --recursive

# Update submodule to latest
cd models/ && git pull origin main && cd ..
git add models && git commit -m "Update models"

# Check submodule status
git submodule status

# Work on submodule
cd models/
git checkout main
# make changes
git commit && git push
cd .. && git add models && git commit
```

---

## 🎓 Conceptual Understanding

### The Key Mental Model

Think of submodules as **snapshots**:

1. `BioImaging-Models` is a **living repository** that evolves
2. `BioImaging` **takes snapshots** of `BioImaging-Models` at specific points
3. Each commit in `BioImaging` says: "Use models at commit abc123"
4. Updating the submodule means: "Now use models at commit def456"

### State Diagram
```bash
┌─────────────────────────────────────────┐
│ BioImaging Repository │
│ ┌───────────────────────────────────┐ │
│ │ Code: src/, notebooks/ │ │
│ └───────────────────────────────────┘ │
│ ┌───────────────────────────────────┐ │
│ │ Reference: models @ commit abc123 │ │
│ │ (stored in .git, not actual files)│ │
│ └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
↓
points to
↓
┌─────────────────────────────────────────┐
│ BioImaging-Models Repository │
│ Commit abc123 (June 2025) │
│ Commit def456 (July 2025) ← latest │
│ Commit ghi789 (Aug 2025) │
└─────────────────────────────────────────┘
```


---

## 🔗 Integration with BioImaging

### Recommended Directory Structure
```bash
BioImaging/
├── src/
│ └── imaging/
│ └── inference.py ← Loads models from ../models/
├── models/ ← Submodule
│ ├── vascx/
│ └── retfound/
├── notebooks/
│ └── analysis.ipynb ← References ../models/
└── README.md
```


### Loading Models in Code

```python
# src/imaging/inference.py
import torch
from pathlib import Path

# Use relative path to submodule
MODEL_DIR = Path(__file__).parent.parent.parent / "models"

def load_vascx_model():
    model_path = MODEL_DIR / "vascx" / "vessels" / "vessels_july24.pt"
    return torch.load(model_path)
```

### Documentation

Update `BioImaging/README.md`:

```markdown
## Installation

1. Clone with submodules:
   ```bash
   git clone --recursive git@gitlab.cnic.es:your-group/bioimaging.git
   ```

2. If already cloned without submodules:
   ```bash
   git submodule update --init --recursive
   ```

3. Download model files (requires Git LFS):
   ```bash
   cd models
   git lfs pull
   ```
```

---

## 📚 Additional Resources

- [Official Git Submodules Documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [Atlassian Git Submodules Tutorial](https://www.atlassian.com/git/tutorials/git-submodule)

---

## ❓ FAQ

**Q: Should I commit inside the submodule directory?**  
A: Yes! Submodules are full repositories. Commit and push in the submodule, then update the parent to track the new commit.

**Q: Can I have submodules inside submodules?**  
A: Yes (nested submodules), but keep it simple when possible.

**Q: What if I accidentally modify submodule without being on a branch?**  
A: Create a branch from the current state: `git checkout -b temp-branch`, commit, then merge.

**Q: Do I need to push the submodule separately?**  
A: Yes! Push in submodule directory, then push parent repository.

**Q: Can multiple projects share the same model repository?**  
A: Absolutely! That's a key benefit. Multiple parent repos can reference the same model repo.

---

**Congratulations!** You now understand git submodules.

🎉 Happy coding!
