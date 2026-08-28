# GitHub setup

This folder is repository-ready but is not automatically pushed to GitHub by the reproducibility script.

From the project root:

```bash
git init
git add .
git commit -m "Initial reproducible capstone submission"
git branch -M main
```

Create an empty repository on GitHub, then connect it:

```bash
git remote add origin https://github.com/<USERNAME>/<REPOSITORY>.git
git push -u origin main
```

The full global UCDP file is intentionally ignored by `.gitignore` because it exceeds GitHub's normal per-file size limit. The compact versioned source snapshot required to reproduce the submitted modeling table is included under `data/interim/`.

Before submission, verify from a fresh environment:

```bash
pip install -r requirements.txt
pytest
python main.py --rebuild-data
python main.py
```
