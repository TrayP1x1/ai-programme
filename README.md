# AI programme: 24 weeks to building AI systems independently

A project-based programme, about 10 hours a week, that rebuilds the core of an AI bachelor's (Python, algorithms, maths, ML, IR, NLP, neural networks, vision) through implementation. It ends in a healthcare capstone and a reproducible GitHub portfolio.

> **About "working through it on autopilot":** the notebooks take care of *planning*. Each week tells you exactly what to do, in order, and how to know you're done. But every exercise is designed to make you *think*. Each one has a test that tells you when you've got it, and hints if you're stuck. The struggle is the part that builds skill.

## What's in here

| File | What it's for |
|---|---|
| `ROADMAP.md` | All 24 weeks: objectives, resources, milestones, success criteria, catch-up rules, the capstone design, the compute plan, industry skills, and VU admission notes |
| `notebooks/00_diagnostic.ipynb` | **Start here.** A 2.5-hour assessment of where you are now |
| `notebooks/01_…` to `04_…` | Month 1: Python, debugging, Git, algorithms, a mini search engine |
| `notebooks/05_…` to `08_…` | Month 2: linear algebra, SVD, gradients, MLE/MAP, logistic regression, evaluation, leakage, statistical comparison |
| `notebooks/09_…` to `12_…` | Month 3: text processing, BM25, ranking metrics, classification, embeddings, hybrid search (capstone Experiment A) |
| `notebooks/13_…` to `16_…` | Month 4: neural nets from scratch, PyTorch, CNNs, transfer learning, GPUs, calibration (capstone Experiment B) |
| `notebooks/17_…` to `20_…` | Month 5: protocol, packaging, final experiments, SQL experiment tracking |
| `notebooks/21_…` to `24_…` | Month 6: testing and CI, reproducibility, API + Docker, report, job prep, final assessment |
| `course/` | Dataset loaders used from Month 3 onwards (they download once into `data/`) |
| `COMPETENCY_TRACKER.md` | Your skills, levels and evidence |
| `LOG.md` | Weekly hours, hints used, what was hard |
| `work/` | Where your weekly modules and tests go (Months 1–2) |
| `capstone/` | Created in Week 9: your portfolio project, with its own Git repo (`medsearch-lab`) |
| `requirements.txt` | The Python packages you need |

Every notebook was run from start to finish, in order, with reference answers, so you know each exercise can be solved and each checker works. The reference answers are **not** included. You send me your work and I review it.

## Each week
1. Open that week's notebook and follow Day 1 → Day 4 → Review.
2. Work on a branch (`git switch -c weekNN`), commit often, open a PR, and merge it.
3. Fill in `LOG.md`.
4. **Send me** the PR link or the notebook. I review your code and explanations, update your tracker, and send any remedial exercises.

**AI rule:** don't use AI to write exercise code. Asking for an *explanation of a concept* after 20 minutes of trying is fine.

---

## Setup (Week 1, Day 1)

You're using **Ubuntu in WSL** on Windows. Run every command below in the **Ubuntu terminal**, not PowerShell.

### 1. System packages
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git unzip
python3 --version      # 3.10 or newer is fine
```

### 2. Put the course folder in your Linux home
Download `ai-programme.zip` from our project. If it went to your Windows *Downloads* folder:
```bash
cp /mnt/c/Users/<YOUR_WINDOWS_NAME>/Downloads/ai-programme.zip ~
cd ~ && unzip ai-programme.zip && cd ai-programme
```
Keep your work in `~/` (the Linux file system), **not** under `/mnt/c/...`. Files on the Windows side are much slower to access from WSL.

### 3. Virtual environment
A virtual environment keeps this project's packages separate from everything else on your machine. That's the first step towards reproducibility.
```bash
python3 -m venv .venv
source .venv/bin/activate          # do this every time you open a new terminal
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU build: ~200 MB instead of several GB
pip install -r requirements.txt
```

Installing PyTorch's CPU build *before* the requirements matters: otherwise `sentence-transformers` pulls in the default GPU build, which is several GB and useless on your laptop.

### 4. VS Code
1. Install VS Code **on Windows**, plus the **WSL** extension.
2. In the Ubuntu terminal, inside `~/ai-programme`, run `code .`. VS Code opens, connected to WSL.
3. Install the **Python** and **Jupyter** extensions (VS Code installs them on the WSL side).
4. Open `notebooks/00_diagnostic.ipynb` → **Select Kernel** → **Python Environments** → choose `.venv`.

### 5. Git and GitHub
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# SSH key, so you can push without typing a password
ssh-keygen -t ed25519 -C "you@example.com"     # press Enter for the defaults
cat ~/.ssh/id_ed25519.pub                       # copy this line
```
On GitHub, go to **Settings → SSH and GPG keys → New SSH key**, paste the line, and save. Then create a **private** repository called `ai-programme`, *without* a README. In the terminal:
```bash
cd ~/ai-programme
git init -b main
git add .
git commit -m "Add course materials"
git remote add origin git@github.com:<YOUR_GITHUB_NAME>/ai-programme.git
git push -u origin main
```
Now run the first code cell in `notebooks/01_python_foundations_1.ipynb`. If it prints the package versions, setup is done.

### Troubleshooting
- **`ModuleNotFoundError` in a notebook:** the wrong kernel is selected. Choose `.venv`.
- **`pip install` fails with "externally-managed-environment":** you forgot `source .venv/bin/activate`.
- **`git push` asks for a password or is denied:** the SSH key isn't on GitHub, or the remote uses `https://`. Check with `git remote -v`.

---

*All capstone models in this programme are research experiments. They are **not clinically validated** and must never be used for diagnosis or treatment. Only public, appropriately licensed data is used; no confidential patient information.*
