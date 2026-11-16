# 24-week roadmap: from "I passed the courses" to "I can build it myself"

**Learner profile (October 2026):** a third-year AI bachelor's student at UvA with about **10 hours a week**. Python: beginner, comfortable with functions but rusty on classes. Shaky areas: linear algebra (eigenvalues, SVD), the intuition behind MLE and MAP, choosing loss and activation functions, and statistical model comparison. Hardware: a Lenovo with a Ryzen 7 5000-series CPU, integrated Radeon graphics, 16 GB of RAM, running Ubuntu on WSL. Target: the **VU MSc Artificial Intelligence, AI for Health track**.

**Total:** 24 weeks × 10 h ≈ 240 hours. That is roughly 10 ECTS of focused work, so the plan trims breadth and keeps depth. Topics marked **(optional)** are the first to drop if you fall behind.

**The weekly rhythm:** 4 sessions (about 2–2.5 h each) plus a 30-minute review. Every week ends with spaced recall from earlier weeks, a blank-page rebuild of one key function, a commit or PR, and a short entry in `LOG.md`.

**All 25 notebooks (Weeks 0–24) are ready** in `notebooks/`. Each one was run from start to finish with reference answers, in order, the way you'll work through them. If the diagnostic or a later week shows that something needs more (or less) time, I'll restructure the affected notebooks. The plan is a contract, but the details can move.

**Two goals, one programme:** (1) the skills for a health-AI master's, and (2) **job-readiness**, meaning you'd hold your own in a junior AI/ML engineering role or a side job after your bachelor's. See "Industry skills" below for where each workplace skill is taught.

**Adjustments after the Week 0 diagnostic (November 2026):**
- *Python basics are stronger than assumed* (A1–A4 all passed). Week 1 Day 1 is now Git practice, since setup was already done and Git beyond add/commit/push was flagged. Exercises 1.1–1.3 are a timed speed run.
- *Conditional probability needs work* (C5: Bayes' rule known, but the translation from words was missing). Week 7 now starts with a 1-hour warm-up (W.1–W.3) before Bayes' rule.
- *Kept at full time, as planned:* recursion and binary search (Weeks 3–4), all of Month 2, and leakage and model comparison (Week 8).

---

## The capstone (built gradually from Week 4)

**Working title:** *MedSearch Lab: baselines versus learned models for biomedical text retrieval and medical image classification.*

There are **two separate experiments that share one repository**, plus shared evaluation, configuration and reporting code. They stay separate on purpose, so neither depends on the other working.

| | Experiment A: literature retrieval | Experiment B: image classification |
|---|---|---|
| Research question | On a public biomedical retrieval benchmark, how much do dense embeddings and a hybrid ranker improve over a BM25 baseline you implement yourself, and is the difference statistically significant across queries? | On a public chest X-ray dataset, how much does transfer learning improve over logistic regression and a small CNN, measured by AUROC with confidence intervals, and how well calibrated are the models? |
| Data | NFCorpus (a medical/nutrition retrieval benchmark with relevance judgements, available through `ir_datasets`/BEIR). Fallback: TREC-COVID. | PneumoniaMNIST from MedMNIST v2 (small images, runs on a CPU). Stretch goal: the 224×224 version on a GPU. |
| Baseline | Boolean search → TF-IDF → **BM25 from scratch** (NumPy and dicts) | Logistic regression on pixels |
| Improved models | Sentence-embedding retrieval; hybrid BM25 + embeddings | Small CNN → pretrained ResNet-18 (transfer learning) |
| Metrics | nDCG@10, MAP, Recall@100, a paired significance test across queries | AUROC, sensitivity and specificity at a fixed threshold, calibration (ECE, reliability plot), bootstrap 95% CIs |
| Measurable objectives | Your BM25 matches a reference library to within 0.01 nDCG@10; the results table is reproduced with one command | The baseline, CNN and transfer models are each trained with 3 seeds; the results table is reproduced with one command |

**Data and ethics rules:** only public, appropriately licensed data. You'll write a licence entry for every dataset in `DATASETS.md` in Week 9 and Week 15. No confidential patient information, ever. Every README, notebook and report states: ***these models are research experiments, not clinically validated, and must not be used for diagnosis or treatment.***

**Optional integration (Week 20):** a small Streamlit demo with a "search abstracts" tab and an "image model" tab, carrying that disclaimer.

---

## Month 1: Python, debugging, Git, algorithms (Weeks 0–4) ✅ notebooks ready

| Wk | Objectives | Notebook / deliverable | Success criteria |
|---|---|---|---|
| 0 | Measure your starting point honestly | `00_diagnostic.ipynb` (2.5 h, can sit inside Week 1) | Fully attempted and sent to me; blanks are fine |
| 1 | Setup check; **Git practice** (restore, amend, stash, reset vs revert, merge conflicts, reflog); values, lists, loops, functions; strings and dicts; reading tracebacks; the debugging method | `01_python_foundations_1.ipynb`; repo on GitHub | All tests pass; rebuild `tokenize`/`word_counts` from a blank page in under 10 min; 3+ commits |
| 2 | Comprehensions, sets, sorting with `key`; files, CSV, exceptions; modules plus **pytest**; the VS Code debugger; branches, PRs, `.gitignore` | `02_python_foundations_2.ipynb`; `work/week02/text_utils.py` with 6+ of your own tests | pytest green; first merged PR |
| 3 | Classes and dataclasses; big-O by measurement; hash-based structures; binary search with invariants; randomised testing; package layout; test-first development | `03_classes_and_complexity.ipynb`; `work/week03/vocab/` package | Binary search from a blank page; 6+ tests; can explain why sets are O(1) |
| 4 | Recursion, merge sort, stacks, queues, trees; **mini search engine** (inverted index); **cumulative check 1** | `04_recursion_and_mini_search.ipynb`; `work/week04/minisearch/` | 8+ tests; run-length encoding done in ≤ 60 min with no hints; tracker updated |

**Resources (pick one per row; don't collect resources):**
- *Python:* the official [Python tutorial](https://docs.python.org/3/tutorial/), chapters 3–9. Read the chapter that matches the week, *after* trying the exercises.
- *Git:* [Pro Git](https://git-scm.com/book/en/v2), chapters 1–3, and MIT's [Missing Semester](https://missing.csail.mit.edu/), the "Version Control" lecture. Learn the model behind it (snapshots, branches as pointers), not just the commands.
- *Algorithms:* [Problem Solving with Algorithms and Data Structures using Python](https://runestone.academy/ns/books/published/pythonds3/index.html), chapters on analysis, recursion, and sorting and searching. Implement the examples yourself, then compare.

---

## Month 2: Mathematics, classical ML, statistical evaluation (Weeks 5–8) ✅ notebooks ready

Your diagnostic flagged this month as the most important for understanding. Every concept follows the same pattern: **intuition → derivation on paper → numerical example → NumPy implementation → the place it appears in ML.**

| Wk | Objectives | Exercises (NumPy first) | Success criteria |
|---|---|---|---|
| 5 | **Linear algebra as geometry:** vectors, dot products and similarity, matrices as transformations, solving $Ax=b$, **eigenvalues and eigenvectors** (what $\lambda$ *means*: the directions a matrix only stretches), PCA as eigenvectors of the covariance matrix | Implement cosine similarity; visualise 2×2 matrices transforming a grid; power iteration to find the top eigenvector; PCA from scratch on a small dataset, checked against scikit-learn | Find a 2×2 eigendecomposition by hand and explain it without formulas; your PCA matches sklearn up to sign |
| 6 | **SVD** (rotate → stretch → rotate; the best low-rank approximation); **calculus for ML:** derivatives, partial derivatives, the chain rule, gradients as the direction of steepest ascent; **gradient descent**; numerical gradient checking | Compress an image with a truncated SVD; latent semantic analysis (SVD of a term-document matrix from your minisearch corpus); gradient descent on a quadratic bowl with different learning rates; a `grad_check` utility | Explain why SVD gives the best rank-k approximation (intuition); diagnose divergence from a loss plot |
| 7 | **Warm-up:** turning words into conditional probabilities, natural frequencies; **Probability → loss functions:** distributions; **MLE** (choosing the parameters that make the data most probable) and **why we take logs** (products become sums, numerical stability); MSE as Gaussian MLE; cross-entropy as Bernoulli/categorical MLE; **MAP = MLE + prior = regularisation** (Gaussian prior → L2); **logistic regression from scratch**, including the sigmoid derivative and its gradient | Derive the coin MLE and MAP; plot log-likelihood surfaces; logistic regression with gradient descent in NumPy, checked with `grad_check`, compared with sklearn; add an L2 penalty and show it equals a Gaussian prior | Derive the logistic regression gradient on paper from a blank page; your implementation matches sklearn's accuracy within 1% |
| 8 | **Evaluation and generalisation:** train/validation/test splits, cross-validation, overfitting, bias and variance, **data leakage** (preprocessing fitted on test data, patient-level splits, tuning on test); metrics (precision, recall, F1, ROC-AUC, calibration); **comparing two models:** bootstrap confidence intervals, McNemar's test, paired permutation tests, why "87% beats 85%" can be noise, and multiple comparisons; scikit-learn `Pipeline` | Build a leaky pipeline on purpose and measure how much it inflates the results; implement bootstrap CIs and a permutation test from scratch; compare two classifiers properly; **cumulative check 2** | Find all the leaks in a pipeline you haven't seen before; report a model comparison with a CI and a p-value and interpret both correctly |

**Buffer:** the second half of Week 8 is catch-up time. If you're behind, drop the **(optional)** pieces: LSA in Week 6 and calibration in Week 8. Never drop leakage or grad-checking.

**Resources:**
- 3Blue1Brown, [*Essence of Linear Algebra*](https://www.3blue1brown.com/topics/linear-algebra), chapters 1–4, 7 and 14 (eigen). Watch these *before* the derivations; they repair intuition fast.
- [*Mathematics for Machine Learning*](https://mml-book.github.io/) (Deisenroth, Faisal, Ong; free PDF): chapter 4.2–4.5 (eigen, SVD), chapter 5 (vector calculus), chapter 6 (probability), chapter 8.3 (MLE and MAP). Do the derivations, and skip the proofs on a first pass.
- [*An Introduction to Statistical Learning, with Python*](https://www.statlearning.com/) (ISLP, free): chapters 2, 4, 5 and 6 for generalisation, classification, resampling and regularisation.
- scikit-learn's [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) page, on data leakage.
- Dror et al. (2018), *The Hitchhiker's Guide to Testing Statistical Significance in NLP* (ACL), for which test to use when comparing models.

---

## Month 3: Information retrieval and NLP (Weeks 9–12) — Capstone Experiment A begins ✅ notebooks ready

| Wk | Objectives | Exercises / capstone milestone | Success criteria |
|---|---|---|---|
| 9 | Text processing with NLTK (tokenisation, stop words, stemming); loading NFCorpus; writing `DATASETS.md` (source, licence, size, known issues); scaling the minisearch index to thousands of documents with good data structures | Turn `minisearch` into `medsearch/` in the capstone repo; preprocessing pipeline with tests; profile indexing time | **M1:** the index builds over NFCorpus in under a minute with tests passing; `DATASETS.md` written |
| 10 | **TF-IDF** (why rare words matter); cosine ranking; **BM25** (term-frequency saturation `k1`, length normalisation `b`), derived and implemented; **ranking metrics** (P@k, MAP, nDCG) implemented from scratch | BM25 from scratch; compare with the `rank_bm25` library; small grid search over `k1` and `b` on **validation** queries only | **M2:** BM25 baseline nDCG@10 within 0.01 of the reference library; the metrics pass hand-computed test cases |
| 11 | **Text classification:** naive Bayes and logistic regression on TF-IDF; **embeddings:** distributional semantics, word2vec intuition, sentence embeddings; dense retrieval with a small pretrained sentence-transformer (runs on a CPU) | Classify abstracts by topic; dense retrieval on NFCorpus, using cached embeddings | **M3:** a dense retrieval run evaluated with the same metrics code |
| 12 | Hybrid ranking (score fusion); **paired significance tests across queries**; error analysis (read the failures!); **cumulative check 3**; buffer | A table comparing BM25, dense and hybrid with CIs and p-values; 5 annotated failure cases | **M4:** Experiment A's first results table, reproduced with `python -m medsearch.evaluate --config configs/bm25.yaml` |

**Resources:**
- Manning, Raghavan and Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/) (free): chapter 1 (inverted index), 2 (tokenisation), 6 (TF-IDF and the vector space model), 8 (evaluation), 11.4 (BM25). The core text for Experiment A.
- Jurafsky and Martin, [*Speech and Language Processing*, 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3/) (free): chapter 2 (text normalisation), the naive Bayes and logistic regression chapters, and the vector semantics and embeddings chapter.
- The [NLTK book](https://www.nltk.org/book/), chapters 1–3, used as a reference rather than read cover to cover.
- (optional) Thakur et al. (2021), *BEIR*, to see where NFCorpus comes from and how strong BM25 is in practice.

---

## Month 4: Neural networks, PyTorch, computer vision, GPUs (Weeks 13–16) — Experiment B begins ✅ notebooks ready

| Wk | Objectives | Exercises / capstone milestone | Success criteria |
|---|---|---|---|
| 13 | **Neural networks from scratch:** the MLP forward pass; **why non-linear activations** (stacked linear layers collapse into one); sigmoid, tanh and ReLU and their gradients; **backpropagation as the chain rule** on a computation graph; initialisation | A micrograd-style scalar autograd engine; a NumPy MLP on a toy dataset with gradient checking | Derive backprop for a 2-layer MLP on paper; the NumPy MLP passes the gradient check |
| 14 | **PyTorch:** tensors, autograd, `nn.Module`, `DataLoader`, the training loop; **training diagnostics:** overfit one batch, the learning-rate range test, loss curves, exploding and vanishing gradients; seeds and determinism | Port your NumPy MLP to PyTorch and check that the outputs match; a deliberately broken training-loop debugging exercise | Find 5 planted bugs in a training script; get the same results from the same seed |
| 15 | **Image processing and CNNs:** images as arrays, convolution implemented by hand, pooling, receptive fields; MedMNIST and its licence; patient-level considerations; class imbalance | Convolution in NumPy checked against PyTorch; logistic regression baseline and a small CNN on PneumoniaMNIST **on the CPU** | **M5:** baseline and CNN AUROC with bootstrap CIs; `DATASETS.md` updated |
| 16 | **Transfer learning** (a pretrained ResNet-18 used as a feature extractor, then fine-tuned); augmentation; **GPU computing on Colab or Kaggle**: device handling, mixed precision, checkpointing to Google Drive; calibration; buffer | Fine-tune on a GPU with 3 seeds; reliability diagram | **M6:** Experiment B results table; the same code runs on CPU and GPU via one config flag |

**Resources:**
- Andrej Karpathy, [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html), lecture 1 (micrograd), and lecture 2 (optional). Code along line by line, then rebuild it without the video.
- Michael Nielsen, [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/), chapter 2 (how backprop works).
- The PyTorch [*Learn the Basics*](https://pytorch.org/tutorials/beginner/basics/intro.html) tutorials, plus Karpathy's blog post [*A Recipe for Training Neural Networks*](https://karpathy.github.io/2019/04/25/recipe/), which is a diagnostics checklist you'll use for the rest of the course.
- [CS231n course notes](https://cs231n.github.io/) on convolutional networks and transfer learning.
- [MedMNIST v2](https://medmnist.com/) (Yang et al., 2023): read the paper's dataset table and check the licence for the specific subset you use.

---

## Month 5: Capstone integration, experiments, evaluation (Weeks 17–20) ✅ notebooks ready

| Wk | Objectives | Milestone | Success criteria |
|---|---|---|---|
| 17 | **Pre-register** `PROTOCOL.md` before any final run; package the capstone (`pyproject.toml`, `pip install -e .`); config validation; `logging`; an imaging CLI | `PROTOCOL.md`, `medlab/config.py`, `python -m medlab.imaging.run` | A bad config fails fast; both experiments run from one command |
| 18 | Final runs for Experiment A from the CLI; `medlab/report.py` builds the table with CIs, p-values and Holm correction; figures; the results write-up | **M7:** `table_A.md`, `fig_A_*.png`, `results_A.md` | The table is generated by code; every hypothesis is answered |
| 19 | Final runs for Experiment B: select on val, test once; thresholds chosen on val; sensitivity/specificity with CIs; ECE; a robustness probe | **M8:** `table_B.md`, `fig_B_*.png`, `results_B.md` | No test-set decisions; limitations stated |
| 20 | **SQL** with SQLite; an experiment tracker (every run logged with its config, metrics and Git commit); *(optional)* a Streamlit demo; **buffer** | `medlab/tracking.py`, `runs.db` | You can write JOIN / GROUP BY / HAVING queries without notes |

---

## Month 6: Testing, reproducibility, report, independence (Weeks 21–24) ✅ notebooks ready

| Wk | Objectives | Deliverable | Success criteria |
|---|---|---|---|
| 21 | Fixtures, `conftest.py`, parametrize, property tests for ML code; coverage; ruff lint/format; type hints; **GitHub Actions CI** | A green CI badge | Core coverage ≥ 80%; ruff clean |
| 22 | **Reproducibility** (a lock file, SHA-256 data checksums, `scripts/reproduce.sh`, a fresh-clone test); a **FastAPI** search service with tests; a **Dockerfile**; `MODEL_CARD.md`; make the repo public | A public repo; `uvicorn medlab.api:app` works | A fresh clone reproduces Table A exactly |
| 23 | **Research report** (11 required sections); the portfolio README; **job prep** (CV bullets, a 2-minute pitch, 15 interview questions about your project) | `report/report.md`, `README.md`, `INTERVIEW_PREP.md` | A friend can say what you found after reading the abstract |
| 24 | **Final independence assessment:** an unfamiliar regression package from scratch (3 h), a debugging challenge, derivations on paper, a fresh-clone reproduction, recorded explanations, the final competency matrix | `CHALLENGE.md`, `FINAL_REFLECTION.md`, tracker | See "What 'independent' means" below |

**Resources:** Patrick Mineault, [*The Good Research Code Handbook*](https://goodresearch.dev/), the whole thing (it's short); [*The Turing Way*](https://book.the-turing-way.org/) chapters on reproducibility; the GitHub Actions Python quick-start.

---

## Industry skills (for a side job or junior AI role)

| Skill employers ask for | Where you learn it | Evidence in your portfolio |
|---|---|---|
| Python, packaging, virtual environments | Weeks 1–3, 17 | `pyproject.toml`, `medlab/` |
| Git, branches, pull requests | Weeks 1–4, then every week | Your commit history |
| Testing (pytest), coverage, linting | Weeks 2, 9–12, 21 | `tests/`, CI badge |
| Data wrangling (pandas) | Weeks 9, 11, 18–19 | Notebooks, `report.py` |
| SQL | Week 20 | `medlab/tracking.py` |
| Classical ML and evaluation (scikit-learn) | Weeks 7–8, 11, 15 | Weeks 7–8 notebooks, baselines |
| Deep learning (PyTorch), GPU workflow | Weeks 13–16 | `medlab/imaging/` |
| NLP and search (BM25, embeddings, hybrid) | Weeks 9–12 | `medlab/ranking.py`, `dense.py`, `fusion.py` |
| Statistics for A/B-style model comparison | Weeks 8, 12, 18–19 | `medlab/stats.py`, Tables A and B |
| REST APIs (FastAPI), Docker | Week 22 | `medlab/api.py`, `Dockerfile` |
| CI/CD (GitHub Actions) | Week 21 | `.github/workflows/ci.yml` |
| Experiment tracking, reproducibility | Weeks 17, 20, 22 | `runs.db`, `reproduce.sh` |
| Communicating results | Weeks 18–19, 23 | `report/`, README, recorded pitch |

Not covered on purpose (they'd dilute the core in 240 hours): cloud deployment (AWS/GCP), large language model fine-tuning, and big-data tools (Spark). Once you've finished, each one is a good 2–4 week follow-up.

## What "independent" means at the end (Week 24)

You've reached the goal when, **without AI help and with only official documentation**, you can:
1. write modular Python with tests for a new small problem, and explain every line;
2. derive the gradient of logistic regression and backprop for a 2-layer MLP on paper;
3. implement BM25, nDCG, a bootstrap CI and a training loop from a blank page;
4. train and evaluate a model with a leakage-free split and a fair comparison;
5. debug a broken training script or pipeline methodically;
6. reproduce your capstone results from a clean clone.

Each of these is a row in `COMPETENCY_TRACKER.md` and needs **evidence**: a file, a PR or a recorded explanation.

---

## Catch-up rules (because life happens)

- **One bad week:** use the next buffer (Weeks 8, 12, 16 and 20). Don't skip the review sessions; they're the cheapest way to keep what you've learnt.
- **Two or more weeks behind:** drop the **(optional)** topics, and Experiment A's hybrid ranker if needed. Never drop tests, leakage, grad-checking or the reproducibility week.
- **A topic still shaky after a week:** tell me. I'll write a short remedial notebook instead of letting you carry the gap forward.
- **Exams:** in exam weeks, aim for 4–5 hours (one review plus one session). The plan has enough buffer for about 3 such weeks.

---

## Compute plan

| Weeks | Where | Why |
|---|---|---|
| 1–14 | **Your laptop, on the CPU** | 16 GB of RAM is plenty for NumPy, scikit-learn, BM25 over NFCorpus, and small PyTorch models |
| 15 | Laptop CPU | PneumoniaMNIST at 28×28 trains in minutes on a CPU |
| 16, 19 | **Google Colab (free T4 GPU)** or **Kaggle Notebooks (free weekly GPU quota)** | Fine-tuning ResNet-18. Save checkpoints to Drive. |
| optional | Colab Pro (about €10/month) | Only if the free sessions keep disconnecting |

Your Radeon graphics are **integrated**. PyTorch's AMD backend (ROCm) targets dedicated Radeon/Instinct GPUs and doesn't support integrated graphics under WSL, so plan on the CPU locally and a GPU in the cloud. **UvA/SURF:** SURF's Snellius cluster is normally accessed through a course, a thesis supervisor or a research grant, not on request by individual bachelor's students. This project doesn't need it; if you want it, ask your thesis supervisor or UvA research IT, and check SURF's current access rules for your situation.

---

## The VU MSc AI (AI for Health): what's formal and what's useful

*Checked October 2026 on vu.nl (the [admissions](https://vu.nl/en/education/master/artificial-intelligence/admissions) and [curriculum](https://vu.nl/en/education/master/artificial-intelligence/curriculum) pages). Re-check in spring, because requirements change between intakes.*

**Formal requirements** (these decide admission):
- A Dutch university (WO) bachelor's in AI is on the list of degrees that give **direct admission**. The page also asks for at least 18 ECTS in AI areas, which your degree should cover.
- Programming skills in Python or Java. The compulsory Python test applies only to applicants **without** an AI or CS bachelor's, so it shouldn't apply to you, but confirm this when you apply.
- Transcript and GPA uploaded; no minimum GPA is stated.
- **Deadline for Dutch-degree holders with Dutch/EU nationality: 1 June** (for a September start). Apply through Studielink.

**The AI for Health track** (from the curriculum page): developed with Medical Informatics at Amsterdam UMC. It covers medical informatics fundamentals, **medical imaging**, and **clinical NLP**, plus combining machine learning with reasoning methods. Two of its courses are taught at Amsterdam UMC.

**Useful skills** (not required, but they'll make the master's much easier): fluent Python and PyTorch, solid probability and linear algebra, evaluation and leakage awareness, NLP and retrieval, and imaging pipelines. That's exactly what the capstone exercises.

**Important:** this programme prepares you to *do well* in the master's. **It does not guarantee admission**, and a portfolio project isn't a formal admission criterion on the page above.
