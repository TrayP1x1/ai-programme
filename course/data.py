"""Dataset loaders used from Month 3 onwards.

Each loader downloads its dataset once into ``ai-programme/data/`` and reads it from
disk afterwards. ``data/`` is in ``.gitignore``: datasets never go into Git.

Set the environment variable ``COURSE_FAKE_DATA=1`` to get small synthetic stand-ins
with exactly the same format. That is how the notebooks are tested offline, and it is
handy for writing fast unit tests of your own pipeline code.
"""
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def fake_mode():
    return os.environ.get("COURSE_FAKE_DATA") == "1"


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


# ---------------------------------------------------------------- NFCorpus (BEIR)

NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"


def load_nfcorpus(split="test"):
    """Load the NFCorpus retrieval benchmark (medical/nutrition), as distributed by BEIR.

    Returns ``(corpus, queries, qrels)``:
      corpus  -- {doc_id: {"title": str, "text": str}}
      queries -- {query_id: str}, only queries judged in ``split``
      qrels   -- {query_id: {doc_id: relevance}}, relevance is 1 or 2 (higher = more relevant)
    ``split`` is "train", "dev" or "test". Tune on "dev", report on "test".
    """
    if fake_mode():
        return _fake_nfcorpus(split)
    folder = DATA_DIR / "nfcorpus"
    if not folder.exists():
        z = _download(NFCORPUS_URL, DATA_DIR / "nfcorpus.zip")
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA_DIR)
    corpus = {}
    with open(folder / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d["text"]}
    qrels = {}
    with open(folder / "qrels" / f"{split}.tsv") as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    queries = {}
    with open(folder / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in qrels:
                queries[q["_id"]] = q["text"]
    return corpus, queries, qrels


_TOPICS = {
    "cardio": "heart cardiovascular cholesterol blood pressure artery statin stroke hypertension cardiac lipid",
    "diabetes": "diabetes insulin glucose sugar glycemic type obesity metabolic pancreas weight",
    "cancer": "cancer tumor breast prostate colon carcinogen chemotherapy oncology malignant risk",
    "gut": "gut microbiome bacteria fiber intestinal probiotic digestion bowel flora colitis",
    "bone": "bone calcium vitamin osteoporosis fracture density dairy skeletal mineral hip",
    "brain": "brain cognitive dementia alzheimer memory neuron cognition decline mental neural",
    "kidney": "kidney renal protein sodium stone urinary dialysis nephropathy filtration urine",
    "infant": "infant breastfeeding pregnancy maternal birth neonatal child fetal milk prenatal",
}
_COMMON = "study results patients effect diet intake trial association increased reduced data analysis group levels".split()


def _fake_nfcorpus(split):
    rng = np.random.default_rng(0)
    topics = {k: v.split() for k, v in _TOPICS.items()}
    names = list(topics)
    corpus, doc_topic = {}, {}
    for t_i, topic in enumerate(names):
        for j in range(50):
            other = topics[names[int(rng.integers(len(names)))]]
            toks = []
            for _ in range(int(rng.integers(20, 120))):
                u = rng.random()
                pool = topics[topic] if u < 0.2 else other if u < 0.35 else _COMMON
                toks.append(pool[int(rng.integers(len(pool)))])
            did = f"MED-{t_i * 50 + j}"
            corpus[did] = {"title": f"study {j}", "text": " ".join(toks)}
            doc_topic[did] = topic
    queries, qrels = {}, {}
    offset = {"train": 0, "dev": 1, "test": 2}[split]
    for t_i, topic in enumerate(names):
        words = topics[topic]
        for r in range(3):
            qid = f"PLAIN-{split}-{t_i}-{r}"
            q = [words[(offset * 3 + r + k) % len(words)] for k in range(2)]
            queries[qid] = " ".join(q)
            rel = {}
            for did, tp in doc_topic.items():
                if tp == topic:
                    toks = corpus[did]["text"].split()
                    rel[did] = 2 if any(w in toks for w in q) else 1
            qrels[qid] = rel
    return corpus, queries, qrels


# ---------------------------------------------------------------- PubMed RCT

PUBMED_RCT_URL = ("https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/"
                  "PubMed_20k_RCT_numbers_replaced_with_at_sign/{split}.txt")


def load_pubmed_rct(split="train", max_sentences=None):
    """Sentences from randomised-controlled-trial abstracts, labelled with their role:
    BACKGROUND, OBJECTIVE, METHODS, RESULTS or CONCLUSIONS (PubMed 20k RCT).

    Returns ``(texts, labels, abstract_ids)`` as three lists of equal length.
    ``split`` is "train", "dev" or "test". Keep ``abstract_ids`` -- sentences from the
    same abstract must never be split across train and test.
    """
    if fake_mode():
        return _fake_rct(split, max_sentences)
    path = _download(PUBMED_RCT_URL.format(split=split), DATA_DIR / "pubmed_rct" / f"{split}.txt")
    texts, labels, ids = [], [], []
    current = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("###"):
                current = line[3:]
            elif line.strip():
                label, sentence = line.split("\t", 1)
                texts.append(sentence)
                labels.append(label)
                ids.append(current)
                if max_sentences and len(texts) >= max_sentences:
                    break
    return texts, labels, ids


_RCT_VOCAB = {
    "BACKGROUND": "disease common burden prevalence important unclear little known worldwide",
    "OBJECTIVE": "aim objective evaluate investigate assess determine whether purpose",
    "METHODS": "randomized assigned participants received placebo measured enrolled double blind",
    "RESULTS": "significantly compared observed difference ci p versus reduction higher",
    "CONCLUSIONS": "suggest conclude may effective safe further warranted findings support",
}


def _fake_rct(split, max_sentences):
    rng = np.random.default_rng({"train": 0, "dev": 1, "test": 2}[split])
    n_abs = {"train": 150, "dev": 40, "test": 40}[split]
    texts, labels, ids = [], [], []
    for a in range(n_abs):
        for label, words in _RCT_VOCAB.items():
            words = words.split()
            toks = [words[k % len(words)] if rng.random() < 0.5 else _COMMON[k % len(_COMMON)]
                    for k in rng.integers(0, 20, size=int(rng.integers(8, 20)))]
            texts.append(" ".join(toks))
            labels.append(label)
            ids.append(f"{split}{a}")
    if max_sentences:
        texts, labels, ids = texts[:max_sentences], labels[:max_sentences], ids[:max_sentences]
    return texts, labels, ids


# ---------------------------------------------------------------- MedMNIST


def load_medmnist(name="pneumoniamnist", size=28):
    """Load a MedMNIST v2/v3 dataset (default: PneumoniaMNIST, paediatric chest X-rays).

    Returns {"train": (X, y), "val": (X, y), "test": (X, y)} with
    X uint8 of shape (N, size, size) and y int64 of shape (N,).
    Requires ``pip install medmnist``. Sizes: 28, 64, 128, 224.
    """
    if fake_mode():
        return _fake_medmnist(size)
    import medmnist
    from medmnist import INFO

    cls = getattr(medmnist, INFO[name]["python_class"])
    root = DATA_DIR / "medmnist"
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ["train", "val", "test"]:
        kwargs = {"split": split, "download": True, "root": str(root)}
        if size != 28:
            kwargs["size"] = size
        ds = cls(**kwargs)
        out[split] = (np.asarray(ds.imgs), np.asarray(ds.labels).reshape(-1).astype(np.int64))
    return out


def _fake_medmnist(size):
    rng = np.random.default_rng(0)
    out = {}
    yy, xx = np.mgrid[0:size, 0:size]
    for split, n in [("train", 600), ("val", 150), ("test", 200)]:
        y = (rng.random(n) < 0.7).astype(np.int64)
        X = rng.normal(90, 25, size=(n, size, size))
        for i in np.where(y == 1)[0]:
            cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
            r = size / 6
            X[i] += 28 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        out[split] = (np.clip(X, 0, 255).astype(np.uint8), y)
    return out
