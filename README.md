# DSA4213 Assignment 1

Edward Chew (A0271710N)

Everything is in `Assignment1.ipynb`: n-gram / RNN / LSTM / Transformer language models on three Jane Austen novels (Part I), the embedding ablation on the LSTM (Part II), and SST-2 classification from the frozen LM features (Part III). The report is `report/report.pdf`.

## Running it

Python 3.11, run from this folder. I used my laptop RTX 4060 (8 GB); the five LM trainings take about 13 minutes of GPU time in total (the rest of the notebook adds a few more minutes, plus the dataset downloads).

```
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Assignment1.ipynb
```

(or just open the notebook and Run All). The CPU build of torch also works but is much slower.

The notebook creates `data/`, `checkpoints/`, `results/` and `report/figures/` itself, so those are not in the repo.

## Data

Nothing is included, it all downloads when the notebook runs:

- Austen (Emma, Persuasion, Sense and Sensibility): `nltk.download('gutenberg')`
- GloVe 6B 100d: `gensim.downloader`, `glove-wiki-gigaword-100`
- SST-2: HuggingFace `datasets`, `nyu-mll/glue`

Preprocessing (chapter-based 80/10/10 split per novel, lowercasing, NLTK tokenisation, train-only vocabulary with min count 3) is described in section 1 of the notebook.

## Seeds and settings

Seed is 42 everywhere (python, numpy, torch, gensim, SST-2 sampling). Hyperparameters for every model are in the notebook (section 1.4 and the model cells), and the learning rate for each architecture was picked from {1e-3, 3e-3} on validation perplexity (the sweep cell).

## Attribution

Libraries: PyTorch, gensim, scikit-learn, NLTK, HuggingFace datasets. I used Claude Code as a coding assistant and for debugging, and to draft the report based on my findings.
