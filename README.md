# Taglish Smishing Detection: Prototype

Monorepo for the thesis prototype (Chapter 3, Section 3.3).

    ml/        corpus prep, training, evaluation (Python)
    server/    FastAPI inference + SHAP (Increment 3)
    android/   Kotlin client (Increment 4)
    tests/     preprocessing contract tests
    data/      NOT in git: put raw CSVs in data/raw/ locally

## Rules that protect the results chapter

1. `ml/textnorm.py` is the only preprocessing code. The server imports it.
   If you change it, rerun the tests and retrain.
2. `python ml/prepare_corpus.py --write` runs **once**, at corpus freeze
   (Oct 21). It refuses to overwrite the split afterwards. Never delete
   `data/processed/corpus.csv` to "redo" the split.
3. The **test** split is not opened until the final configuration is
   frozen. Tune and compare on **val** only.
4. Never commit data, model weights, or `audit_key_DO_NOT_SHARE.csv`.

## Git workflow (for every change)

    git checkout main
    git pull                          # get everyone's latest work
    git checkout -b yourname/short-topic
    # ... edit files ...
    git add -A
    git commit -m "What you changed, in one line"
    git push -u origin yourname/short-topic

Then open a Pull Request on GitHub into `main`; one lead developer reviews
and merges. Do not push directly to `main` (it is protected).
If `git pull` reports a conflict, stop and ask a lead developer.

## Run the checks

    pip install -r requirements.txt
    python -m pytest tests/
    python ml/prepare_corpus.py        # dry run, prints corpus stats
