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

## Audit workflow (Section 3.6.4)

    python ml/prepare_corpus.py --audit    # makes data/audit/audit_input_blind.csv
    python ml/ai_prelabel.py batches       # 18 prompt files in data/audit/ai_batches/
    # paste each into a NEW AI chat; save the reply as data/audit/ai_outputs/batch_XX.csv
    python ml/ai_prelabel.py collect       # validates, writes data/audit/ai_labels.csv
    python ml/make_audit_sheets.py         # one blind sheet per annotator
    python ml/audit_agreement.py           # kappa, after the sheets are filled

The person running the AI pre-labeling must not read ai_labels.csv.

## SHA-256 Hash of the corpus CSV

Check it with `python scripts/check_corpus_hash.py` (exits non-zero on a mismatch).

   - c0a295ab80e6c82b5f0d7a58cc07514b3d0c737a91b6ba193fd3d1211b41a985 (corpus.csv)
     Recorded 2026-10-23, the day of the freeze. An earlier value in this file was a placeholder.
   - 8489bff3527aee7e6132ef808347637930ec0ef3a7a9344cb737202a80362386 (corpus_kaggle.csv)