# Project context and decision log (as of Oct 23, 2026)

Source of truth for everything decided before Claude Code joined. The manuscript master is in Google Docs; the latest review PDFs are the "v3" chapter files.

## 1. The thesis
- **System:** an explainable, multimodal, cloud-based smishing detector for Philippine SMS. A Kotlin Android client captures SMS (SMS_RECEIVED with RECEIVE_SMS only, plus share sheet, text selection and in-app paste; **no clipboard monitoring**). It masks phone numbers and OTPs on the device and sends JSON to FastAPI. The server preprocesses with `ml/textnorm.py`, classifies text with fine-tuned RoBERTa Tagalog, scores URLs with lexical features, fuses them (`S = w * p_text + (1 - w) * p_url`, w tuned on validation), computes SHAP on the text, and returns character offsets and weights, never message text (Chapter 3, Figure 3).
- **Model:** `jcblaise/roberta-tagalog-base` (confirmed: it downloaded successfully on Kaggle). **MAX_LEN = 256**, which covers more than 99.9% of train+val messages (median 41 tokens, p95 123, p99 166, max 264). 128 tokens would cover only 95.5% and would truncate long legitimate promos most often.
- **SOPs:** (1) text classification vs. non-contextual baselines, with per-subclass recall; (2) URL lexical features; (3) text + URL fusion vs. text-only (McNemar); (4) SHAP faithfulness (comprehensiveness and sufficiency vs. random-mask baseline, Wilcoxon) and UAT interpretability; (5) end-to-end latency (target under 5 s).
- **Timeline:** coding Oct 22 to 31; integration Nov 1 to 5; softbound printing Nov 1 to 7; **title defense Nov 8** (a working prototype is expected, not final evaluation); full evaluation (test metrics, latency, faithfulness, UAT) after the defense.
- **Group:** 6 members, two lead developers. Most members are new to git: keep instructions simple.

## 2. Label scheme (decided)
- **Positive class = malicious = smishing + spam. Negative = legitimate.** The three-class label (`label3`) is kept, so recall is reported separately for smishing and spam. The binary result is primary; the smishing/spam split is exploratory.
- **Codebook (Chapter 2, Section 2.3).** Smishing if at least one of:
  - **C1:** asks for credentials or verification.
  - **C2:** a non-official, lookalike or shortened link inside a deceptive message.
  - **C3:** impersonates a bank, e-wallet, telco, government agency or courier with a non-official link.
  - **C4:** a fraudulent pretext (already won or credited, advance fee, earn-money-for-tasks, unsolicited loan grant).

  Spam is an open promotion without these features. Legitimate is a genuine institutional message, or a transactional or personal message. Unclassifiable messages are excluded.
- **Decision rules (adopted):**
  - **C4 rule:** C4 applies only to *assertions* (already won, credited, selected, holds a ticket). Conditional offers ("register and get", "stand a chance to win") are spam.
  - **Rule 1:** a link alone never makes a message smishing. C2 applies only if the link imitates a known brand or the message misrepresents its sender. Gambling or shop ads linking to their own odd domain are spam.
  - **Rule 2:** a promo or notice from a well-known bank, e-wallet, telco or government agency, under its own name, with no suspicious link and no information request, is legitimate. Gambling is spam from any sender. Cold promotions from other businesses (property, hiring, venues, shops) are spam. A loan offer from an unnamed lender or a person, or one claiming "approved" or "qualified", is smishing (C4). A named bank describing its own loan product is legitimate. Easy-money task offers are C4.
  - **Rule 3:** a gambling lure written as a personal message or insider tip ("I always win", "there's a bug", "long time no see") is smishing (C4).
  - **Tie-breakers:** spam over smishing when torn. Legitimate only if a real bank, telco, e-wallet or government sender is clear.
- **Literature grounding (verified by a research pass):** the codebook is **the group's own operationalization, informed by** Wang et al. (2025, SOUPS; three-way phishing/spam/legitimate, spam does not deceive about the sender), Mishra and Soni (2020, 2022 dataset, 2023), Timko and Rahman (2024), Nahapetyan et al. (2024; excluded casinos and loans from phishing), the FTC, RA 12010 (AFASA) and BSP, and CICC/PAOCC advisories. Do **not** write "adapted from Sayın & Sandıkkaya (2026)": that paper is paywalled and the verifiable part supports only impersonation. Pritom et al. (2026) is an arXiv preprint. The decision rules have no direct literature support; present them as group conventions. The Herrera et al. (2022) URL in the references is wrong (it duplicates Cruz and Cheng) and still needs fixing.

## 3. Data pipeline and corpus (frozen Oct 23)
- **Sources:** Chua (2024) Kaggle PH Spam + Marketing SMS (categories ads/gov/notifs/spam/OTP; spam maps to candidate malicious, ads/gov/notifs to legitimate, OTP and withheld text excluded, `<REDACTED>` and `<BLANK>`). Wandowando (2025) Kaggle Philippine Spam/Scam SMS (all candidate malicious). 36 self-collected messages, labeled by researcher pairs and **never sent to an AI**.
- **Preprocessing (`ml/textnorm.py`):** keep casing; convert literal `\n`; mask emails **before** URL extraction; open URL regex (scheme, `www.`, or bare domain with an all-lowercase alphabetic TLD; a closed TLD list missed about 800 scam domains); replace URLs with `[LINK]` (keeping trailing punctuation); mask `[PHONE_NUM]`, `[OTP]`, `[AMOUNT]`, `[REF]`. Names cannot be regex-masked.
- **Stable IDs:** `msg_id` is a SHA-1 of the normalized text. The split is 70/15/15, stratified on the audited binary label and grouped by normalized template (no near-duplicates across splits), seed 42.
- **Raw counts:** 9,952 raw rows; 3,040 after exclusions; 2,568 unique (2,532 public + 36 self). 19 unclassifiable were excluded, leaving **2,549**.
- **Final classes:** malicious **1,576 (61.8%)**, made up of smishing 566 and spam 1,010; legitimate 973. Legitimate is the minority class (relevant for SMOTE).

| Split | n | legitimate | smishing | spam | with URL | malicious w/o URL | legitimate with URL |
|---|---|---|---|---|---|---|---|
| train | 1,786 | 683 | 413 | 690 | 1,124 | 61 | 82 |
| val | 381 | 145 | 62 | 174 | 237 | 20 | 21 |
| test | 382 | 145 | 91 | 146 | 239 | **15** | **17** |

- **Confounds (final labels):** 93.9% of malicious vs. 12.3% of legitimate messages contain a URL; median length 125 vs. 198 characters. This is why there are length-only, URL-presence-only and TF-IDF baselines, plus an optional `[LINK]`-removed ablation.
- **Small subsets:** the test set has only 15 malicious messages without a URL and 17 legitimate messages with a URL. Report them descriptively with exact binomial intervals. Smishing counts differ between val (62) and test (91).
- **Label provenance (n = 2,549):**
  - AI label only: 1,381 (54.2%)
  - source label only: 785 (30.8%)
  - human-decided: 347 (13.6%), made up of human 116, adjudicated 111, human_r2 60, human_majority 22, human_majority_r2 29, group 9
  - self_pair: 36 (1.4%)

  Never describe the corpus as "human-validated".
- **Merge outcome:** of 1,597 source-spam messages, 33.8% became smishing, 62.4% spam, 2.9% legitimate and 0.9% unclassifiable. Of the 150 audited legitimate messages, 6 (4%) became malicious; extrapolated, roughly 15 to 65 of the 785 unaudited legitimate messages may be mislabeled. Report this as a limitation.
- **Self-collected messages:** 36 (27 train, 5 val, 4 test). A possible style shortcut, but small.
- **Language:** the function-word heuristic in `corpus_stats.py` is **invalid** (it undercounts Taglish; about a fifth of a hand-checked sample was wrong). Do not report its 5.4% Taglish figure. Hand language labels for the test split are in progress (`make_language_sheets.py`, `language_results.py`, `docs/Language_Guide.md`).

## 4. Label audit history (Chapter 4 must report both rounds)
- **AI pre-annotation:** 1,747 messages (all 1,597 candidates plus a random 150 legitimate), Claude Sonnet 5.5 at medium effort, 18 batches in fresh chat sessions, prompt `ml/ai_labeling_prompt.md`. The prompt had the C4 rule only; Rules 1 to 3 came later, so part of the AI-human disagreement reflects that rule mismatch. The `criteria` field is inconsistent and not used for statistics. AI totals: spam 989, smishing 531, legitimate 203, unclassifiable 24.
- **Round 1:** 180 individual items (60 per AI class) plus 30 shared, six annotators, blind.

  | Measure | Value |
  |---|---|
  | Three-class κ | 0.449 |
  | Binary κ | **0.670** (bootstrap CI 0.547 to 0.787), below the declared 0.70 |
  | Fleiss' κ | 0.291 |

  Diagnostics: annotators E and F were outliers (mean pairwise κ 0.20 and 0.09; each agreed with the AI on 12 of 30). The disagreement clustered on gambling promos (C2 ambiguity) and institution promos (legitimate vs. spam).
- **Remedy:** Rules 1 to 3 written, Labeling Guide v2, recalibration session. **Round 2** used fresh items not in round 1 or adjudication: 30 shared for all six, 30 individual each for E and F, and pair EF's adjudication redone by AB and CD. Targets were fixed beforehand: Fleiss ≥ 0.60 and pooled binary ≥ 0.70.

  | Measure | Value |
  |---|---|
  | Fleiss' κ | **0.842** (CI 0.733 to 0.933) |
  | E/F binary κ | 0.962 (n = 60) |
  | Pooled binary κ (post hoc) | **0.897** (CI 0.824 to 0.961, n = 176) |
  | Pooled three-class κ | 0.687 (exploratory) |
- **Caveats to state:**
  - The round-2 pool excluded adjudication items, which are the hardest messages, so round 2 was measured on an easier population.
  - The pooled estimate mixes rounds.
  - Round-1 numbers stay on record.

## 5a. Increment 1 validation results (Oct 23; full run, 3 seeds; TEST UNTOUCHED)
Validation: n = 381 (236 malicious, 145 legitimate; 62 smishing; 20 malicious without a URL). Validation figures are optimistic because the epoch was chosen on validation.

| Model | F1 (mean ± sd) | Errors (FP + FN) | Smishing recall | No-URL recall |
|---|---|---|---|---|
| Majority | 0.765 | 145 | 1.00 | 1.00 |
| Length-only LR | 0.814 | 88 | 0.73 | 0.25 |
| URL-presence-only LR | 0.913 | 41 | 0.87 | 0.00 |
| TF-IDF + LR | 0.964 (CI 0.945 to 0.979) | 17 | 57/62 (0.92) | 11/20 (0.55; exact CI 0.32 to 0.77) |
| RoBERTa class-weighted | 0.977 ± 0.002 | 10 to 12 | 60 to 61/62 | 16/20 in every seed (0.80; exact CI 0.56 to 0.94) |
| RoBERTa unweighted | 0.978 ± 0.003 | 9 to 12 | 0.98 | 0.82 |
| RoBERTa unweighted + SMOTE head | 0.973 ± 0.007 | 10 to 16 | 0.97 | 0.83 |

- **Readings:**
  - RoBERTa beats TF-IDF by about 0.013 F1 (about 6 fewer errors out of 381); the confidence intervals overlap.
  - Its gains concentrate in smishing and no-URL recall, consistently across seeds, but on small subsets.
  - Class weighting makes no difference (the imbalance is mild).
  - **SMOTE adds nothing** (slightly worse, higher variance).
- **Pending decision:** final text model = class-weighted RoBERTa (lowest variance; `roberta_weighted_seed42` saved). Chapters must then say SMOTE was *evaluated and not adopted*; Section 2.2 currently says the study "adopts" it.
- **Still needed:**
  - Per-message validation predictions (for error analysis, the fusion weight, and a paired TF-IDF vs. RoBERTa comparison).
  - Results broken down by `label_method` (human-decided vs. AI vs. source labels) on validation and test.

## 5. Status of increments
- **Increment 1:** corpus, audit and freeze done. The notebook `increment1_baselines_roberta.ipynb` was delivered (baseline cells tested on a simulated corpus; **RoBERTa cells untested**: no GPU or network where written). Runs: class-weighted RoBERTa, and unweighted RoBERTa with a SMOTE head on training embeddings, over 3 seeds, with the test split dropped in Cell 1. Validation numbers are optimistic (epoch selected on validation). Expect TF-IDF and URL-only baselines to be strong; report honestly.
- **Increment 2 (URL features + fusion):** provisional prototype done Oct 25 (`ml/url_model.py`, `ml/train_url_fusion.py`). Full SOP 2 ground truth still to do: URL ground truth = Google Safe Browsing + at least one threat feed + WHOIS domain age + manual check. A URL not flagged by any source is "unknown", not benign; record query dates.
- **URL prototype decisions (Oct 25, Eduard):**
  - **Labels:** each link takes its message's `label_bin` (train split). This is *provisional*; call it "a preliminary URL model trained on message-level labels", never the SOP 2 result.
  - **Model:** logistic regression, 20 lexical features plus a public-suffix indicator. tldextract uses its bundled snapshot. Ch3 names no classifier, so this is the group's choice.
  - **No link expansion:** shortened links are not expanded (deviation from Ch3 §3.2 API Engine: the server must not visit scam links). They are flagged instead.
  - **Tuning rule (fixed before results, commit e99ab93):** w and the threshold were tuned on validation. The rule picked **w = 1.00** (the link score adds nothing on validation) and threshold 0.08.
  - **Threshold kept at 0.50 as a declared deviation.** Its gain is 1 error in 381 (F1 0.9792 vs 0.9767), within noise, and an 8% cut-off raises false alarms. Ch3 says the threshold is tuned on validation, so the paper needs a sentence on this.
  - **Weak point:** validation has only 21 legitimate messages with a link, and training has only 57 legitimate distinct links. The text model false-alarms on official bank and utility notices (for example a real telco overdue-bill SMS), while the URL model rates official domains as safe. The hard-negative set is the test for this.
- **Increment 3 (FastAPI + SHAP):** not started. Cap the SHAP evaluation budget and time classification and SHAP separately. Zero retention: no request-body logging, and exception handlers must never echo message text.
- **Latency target and the two-call flow (decided Oct 25):** the app calls `POST /v1/analyze?explain=false` first and shows the verdict, then calls again with the same text for the SHAP highlights (the server keeps no text between calls). The SOP 5 target (under 5 s) applies to **time to verdict**, which matches Chapter 3, Section 3.3.3 ("from SMS receipt to verdict rendering"). Time to highlights is reported separately and has no 5 s target. Laptop benchmark Oct 23 (`max_evals=256`): classification about 30 ms, SHAP median 1.9 s, p95 4.6 s, max 4.8 s. **Chapter 3 needs a sentence on this**: Section 3.2 (Figure 2 sequence and Section 3.2.4) still describes one response carrying both the verdict and the attributions, and should say the attributions arrive in a second request.
- **Increment 4 (Android):** not started. It should run against a stubbed API (Figure 3 JSON) in parallel. The device-side masking needs an offset map back to the original text. Kotlin masking must pass the same cases as `tests/test_textnorm.py`.

## 6. Open items before printing (Nov 1)
- **Hard-negative set:** at least 100 researcher-written legitimate messages (bank and e-wallet alerts, OTPs, legitimate messages with official links). Evaluation only, never in training. **Not written yet.**
- **Chapter edits pending** (apply in Google Docs):
  - Remove highlights for the adopted rules.
  - Chapter 1 spam definition (remove "loan solicitations"; gambling is always spam) and HAM definition fix.
  - Section 2.3 decision-rules paragraph and legitimate definition.
  - Section 3.6.4: round-2 paragraph, AI rule-mismatch disclosure, easier-pool caveat.
  - Update confound figures to 93.9% / 12.3% / 125 / 198.
  - Combined corpus 2,549; malicious share 61.8%; split sizes 1,786 / 381 / 382.
  - Evaluation-plan subset counts (15 and 17) and SOP 3 expectations.
- **Placeholders still needing data:** `[128/256]` max length and `[X]%` coverage (notebook Cell 2); Taglish share (hand language labels); hard-negative size; `[N]` latency requests; questionnaire `[source]`; `[2]` vCPU; the Herrera et al. URL.

## 7. Lessons from bugs already fixed (don't reintroduce)
- Chua has 2 malformed CSV rows (6 fields): read with `csv.reader`, not default pandas.
- A closed TLD list misses most scam domains. Emails must be masked before URL extraction. A `[LINK]` replacement must keep the trailing sentence punctuation.
- `msg_id` must be content-hashed, not row-numbered, so IDs survive added rows.
- Self-collected messages carry pair labels in `data/raw/self_collected.csv` (`text`, `label`).
