# Kazakh profanity detector for product reviews

Binary text classifier: **one Kazakh comment or review in → `0` (no Kazakh profanity) or `1` (Kazakh profanity present), plus a probability.**

Fine-tuned `amandyk/KazakhBERTmulti` (Kazakh BERT) with a 2-class head, trained on a dataset built in this repository
from public Kazakh/Russian sources, with every label produced by the annotation protocol documented below.

> ⚠️ This repository necessarily contains profanity (lexicon, data, examples). It exists for content-moderation research.

---

## 1. Task definition

| | |
|---|---|
| Input | one Kazakh product review / comment / sentence |
| Output | `label ∈ {0, 1}` and `P(profanity) ∈ [0, 1]` (decision threshold calibrated on validation) |
| `1` means | the comment contains **Kazakh** profanity (obscene vocabulary or a profanity-grade insult) |
| `0` means | no Kazakh profanity — including **angry but clean** reviews, **mild insults**, and **Russian/English profanity** (see the mixed-language rule) |

### Mixed-language rule (and why Russian/English profanity is `0`)

Public discussion in Kazakhstan is routinely mixed Kazakh–Russian, so the rule must be explicit:

* profanity **of Kazakh origin** in a Kazakh-language comment → `1`
* Russian мат (`пиздец`, `нахуй`, `сука` …) or English profanity in a Kazakh or Russian comment → **`0`**
  (these rows are kept in the data as **hard negatives**, so the model learns the boundary instead of guessing)
* a pure Russian or English comment is never `1`

This is the *literal* task ("Kazakh profanity"), and it is what the model is optimised for. A unified
"profanity in any language" label would require a multilingual model and is listed under *Limitations / future work*.

---

## 2. Repository layout

```
config.yaml                  seed, splits, hyper-parameters (single source of truth)
requirements.txt             pinned dependencies (CPU torch)
data/
  lexicon.json               annotation lexicon (kk/ru/en entries, exceptions, variants)
  label_fixes.csv            14 human-approved label corrections, applied by 03_prepare_data.py
  raw/                       downloaded sources + MANIFEST.json (urls + sha256)
  raw/_derived/              cached Telegram mining results (deterministic)
  processed/                 train.csv / val.csv / test.csv, split_report.json, audit files
src/
  01_download_data.py        download all public sources
  02_build_lexicon.py        build the annotation lexicon
  03_prepare_data.py         PII strip, label, synthetic generation, dedup, split
  04_train.py                fine-tuning (grid stage + final stage)
  05_evaluate.py             one-shot test evaluation, probes, trace
  06_label_audit.py          flags probable label errors for human review
  07_build_external.py       builds the external (never-trained-on) evaluation set
  08_external_eval.py        evaluates the model on the external datasets
  lexicon_utils.py           normalisation + profanity matching (labeling only)
  text_utils.py              language detection, PII stripping, model-side normalisation
  synthetic.py               synthetic review generator
  model.py                   ProfanityClassifier (load + predict + explain)
  predict.py                 CLI inference
artifacts/
  model/                     trained model + tokenizer + model_meta.json
  grid_results.json          validation search results
  training_summary.json      final training settings + threshold sweep
  metrics.json               test metrics + probes
  confusion_matrix.txt       readable confusion matrix
  sample_predictions.csv     every test row with P(profanity)
  errors.csv                 test errors sorted by decision margin (header only for this checkpoint)
  trace_example.txt          end-to-end tokenisation trace
  external_metrics.json      external evaluation summary (FPR on never-trained-on data)
  external_predictions.csv    every external row with P(profanity)
```

**Model access.** The trained model is published as a GitHub Release asset:
**<https://github.com/Dana-Kaziyeva/Kazakh-Profanity-Detection/releases/tag/model-v1.1.0>**
(zip: `kazakh-profanity-model-v1.1.0.zip`, 608 MB; contains `model.safetensors` 651 MB, tokenizer files,
`model_meta.json` with `threshold=0.3`, `max_len=96`).
This is the model **retrained after the manual label corrections** (§10.1b) — the numbers in §6 and §10.2 come from
this checkpoint. The earlier release `model-v1.0.0` is kept only for history and is superseded.
It is too large for the git tree (GitHub blocks files > 100 MB), so `artifacts/model/` is git-ignored.
To use the model **without retraining**:

```bash
# download and unpack the model into artifacts/model/
curl -L -o model.zip https://github.com/Dana-Kaziyeva/Kazakh-Profanity-Detection/releases/download/model-v1.1.0/kazakh-profanity-model-v1.1.0.zip
tar -xf model.zip -C artifacts/model      # or any unzip tool, unpack into artifacts/model/
python -m src.predict "Бұл телефон сік, ақшамды қайтарыңдар"
```

Everything needed for inference is inside that folder — no retraining required. The model is regenerated
deterministically by `python src/04_train.py --stage final` (seed 42).

**All artefacts are regenerated deterministically** by the five numbered scripts in order, with `seed: 42`.

---

## 3. Dataset

### 3.1 Sources (all public, no credentials; download script: `src/01_download_data.py`)

| Source (Hugging Face / web) | Used for | Rows downloaded | Licence |
|---|---|---|---|
| [`kkenbbi/kazakh_reviews_2gis`](https://huggingface.co/datasets/kkenbbi/kazakh_reviews_2gis) | real Kazakh product/service reviews (restaurants, pharmacies, schools) | 136 | no explicit licence on the card (doi:10.57967/hf/5587) — academic use |
| [`R3iwan/entertainment-reviews-kazakh`](https://huggingface.co/datasets/R3iwan/entertainment-reviews-kazakh) | Kazakh review-style text (movies, books …) | 1,250 → **50 unique** (the file repeats each review 25× with an index prefix) | Apache-2.0 |
| [`Rustem-Kaimolla/kazakh-swear-words`](https://huggingface.co/datasets/Rustem-Kaimolla/kazakh-swear-words) | Kazakh profanity lexicon seeds + author-written phrases | 119 entries | MIT |
| [`R3iwan/kazakh_names`](https://huggingface.co/datasets/R3iwan/kazakh_names) | personal-name list for **PII stripping** (names are removed, not used as features) | 1,000 | Apache-2.0 |
| [`kurumikz/telegram-corpus-russian-kazakh`](https://huggingface.co/datasets/kurumikz/telegram-corpus-russian-kazakh) | **real Kazakh profanity** mined from chat (the public reviews contain almost none) + clean Kazakh messages as negatives | 1,493,124 messages → 6,538 Kazakh-script | CC BY-NC-SA 4.0 |
| [`Mnwa/russian-toxic`](https://huggingface.co/datasets/Mnwa/russian-toxic) | Russian **hard negatives**: rows verified to contain Russian мат → labelled `0` | 347,527 (30,000 scanned) | MIT |
| [`ai-forever/ru-reviews-classification`](https://huggingface.co/datasets/ai-forever/ru-reviews-classification) | clean Russian product reviews (negatives) | 75,000 (1,000 sampled) | Apache-2.0 |
| [LDNOOBW profanity list](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) | English profanity lexicon (detection + hard negatives) | 403 → 274 single-word entries | public list |

**How this relates to Kazakh product reviews.** The 2GIS/entertainment sources *are* Kazakh product-review text;
they are the domain of the task. However, public review platforms are moderated: **none of the 186 unique real
Kazakh reviews** collected here (136 2GIS + 50 unique entertainment reviews) contained Kazakh profanity — the one
review-like positive in the first data build came from a Kazakh text inside the Russian review corpus.
To train a real classifier we therefore also mine real Kazakh profanity from a
large public Kazakh/Russian chat corpus (the same linguistic community, but a chat rather than review register) and
balance the review domain with **explicitly flagged synthetic** review sentences. This trade-off is quantified in
§6 (per-subset metrics) and discussed in §8 (limitations).

### 3.2 Annotation protocol (labels created by us, originals discarded)

No source label is reused:

* 2GIS and entertainment datasets carry **sentiment** labels (`0/1`, `positive/negative/neutral`) → **discarded**.
  Sentiment ≠ profanity: *"Донер дәмсіз. Дым ұнамады."* (very negative) is labelled `0`.
* `kazakh-swear-words` is used as **lexicon material**, every entry manually reviewed. Its *labels* are revised:
  items of Russian origin that it marks as "Kazakh profanity" (`Кет нахуй`, `Маған похуй`, `Долбоебсын ба`) are
  moved to the Russian list and labelled `0` in this dataset.
* Russian toxic corpus labels (`toxic/non-toxic`) do not match this task → ignored; rows are re-scanned with the
  Russian lexicon and every hit is **relabelled `0`** (hard negative), because Russian мат is not Kazakh profanity.

The full protocol lives in `data/lexicon.json` (entries, variants, exceptions, severity, source per entry) and
`src/02_build_lexicon.py`.

### 3.3 Labeling rules

1. **Profanity, not negativity.** Negative reviews, complaints, insults without obscenity (ақымақ "fool",
   сасық "stinky", арсыз "shameless") → `0`. Only Kazakh obscene/profanity-grade vocabulary → `1`.
2. **Mixed language** — see §1: the profanity must be Kazakh; RU/EN profanity → `0` (hard negative).
3. **Masked words.** `с**к`, `с*к`, `С*К` are treated as profanity when ≥2 letters are visible, the first letter
   matches and the visible letters are an in-order subsequence of a lexicon root. `с***` (one visible letter) is
   **ambiguous** → not matched (documented rule, and the model is trained to handle it from data).
4. **Spelling variation.** Letter repetition is collapsed (`сіік` → `сік`), Latin homoglyphs are mapped to Cyrillic
   (`cік` → `сік`), inflected forms match through the root (`сік` → `сігейін`, `қотақ` → `қотағы`).
5. **Innocent look-alikes are excluded** through per-entry exception lists: `көтерді` ("raised") vs `көт`,
   `бокс` ("boxing") vs `бок`, `сукно` vs `сука`, `хлеб`/`себе` vs `еб`, `аман`/`амал` vs `ам`.
6. **Ambiguous cases** are excluded from automatic labelling rather than guessed (e.g. `пұшық`, `итпұшық`;
   single visible letter masks). The lexicon keeps them documented in `non_profane_insults`.
7. **Spam/links/names** are removed before labelling; rows without letters are dropped.

**Labelled examples** (also reproduced by `data/processed/audit_*.csv`):

| text | label | why |
|---|---|---|
| `Өнім өте жақсы, рахмет!` | 0 | clean praise |
| `Донер дәмсіз. Дым ұнамады.` | 0 | negative review, no profanity |
| `ақымақ сатушы, бірақ тауар жақсы` | 0 | insult without obscenity |
| `бағаны көтерді` | 0 | innocent word containing `көт` (exception list) |
| `Бұл телефон сік, ақшамды қайтарыңдар` | 1 | Kazakh profanity |
| `Қотақбас жігіт болсаң жауап берсей` | 1 | real Kazakh comment |
| `с**к, қайта келмеймін` | 1 | masked profanity |
| `сіік, тамақ нашар` | 1 | stretched spelling |
| `cік, тамак дамсыз` | 1 | Latin homoglyph `c` |
| `пиздец, доставка опять задержалась` | 0 | Russian profanity → hard negative |
| `the phone is shit` | 0 | English profanity → hard negative |

### 3.4 Synthetic data (all flagged `synthetic=True`)

`src/synthetic.py` generates review-style sentences from templates + product lists:

| template id | label | what it teaches | rows (all splits) |
|---|---|---|---|
| `kk_pos_profane` | 1 | Kazakh profanity in review frames (base / masked / stretched / homoglyph) | 1,233 |
| `kk_pos_profane_mixed_ru_words` | 1 | Kazakh frame with Russian loanwords + Kazakh profanity | 251 |
| `kk_negative_clean` | 0 | clean positive/neutral/negative reviews | 478 |
| `kk_negative_insult_without_profanity` | 0 | mild insults that are not profanity | 183 |
| `hard_neg_ru_profanity_in_kk_frame` | 0 | Russian мат inside a Kazakh frame | 414 |
| `hard_neg_en_profanity` / `_in_kk_frame` | 0 | English profanity (English and Kazakh frames) | 145 + 142 |
| **total** | | | **2,846** (49.3 % of the dataset) |

Every synthetic row is validated against the lexicon (positives must match, negatives must not); 132 inconsistent
rows were dropped. Synthetic rows are never mixed into a split together with their near-duplicates.

### 3.5 PII

Before splitting, all texts are passed through `text_utils.strip_pii`:
URLs, e-mails, `@usernames`, phone numbers and long digit strings are removed, and personal names from the
`R3iwan/kazakh_names` list (923 names of length ≥ 4) are replaced with `[ИМЯ]`.
Result: **538 substitutions, 0 rows with detectable PII left**. No reviewer names, contacts or identifiers are stored.

### 3.6 Deduplication and splits

* exact duplicates: **1,572 removed** (mostly the 25× repetition in the entertainment dataset)
* near-duplicate groups: normalised **token-set signature** (order- and punctuation-insensitive); whole groups are
  assigned to one split → verified **0 exact text overlap** between train/val/test
* stratified by label, group-aware, greedy assignment → **70 / 15 / 15**, seed **42** (`config.yaml`)

| split | rows | positives | profanity rate | real positives | synthetic share |
|---|---|---|---|---|---|
| train | 4,043 | 1,189 | 29.4 % | 153 | 49.3 % |
| val | 867 | 255 | 29.4 % | 27 | 49.3 % |
| test | 867 | 255 | 29.4 % | 29 | 48.9 % |
| **total** | **5,777** | **1,699** | **29.4 %** | **209** | **49.3 %** |

Language distribution: `kk` 3,258 · `ru` 2,345 · `en` 165 · `mixed` 7 · `other` 2.
Class distribution by source: synthetic 2,846 · Russian toxic hard negatives 1,197 · Russian reviews 1,000 ·
Telegram clean 331 · Telegram profanity 192 · 2GIS 136 · entertainment 50 · swear-word phrases 25.

---

## 4. Model

**Base:** [`amandyk/KazakhBERTmulti`](https://huggingface.co/amandyk/KazakhBERTmulti) — a BERT-base encoder
pretrained on Kazakh (fill-mask checkpoint; the encoder is reused, a fresh 2-class head is trained). 163M
parameters (≈100K-token Kazakh vocabulary).

**Why:** the label is *Kazakh* profanity — a Kazakh-specific tokenizer/weights represent Kazakh morphology
(`сік` → `сі|##к`, `қотағы` → `қота|ғы`) far better than a multilingual model, and the specialist's weak Russian/English
representation aligns with the label rule (RU/EN profanity → 0). Fine-tuning a pretrained encoder is explicitly
allowed by the task ("train a small model **or** fine-tune a pretrained one"); no model was trained from scratch.

**Text → model input**

1. `text_utils.model_normalize`: Unicode NFKC + whitespace collapse (identical at train and inference time;
   no fitted preprocessing — the tokenizer is frozen, so nothing is learned from the test data).
2. WordPiece tokenizer (Kazakh vocabulary) → subword ids; `[CLS] … [SEP]`.
3. Truncation at **96 tokens** (longest reviews are cut; the share of truncated test rows is reported in §6).
   Pad tokens (dynamic per-batch padding) are masked out.
4. Punctuation and masking characters (`*`, `#`, `@` …) are kept as tokens — masked forms are part of the signal,
   not something to normalise away.
5. Unknown words are split into subwords instead of `[UNK]`; the `[UNK]` rate per language is measured in §6.

**Training objective:** cross-entropy over 2 classes (`BertForSequenceClassification`, `num_labels=2`).

**Settings** (`config.yaml`): AdamW, lr selected on validation (see §5), weight decay 0.01, warmup 6 %,
batch 16, max 3 epochs, gradient clipping 1.0, early stopping on validation F1 (patience 1), CPU, seed **42**.

**Saved artefacts** (`artifacts/model/`): model weights, tokenizer files, `model_meta.json`
(base, max_len, threshold, label mapping, seed). Inference loads only this directory — no retraining needed.

---

## 5. Training procedure

```bash
# 0) environment
python -m venv venv && venv\Scripts\pip install -r requirements.txt

# 1) data
python src/01_download_data.py        # public sources -> data/raw (urls + sha256 in MANIFEST.json)
python src/02_build_lexicon.py        # -> data/lexicon.json
python src/03_prepare_data.py         # -> data/processed/{train,val,test}.csv (+ audit files)

# 2) validation-driven settings (small grid, CPU-friendly)
python src/04_train.py --stage grid   # -> artifacts/grid_results.json

# 3) final model on the full training split (+ threshold sweep on validation)
#    lr / max_len are taken from artifacts/grid_results.json (lr=5e-5, max_len=96);
#    the explicit equivalent is: python src/04_train.py --stage final --lr 5e-5 --max-len 96
python src/04_train.py --stage final  # -> artifacts/model/, artifacts/training_summary.json

# 4) single evaluation on the held-out test set
python src/05_evaluate.py             # -> artifacts/metrics.json, confusion_matrix.txt, ...
```

The grid stage trains on a 800-row training subsample (1 epoch, CPU budget: ~35 min for the whole grid) for each of
`lr ∈ {2e-5, 5e-5} × max_len ∈ {64, 96}` and picks the best **validation F1 of the profanity class**:

| lr | max_len | val precision | val recall | val F1 (profanity) |
|---|---|---|---|---|
| 2e-5 | 64 | 0.745 | 0.795 | 0.770 |
| 2e-5 | 96 | 0.758 | 0.776 | 0.767 |
| 5e-5 | 64 | 0.919 | 0.807 | 0.860 |
| **5e-5** | **96** | **0.928** | **0.807** | **0.863** |

→ `lr=5e-5`, `max_len=96`. The final model is then trained on a class-balanced 2,439-row training subsample
(2 epochs, early stopping on validation F1) and the decision threshold is swept on validation.
Everything is CPU-only; full training took **65 minutes** (2 epochs, 2,439 rows; see `artifacts/training_summary.json`).

The grid was run **before** the manual label corrections (§10.1b). It only selects hyperparameters (lr / max_len), and
the final model was trained again from scratch **after** the corrections — the grid therefore does not leak corrected
labels into the final run, and the selected settings are re-validated in §6.2.

---

## 6. Results

### 6.1 Test metrics (single evaluation on the held-out 867-row test set, threshold 0.30)

The model was retrained **after the manual label corrections** (§10.1b) and evaluated once:

| metric | value |
|---|---|
| test examples | **867** (255 profane, 612 clean) |
| precision (profanity class) | **1.0000** |
| recall (profanity class) | **1.0000** |
| F1 (profanity class) | **1.0000** |
| F1 macro | 1.0000 |
| accuracy | 1.0000 |

|  | predicted 0 | predicted 1 |
|---|---|---|
| **true 0 (clean)** | TN = 612 | FP = 0 |
| **true 1 (profane)** | FN = 0 | TP = 255 |

**How to read a perfect test score.** The labels come from the same annotation protocol the model can learn
(the lexicon + language rule), and the test split is class-balanced with ~49 % synthetic rows. The test set is
therefore an *upper bound*, not an estimate of production quality: **4 of the 11 errors of the round-1 model were
label noise** that the human review removed, the other 7 were genuine, and the retrained model fixed 4 of those 7
(§6.8). The honest generalisation checks are the boundary probes below
and the external evaluation on never-seen domains (§10.2: 2 false positives in 530 comments, FPR 0.38 %).

**What the errors would mean for moderation** (none occurred on this test set):

* **False positive** – a clean review is flagged; it may be hidden or held for review, annoying users and costing
  moderator time. Over-blocking legitimate criticism is a reputational risk.
* **False negative** – profanity stays visible; in review moderation **FN is usually the costlier error**, which is
  why the threshold (0.30) was kept at the most recall-friendly end of an equally-scoring sweep.

Caveat: at a realistic profanity prevalence (<1 %) the same model would have lower precision; the threshold must be
re-tuned to a precision target for deployment.

### 6.2 Validation search and threshold (selection data, not test)

Grid (§5) → `lr=5e-5`, `max_len=96`. Final run: 2,439 class-balanced training rows, 2 epochs (65 min CPU), best
epoch = 2, **validation F1 = 0.988** (P 0.988 / R 0.988).

| threshold | val precision | val recall | val F1 |
|---|---|---|---|
| **0.30** | **0.988** | **0.988** | **0.988** |
| 0.40 | 0.988 | 0.988 | 0.988 |
| 0.50 | 0.988 | 0.988 | 0.988 |
| 0.60 | 0.988 | 0.988 | 0.988 |
| 0.70 | 0.988 | 0.988 | 0.988 |

The sweep is flat, so the lowest threshold (0.30) was kept — the most recall-friendly choice among equally
scoring options.

### 6.3 Boundary probe (Option-B behaviour, test split)

| subset | n | metric | value |
|---|---|---|---|
| real Telegram positives (Kazakh profanity) | 29 | recall | **1.000** |
| synthetic positives (all templates) | 225 | recall | 1.000 |
| masked-form positives (`с**к`, `қ***қ`, …) | 31 | recall | **1.000** |
| mixed KK+RU comments **containing** Kazakh profanity | 47 | recall | 1.000 |
| RU hard negatives — real Russian toxic corpus | 179 | correct-0 rate | **1.000** |
| RU hard negatives — synthetic, Kazakh frame | 64 | correct-0 rate | 1.000 |
| EN hard negatives (synthetic) | 37 | correct-0 rate | 1.000 |
| real 2GIS/entertainment reviews labelled 0 | 32 | correct-0 rate | 1.000 |
| insult-without-profanity negatives | 28 | correct-0 rate | 1.000 |

The model keeps the language boundary on real Russian profanity (179/179) and catches every real Kazakh profanity
comment, including the masked and mixed-language ones.

### 6.4 Per-language breakdown

| language | n | precision | recall | F1 |
|---|---|---|---|---|
| kk | 489 | 1.000 | 1.000 | 1.000 |
| ru | 356 | 1.000 (8 TP / 348 TN) | 1.000 | 1.000 |
| en | 22 | – (no positives) | – | – |

### 6.5 Error analysis

**The retrained model makes no errors on the current test split.** The 11 errors of the round-1 model are still the
best material for understanding what is hard here — but only **4 of them were label noise**; the remaining 7 were
genuine model errors, and 4 of those are now fixed (§6.8):

1. **False negative caused by a label error** — `Качество ткани … сшили по бокам …`: the Russian phrase *по бокам*
   matched the Kazakh root `бок`. The model said `0`, the annotation said `1`; the human review fixed the label
   to `0`, so this was not a model mistake at all.
2. **Three false positives that were label errors** — `Шешенды амын сыгем`, `Шешенды сыгем`, `Шешен амын сыгейн`
   were labelled `0` only because `detect_lang` saw no Kazakh-specific letters; the model scored them 0.9998.
   The reviewer confirmed they are Kazakh profanity, so the labels became `1` and the "errors" disappear.
3. **Proper nouns / nicknames outside the test set** — `Бокеевской Орде`, `Белобока`, `«Сігео»` matched short roots
   inside names. These were training/validation label errors; labels corrected to `0`.
4. **Russian profanity inside Kazakh frames** (`заалупа`, `муудак`, `иди на хуй пидор …`) — these stay `0` by the
   mixed-language rule. Round 1 got **4 / 66** of them wrong (it keyed on the profanity token instead of the
   language); after retraining: **0 / 64** errors on this probe. A residual case, `суши ш***а, сапасы нашар.`
   (masked Russian profanity in a Kazakh frame), is still flagged when re-scored — it simply is not part of the
   current split any more (§6.8).
5. **External false positives (new domains, §10.2)** — with the retrained model only **2 of 530**:
   a military-qualification paragraph (p=0.857) and `Сайлау қорытындысы таластырылуда.` (p=0.677, "the election
   results are being disputed"). Neither contains profanity and neither matches the lexicon, so these are pure
   domain-shift errors of the encoder. *Practical fix*: keep adding formal/news Kazakh text as negatives — the
   external corpus is exactly that resource.

### 6.6 Tokenizer probe ([UNK] share, 50 test texts per language)

| language | [UNK] tokens | total tokens | [UNK] rate |
|---|---|---|---|
| kk | 0 | 919 | 0.00 % |
| ru | 0 | 2,182 | 0.00 % |
| en | 0 | 472 | 0.00 % |

Long comments: 6.46 % of test texts (56 / 867) exceed 96 tokens and are truncated; the longest test comment has
1,874 tokens and the 95th percentile is 118 tokens.

### 6.7 Tracing one comment end-to-end (`artifacts/trace_example.txt`)

```
text      : жеткізу пришел, амың, размер не тот.
tokens    : [CLS] | жеткізу | при | ##шел | , | ам | ##ың | , | размер | не | тот | . | [SEP]
[UNK]     : none  (mixed Kazakh-Russian text, zero unknowns)
logits    : [-4.376, 4.176]
softmax   : [0.00019, 0.99981]  ->  P(profanity) = 0.9998
threshold : 0.30  ->  prediction = 1   (true label = 1)
```

The mixed Kazakh/Russian frame is tokenised without unknowns; the Kazakh profanity `амың` is split into
`ам | ##ың` and still recognised.

### 6.8 Round 1 (before the review) vs round 2 (after the 14 label fixes)

Same base model, same settings, same seed — only the labels changed (14 rows) and the deterministic re-split that
follows from them.

| | round 1 | round 2 (published) |
|---|---|---|
| training rows (class-balanced) | 2,435 | 2,439 |
| lr / max_len / epochs | 5e-5 / 96 / 2 | 5e-5 / 96 / 2 |
| CPU training time | 45 min | 65 min |
| validation F1 (threshold chosen on val) | 0.9902 @ 0.40 | 0.9882 @ 0.30 |
| **test P / R / F1** | **0.962 / 0.9961 / 0.9787** | **1.000 / 1.000 / 1.000** |
| test errors | 10 FP + 1 FN | 0 |
| **external FPR (same 530 rows)** | **4 / 530 = 0.75 %** | **2 / 530 = 0.38 %** |
| RU hard negatives, synthetic KK frame | 4 / 66 wrong | 0 / 64 |
| real Kazakh reviews labelled 0 | 2 / 23 wrong | 0 / 32 |
| masked RU profanity in KK frame (`заалупа`, `м***к`) | 3 of 3 FP | 0 of 3 FP (re-scored) |

**The two test scores are not measured on the same rows.** Changing 14 labels changes the stratified re-split, so the
round-1 and round-2 test samples are not the same rows: of the 11 round-1 errors only `r000041` is still in the test
split. The external 530-row set is the only row-identical comparison, and it is the number to trust for
generalisation.

**Re-scoring all 11 round-1 test errors with the published model** (`true` = human-reviewed label, §10.1b):

| id | round-1 label | round-1 pred | now | P(prof) | verdict |
|---|---|---|---|---|---|
| `r000471` | 1 | 0 | 0 | 0.0005 | label noise — reviewer changed the label to 0 |
| `r002918` | 0 | 1 | 1 | 0.9998 | label noise — reviewer changed the label to 1 |
| `r002919` | 0 | 1 | 1 | 0.9998 | label noise — reviewer changed the label to 1 |
| `r002920` | 0 | 1 | 1 | 0.9998 | label noise — reviewer changed the label to 1 |
| `r000041` | 0 | 1 | **0** | 0.0066 | genuine error, **fixed** |
| `r005165` | 0 | 1 | **0** | 0.0009 | genuine error, **fixed** (`заалупа` in KK frame) |
| `r005092` | 0 | 1 | **0** | 0.0186 | genuine error, **fixed** (`м***к` in KK frame) |
| `r005079` | 0 | 1 | **0** | 0.0005 | genuine error, **fixed** (`заалупа` in KK frame) |
| `r005224` | 0 | 1 | 1 | 0.9992 | genuine error, **still wrong** (masked RU `ш***а` in KK frame) |
| `r002735` | 0 | 1 | 1 | 0.9854 | genuine error, **still wrong** (clean proverb-like text) |
| `r000006` | 0 | 1 | 1 | 0.9982 | genuine error, **still wrong** (positive review) |

Net effect: **4 label-noise rows left the error list, 4 real errors were fixed, 3 real errors persist** — but those
three rows are no longer in the test split, so the 1.000 test score does not hide them; they are stated here
explicitly. Summary: the 0.9787 → 1.0000 jump is roughly **half label correction, half real improvement**; the
independent evidence of real improvement is the external FPR (0.75 % → 0.38 %), the language-boundary probe
(4/66 → 0/64) and the current behaviour on the README probe `пиздец, доставка опять задержалась`
(p=0.0009 → `0`), which the round-1 checkpoint answered as profanity.

## 7. Prediction on new comments (no retraining)

```bash
python -m src.predict "Өнім өте жақсы, рахмет!"
# text       : Өнім өте жақсы, рахмет!
# prediction : 0  (no Kazakh profanity)
# probability: 0.0002  (threshold 0.3)

python -m src.predict "Бұл телефон сік, ақшамды қайтарыңдар"
# prediction : 1  (Kazakh profanity present)   probability: 0.9998

python -m src.predict "с**к, қайта келмеймін"          # masked form
# prediction : 1    probability: 0.9996

python -m src.predict "kөтің, заказ дәріхана пришел с опозданием на неделю."   # Latin k homoglyph + Russian
# prediction : 1    probability: 0.9998

python -m src.predict "пиздец, доставка опять задержалась"   # Russian profanity -> 0 by the label rule
# prediction : 0    probability: 0.0009

python -m src.predict --debug "с**к, қайта келмеймін"   # tokens + lexicon hits
echo "Бұл телефон сік, ақшамды қайтарыңдар" | python -m src.predict
```

Or from Python:

```python
from src.model import ProfanityClassifier
clf = ProfanityClassifier("artifacts/model")
clf.predict("Бұл телефон сік, ақшамды қайтарыңдар")
# {'label': 1, 'probability': 0.9998, 'label_name': 'Kazakh profanity present', 'threshold': 0.3, ...}
clf.explain("Бұл телефон сік, ақшамды қайтарыңдар")   # token-by-token trace
```

Inference speed on this CPU: ~50 ms for one comment, ~150 ms/comment in batches of 32.

> Boundary behaviour after retraining: Russian profanity (`пиздец …`) is correctly `0`, and the only remaining
> false positives are two **clean** formal/news texts on never-seen domains (§10.2, FPR 0.38 %) — i.e. the residual
> error direction is over-triggering on formal prose, not missing profanity.

## 8. Limitations and future work

### What this model cannot do

* **Russian/English profanity is `0` by design.** `пиздец, доставка опять задержалась` is a hard negative.
  Catching it requires the unified multi-language label and a multilingual encoder (see below).
* **Unknown slang / neologisms** that are not in the lexicon were never labelled `1`, so the model cannot learn
  them. Annotation recall is bounded by the lexicon.
* **Single visible-letter masks** (`с***`) are ambiguous and were deliberately not matched; the model may or may
  not generalise to them.
* **Real review positives are scarce.** None of the 186 unique real Kazakh reviews contained Kazakh profanity; real
  positives come from Telegram chat (different register), and 88.2 % of test positives are synthetic. The
  per-subset metrics in §6 quantify this gap honestly instead of hiding it behind one aggregate F1.
* **Prevalence shift.** The test set is ~29 % profane; a production review stream is far below 1 %. At that
  prevalence the same model has much lower precision — threshold and calibration must be re-tuned per deployment.
* **Language-detection heuristic.** Ukrainian/Belarusian text can look Kazakh to the `і` check; a handful of clean
  negatives are affected (they stay label `0`, so training labels are unaffected).

### Practical improvements

1. **More real labelled reviews** (partner moderation data) or active learning on low-confidence predictions.
2. **Hybrid model**: feed lexicon hits / character n-gram features alongside the encoder (the current model must
   learn the lexicon from ~2.8 k examples).
3. **Unified multi-language label** (`KZ+RU+EN`) with `xlm-roberta-base` or `distilbert-base-multilingual-cased`;
   the label definition, not the architecture, is the main change.
4. **Calibration** (temperature scaling / isotonic) and a deployment-specific threshold chosen from a precision
   target rather than F1.
5. **Data augmentation with human-written negatives** that contain profanity look-alikes (already started with the
   exception lists) to harden against false positives.

### Ethical note

The dataset and lexicon exist solely to build moderation tooling. Profanity is never generated at inference time;
the model returns a numeric label and probability. No personal data is stored: names, phones, e-mails and
usernames are stripped before training (§3.5).

---

## 9. Reproducibility checklist

| task requirement | where |
|---|---|
| data sources, access, sizes, licences | §3.1 (`src/01_download_data.py`, `data/raw/MANIFEST.json` with urls + sha256) |
| relation to Kazakh product reviews | §3.1 |
| labelling method + rules (mixed language, masked, variants, ambiguous, negative≠profane) | §3.2–3.3 (`data/lexicon.json`, `src/02_build_lexicon.py`) |
| existing labels checked and revised | §3.2 (sentiment discarded; Russian-origin swear-words relabelled; RU toxic relabelled 0) + §10.1/§10.1b (human review, 14 corrections) |
| synthetic examples clearly identified | §3.4 (`synthetic`/`template_id` columns, 2,846 rows) |
| labelled examples | §3.3 table + `data/processed/audit_*.csv` |
| no personal data | §3.5 (538 PII substitutions, 0 remaining) |
| dedup, near-dups, splits, seed | §3.6 (`split_report.json`, seed 42) |
| preprocessing fitted on train only | §4 (frozen tokenizer; only deterministic normalisation — nothing fitted) |
| model choice + justification | §4 |
| long comments / punctuation / unknown tokens | §4 + §6.6 (truncation 6.46 %, `[UNK]` probe) |
| settings, seed, dependencies recorded | `config.yaml`, `requirements.txt`, `artifacts/training_summary.json` |
| validation-based selection + threshold | §5 (grid table) + §6.2 (threshold sweep) |
| trainable model saved / no-retraining inference | GitHub Release `model-v1.1.0` (zip → `artifacts/model/`), §2 + §7 |
| one evaluation on held-out test | §6.1 (867 examples, P/R/F1, confusion matrix) |
| FP/FN meaning for moderation | §6.1 |
| ≥3 mistakes analysed with fixes | §6.5 (11 round-1 errors: 4 label noise, 4 fixed, 3 persisting) + §6.8 (round-1 vs round-2) |
| trace one comment end-to-end | §6.7 (`artifacts/trace_example.txt`) |
| limitations | §8 |

---

## 10. Label audit and external evaluation (follow-up round)

### 10.1 Label audit

`src/06_label_audit.py` scores every row that has a label under suspicion with the trained model and with rule-based
checks, and writes two files **without changing any label**:

| file | content |
|---|---|
| `data/processed/label_review.csv` | all 2,115 rows flagged at audit time (snapshot **before** the corrections below), every reason, model probability, suggested label |
| `data/processed/label_review_actionable.csv` | the 94 rows that need a human decision, grouped by `review_type` |

Reasons used: `MODEL_SAYS_PROFANE` / `MODEL_SAYS_CLEAN` (strong model disagreement), `RULE_ZERO_DESPITE_KK`
(label 0 only because the language heuristic tagged the text `ru`), `KK_MATCH_IN_RU_TEXT`,
`AMBIGUOUS_SHORT_ROOT` (`ам`/`бок`/`шшс`), `WEAK_MASKED_MATCH`, `PROPER_NOUN`, and the informational `HARD_NEGATIVE`.

Manual corrections are applied through **`data/label_fixes.csv`** (`id,label,note`): `03_prepare_data.py` merges it
before splitting, prints how many labels changed (0→1 / 1→0), and the run is otherwise identical.

### 10.1b Manual label corrections (applied after human review)

The review round changed **14 labels** (`data/label_fixes.csv`, written as UTF-8 with BOM so Excel opens it):

| change | rows | examples |
|---|---|---|
| **0 → 1** | 10 | `Шешен амын сыгейн`, `Бок же`, `Котын сыгейн`, `Сыктыр бар`, … — Kazakh profanity written without Kazakh-specific letters, so the language heuristic had tagged it `ru` |
| **1 → 0** | 4 | `… сшили по бокам …`, `… в Бокеевской Орде …`, `… «Сігео» …`, `… Белобока — прозвище …` — innocent words / proper nouns matched by short roots (`бок`, `сіг`) |

Rows with **Russian** profanity (`иди на хуй пидор …`, `… ссыкун …`, `… заалупа …`) were explicitly kept at `0`:
they are hard negatives by the mixed-language rule, not label errors.

`03_prepare_data.py` applies the fixes before splitting, prints
`label fixes applied: 14 rows (0->1: 10, 1->0: 4)` and re-splits deterministically (seed 42).
All review/label CSVs are written as **UTF-8 with BOM** (`utf-8-sig`); without the BOM Windows Excel
shows the Kazakh/Russian text as ANSI mojibake even though the file itself is valid UTF-8.

### 10.2 External evaluation on never-trained-on datasets

`src/07_build_external.py` + `src/08_external_eval.py` evaluate the model on two datasets that are **not part of any
split**: 500 long scientific Kazakh texts (`Arailym-tleubayeva/small_kazakh_corpus`) and 180 curated opinion sentences
(`mangi-llm/Kazakh_Multi-Task_corpus`). No open non-gated Kazakh dataset with real profanity exists (the 100K KZ
movie-review corpus is gated), so this external set measures **false positives on unseen domains**, not recall.

Result with the **retrained** model: **2 / 530 flagged (FPR 0.38 %)**, both pure model errors (no lexicon match):
a formal military-qualification paragraph (p=0.857) and `Сайлау қорытындысы таластырылуда.` (p=0.677). The first
model flagged 4/530 (0.75 %), so the label corrections and retraining also reduced domain-shift false positives.
Recall on unseen real profanity is covered by the 29 real Telegram positives of the held-out test split (recall 1.00).

