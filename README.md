# toxicity-detection
Comparative NLP study: TF-IDF + Logistic Regression vs. fine-tuned DistilBERT for toxic comment classification on the Jigsaw dataset. Includes GridSearchCV tuning, threshold calibration, multi-label extension, and full evaluation pipeline.
| Approach | Description |
|---|---|
| **TF-IDF + Logistic Regression** | Classical sparse bag-of-words baseline with GridSearchCV hyperparameter tuning |
| **Fine-tuned DistilBERT** | Contextual transformer model (66M params) fine-tuned for binary/multi-label toxicity classification |
## Research Question

> *"To what extent can automated ML models reliably detect toxic language in online comments, and how does fine-tuned DistilBERT compare to a TF-IDF + Logistic Regression baseline across Accuracy, F1 Macro, and ROC-AUC?"*
> ## Dataset

**Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Jigsaw/Google, 2017)

## Results

### Binary Classification

| Model | Accuracy | F1 Macro | ROC-AUC |
|---|---|---|---|
| TF-IDF + LR (tuned, `class_weight='balanced'`) | **0.9475** | 0.8477 | 0.9514 |
| **DistilBERT (fine-tuned, t = 0.35)** | 0.9467 | **0.8633 ✅** | **0.9757 ✅** |

> DistilBERT wins on both primary metrics: **+0.016 F1 Macro** and **+0.024 ROC-AUC**.

**Key Observations:**
- The TF-IDF baseline's accuracy is marginally higher (0.9475 vs 0.9467) due to the lowered DistilBERT threshold generating additional false positives — this is an expected and desirable trade-off for a recall-biased operating point.
- The baseline's F1 Macro of 0.8477 is remarkably strong, corroborating Wang & Manning (2012) — linear classifiers with large n-gram vocabularies are near-optimal for short-text classification.
- DistilBERT's advantage (particularly in ROC-AUC) is best understood as the value of contextual understanding for the hardest cases: sarcasm, quoted toxic speech, and formal hostility without profanity.

---

### Multi-Label Extension

TF-IDF one-vs-rest (OvR) strategy applied to all six sub-labels:

| Sub-label | F1 Score |
|---|---|
| obscene | **0.763** |
| toxic | 0.724 |
| insult | 0.649 |
| threat | 0.471 |
| severe_toxic | 0.451 |
| identity_hate | **0.371** |

**Overall Multi-Label F1 Macro: 0.5716**

The gradient from `obscene` → `identity_hate` reflects:
1. **Vocabulary distinctiveness** — obscenity maps to consistent explicit vocabulary; identity hate uses coded language and euphemisms
2. **Class frequency** — rarer sub-labels have fewer positive training examples (as few as ~160 for ~1% prevalence)
3. **Linguistic complexity** — identity hate employs dog-whistles and irony that defeat bag-of-words approaches

---

## Future Directions

Five research extensions, each grounded in a specific identified limitation:

| # | Direction | Addresses |
|---|---|---|
| 1 | **Bias Auditing by Demographic Group** | Annotator bias against AAVE & identity-mentioning comments (Dixon et al., 2018; Sap et al., 2019) |
| 2 | **Unicode Normalisation (NFKC)** | Adversarial evasion via character substitution — the most urgent technical gap |
| 3 | **DistilBERT for Multi-Label Classification** | Poor OvR performance on rare, complex sub-labels (identity\_hate F1=0.371, threat F1=0.471) |
| 4 | **Cross-Platform Generalisation Assessment** | Domain shift from Wikipedia talk pages to Reddit, Twitter, gaming chat |
| 5 | **SHAP Explainability** | Operational & legal explainability needs (EU AI Act, DSA contestability requirements) |
