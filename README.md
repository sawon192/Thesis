# Thesis
# Bengali Hate Speech Detection 🇧🇩

A Multi-Track Benchmark with Fusion, Explainability and Uncertainty Quantification

This repo contains the code and experiments for our undergraduate thesis at the **University of Global Village**, Department of CSE. We built a large benchmark for detecting cyberbullying/hate speech in Bengali social media comments, comparing 17 models across 5 different modeling tracks, and then went further with model fusion, explainability (SHAP/LIME), and uncertainty quantification.

> 📄 Full thesis PDF is in the repo (`/thesis`) if you want all the details, math, and figures.

## Authors

- Dipta Sarker (12221125)
- Md. Rahmatullah Mia (12221100)
- Mahdi Hasan (12221067)
- M. Shawan Biswas (12221096)

Supervised by **Md Zahid Akon**, Lecturer, Dept. of CSE, UGV.

## Why we did this

Bengali is spoken by 230M+ people but has way fewer NLP resources than English, and most prior Bengali hate-speech papers only test one model on one dataset with no confidence intervals, no explainability, and no idea how confident the model actually is. We wanted to fix that by:

1. Comparing classical ML, embeddings, deep learning, and transformers **under one identical evaluation protocol**
2. Checking if combining models (fusion/ensembling) actually helps, statistically
3. Opening the black box with SHAP, LIME, and attention comparisons
4. Measuring how confident (and how *calibrated*) the models really are
5. Making sure none of it was a fluke, by replicating everything on a second, independent dataset

## TL;DR Results

| Model | F1-macro | Accuracy |
|---|---|---|
| **Stacked Ensemble** 🏆 | **0.8951** | **0.9068** |
| BanglaELECTRA | 0.8927 | 0.9032 |
| TF-IDF + XGBoost | 0.8828 | 0.8959 |
| BanglaBERT | 0.8814 | 0.8916 |
| TextCNN | 0.8589 | 0.8769 |
| GloVe + LogReg | 0.6939 | 0.7140 |

- Best model: a **stacked ensemble** of 6 models, edging out BanglaELECTRA (barely, but statistically significant via McNemar test)
- Temperature scaling cut calibration error (ECE) from 0.0324 → 0.0248
- Re-ran the entire pipeline on a second dataset (BD-SHS, ~50k comments) — model rankings held up almost perfectly (Spearman ρ = 0.988), so the findings aren't a one-dataset fluke

## Dataset

We use the public Bengali hate-speech corpus from Romim et al. (2021) — ~30k YouTube/Facebook comments labeled bullying (1) / non-bullying (0), across 7 categories (sports, politics, religion, crime, entertainment, celebrity, memes).

- After cleaning: **29,829 comments** (33.4% bullying / 66.6% non-bullying)
- Split: 70% train / 15% val / 15% test, stratified

We also replicate the whole thing on **BD-SHS** (50,264 comments) to check generalization.

> ⚠️ Datasets are not redistributed in this repo due to licensing — see the `data/` folder for download instructions and links to the original sources.

## The 5 Modeling Tracks

```
Track A – Classical ML        LogReg, Linear SVM, Naive Bayes, XGBoost (on TF-IDF)
Track B – Static Embeddings   Word2Vec, FastText, GloVe + LogReg
Track C – Deep Learning       TextCNN, BiLSTM + Attention
Track D – Transformers        BanglaBERT, BanglaELECTRA (fine-tuned)
Track E – Fusion & Ensembles  Early fusion, late fusion, soft-vote, stacked ensemble
```

Every single model is scored the same way: 11 metrics (accuracy, F1-macro, ROC-AUC, MCC, ECE, etc.), bootstrapped 95% confidence intervals, and pairwise McNemar significance tests.

## What else is in here

- **Explainability**: SHAP + logistic regression coefficients (global), LIME (local, per-prediction), and a side-by-side comparison of transformer attention vs. BiLSTM attention
- **Uncertainty quantification**: MC-Dropout (aleatoric vs. epistemic uncertainty), deep-ensemble disagreement, risk-coverage/selective prediction, and temperature scaling
- **Ablations**: TF-IDF vs. bag-of-words, threshold tuning, category features, a "stronger transformer" config (turns out bigger isn't always better here — see thesis Section 4.7.5)
- **Data audit**: checked for duplicate leakage / label-parsing bugs before trusting any of the numbers above

## Repo structure

```
├── data/                 # dataset download instructions (not the raw data itself)
├── notebooks/            # Jupyter notebooks for each track (EDA, training, eval)
├── src/                  # reusable python modules (preprocessing, models, metrics)
├── results/              # saved metrics, confusion matrices, figures
├── thesis/               # full thesis PDF + figures used in the paper
└── README.md
```

*(Adjust this section to match your actual folder layout before pushing!)*

## Running it yourself

```bash
git clone https://github.com/<your-username>/bengali-hate-speech-detection.git
cd bengali-hate-speech-detection
pip install -r requirements.txt
```

Then open the notebooks in `notebooks/` in order — they're numbered by track (Track A first, fusion/ensembles last). Everything was run on a Kaggle T4 GPU; the whole pipeline takes about 3 hours end-to-end.

## Limitations (being honest)

- Only tested on standard Bengali — no code-mixed "Banglish" evaluation
- Binary labels only (bullying vs. not) — no fine-grained categories like misogyny, xenophobia, etc.
- The ensemble's improvement over a single transformer is small — for many real deployments, just using BanglaELECTRA alone might be the better trade-off given the extra complexity of running 6 models

See Chapter 5 of the thesis for the full limitations and future work discussion.

## Citation

If this is useful for your own work, feel free to cite the thesis:

```
Sarker, D., Mia, M. R., Hasan, M., & Biswas, M. S. (2026).
Bengali Hate Speech Detection: A Multi-Track Benchmark with Fusion,
Explainability and Uncertainty Quantification.
BSc Thesis, University of Global Village.
```

## Acknowledgements

Thanks to our supervisor Md Zahid Akon, the CSE department at UGV, and the authors of the Romim et al. and BD-SHS datasets that made this whole project possible.

---
*This was our undergraduate thesis project — feedback and issues are welcome!*
