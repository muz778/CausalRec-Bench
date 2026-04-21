# CausalRec-Bench

**A Semi-Synthetic Benchmark for Evaluating Causal Cold-Start Recommendation Under Exposure Bias and Concept Drift**

Ali Hassan — RecSys 2026 CONSEQUENCES Workshop  
Collaborating with Dr. Yan Zhang, Charles Darwin University

---

## Quick Start — 4 Commands

```bash
git clone https://github.com/muz778/CausalRec-Bench.git
cd CausalRec-Bench
pip install -r requirements.txt
python download_data.py
python VERIFY_RESULTS.py
```

Downloads all data and pre-trained models automatically, then runs real evaluation to verify all paper results.

---

## What Is CausalRec-Bench?

The first semi-synthetic benchmark for causal cold-start recommendation with ground-truth causal labels on every interaction. No existing public dataset provides this.

| Statistic | Value |
|-----------|-------|
| Users | 50,000 |
| Items | 4,000 (2 domains) |
| Interactions | 1,944,842 |
| Domains | E-commerce + Streaming |
| Confounders | 5 |
| Cold-start users | 9,896 (19.8%) |
| Non-genuine clicks | 57.6% |
| Evaluation splits | 18 |
| Validation checks | 12/13 passed |

---

## Key Findings

1. Causal MF beats Standard MF on cold-start by **+27.2%** (p<0.001)
2. Causal LightGCN beats Standard LightGCN on warm users by **+18.4%** (p<0.001)
3. Position bias is the strongest confounder — impact **-0.1184** on CP@10
4. Graph methods produce identical cold-start recommendations regardless of causal training — motivating LLM-driven causal profiling
5. Standard metrics (P@10, NDCG@10) systematically favour biased models

---

## Five Confounders

| Confounder | Effect | Ratio |
|------------|--------|-------|
| Promotion bias | +40% exposure, +15% click | 1.47x |
| Popularity bias | +30% exposure, +12% click | — |
| Position bias (novel) | +25% click at position 1, decays to +1% at position 10 | 1.87x |
| Seasonal concept drift | Winter +15% books, Summer +15% outdoor | — |
| New item penalty | -20% exposure | — |

---

## What download_data.py Downloads
Data files (18 splits):
users.csv, items.csv, interactions.csv
train.csv, val.csv, test.csv
cold_start.csv, level1/2/3 splits
seasonal splits, domain splits
position splits
Pre-trained models:
fmf_std_U.npy + fmf_std_V.npy   (Standard MF)
fmf_caus_U.npy + fmf_caus_V.npy (Causal MF)
lgcn_std.pt                      (Standard LightGCN)
lgcn_caus.pt                     (Causal LightGCN)
Total: ~1.7 GB

---

## What VERIFY_RESULTS.py Does

Loads our pre-trained model weights and runs real evaluation from scratch. Produces this verification report:
Scenario               Model                  Paper  Verified  Match

Cold-Start             Standard MF            0.3771    0.3771  PASS
Cold-Start             Causal MF              0.4798    0.4798  PASS
Cold-Start             Standard LightGCN      0.5927    0.5927  PASS
Level 3 - Hard         Standard LightGCN      0.5327    0.5327  PASS
Level 3 - Hard         Causal LightGCN        0.6305    0.6305  PASS
Finding 1 — Causal MF cold-start: +27.2% PASS
Finding 2 — Causal LightGCN Level 3: +18.4% PASS

---

## Main Results (Category Precision@10)

| Model | Level 3 Hard | Cold-Start | Streaming Cold |
|-------|-------------|------------|----------------|
| Popularity | 0.3772 | 0.3877 | 0.0438 |
| Standard MF | 0.3675 | 0.3771 | 0.1917 |
| **Causal MF** | 0.3593 | **0.4798 +27.2%** | 0.2885 |
| Standard LightGCN | 0.5327 | 0.5927 | 0.3525 |
| **Causal LightGCN** | **0.6305 +18.4%** | 0.5927 | 0.3525 |
| Causal Upper Bound | 0.5130 | 0.5927 | 0.3525 |

---

## Statistical Significance

| Comparison | Metric | Improvement | p-value |
|------------|--------|-------------|---------|
| Causal MF vs Standard MF | CP@10 Cold-Start | +27.2% | p<0.001 |
| Causal MF vs Standard MF | CP@20 Cold-Start | +25.7% | p<0.001 |
| Causal LightGCN vs Standard LightGCN | CP@10 Level 3 | +18.4% | p<0.001 |
| Causal LightGCN vs Standard LightGCN | CP@20 Level 3 | +19.7% | p<0.001 |

---

## Ablation Study

| Confounder Removed | Impact on CP@10 |
|-------------------|----------------|
| Position bias | -0.1184 (strongest) |
| Promotion bias | -0.1149 |
| Popularity bias | -0.0490 |
| Seasonal drift | -0.0453 |

---

## Project Structure
CausalRec-Bench/
├── README.md
├── requirements.txt
├── download_data.py        ← downloads everything
├── VERIFY_RESULTS.py       ← verifies paper results
├── data/                   ← downloaded by download_data.py
├── pretrained_models/      ← downloaded by download_data.py
├── models/
│   └── fast_mf.py
├── evaluation/
│   └── metrics.py
├── results/
└── figures/

---

## Google Drive

All files available at:
https://drive.google.com/drive/folders/193xUcjZHh03Hal0v1F-z7lhpKQ0yCKNh

---

## Citation

```bibtex
@inproceedings{hassan2026causalrecbench,
  title     = {CausalRec-Bench: A Semi-Synthetic Benchmark for Evaluating
               Causal Cold-Start Recommendation Under Exposure Bias
               and Concept Drift},
  author    = {Hassan, Ali},
  booktitle = {Proceedings of the CONSEQUENCES Workshop at ACM RecSys 2026},
  year      = {2026},
  note      = {Collaborating with Dr. Yan Zhang,
               Charles Darwin University}
}
```

---

## License

Code: MIT License | Dataset: CC BY 4.0
