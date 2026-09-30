# Fashion-Retailer Retention Analysis

> **Where should a fashion retailer focus retention effort, and how much revenue is at stake?**

*Headline finding and key metrics will be auto-filled after the analysis pipeline runs.*

---

## Quick start

```bash
# 1. Set up environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data (see data/README.md)

# 3. Run full pipeline
make all

# 3b. Or run on a 5% sample (minutes on a laptop)
make sample

# 4. Run tests
make test
```

## Repository map

```
.
├── README.md                 ← you are here
├── Makefile                  ← one target per milestone
├── config.yaml               ← all tunable parameters
├── data/README.md            ← download instructions
├── docs/decisions.md         ← every judgement call + evidence
├── src/                      ← all analysis code
├── notebooks/                ← thin display notebooks
├── tests/                    ← pytest suite incl. leakage tests
├── results/                  ← JSON outputs (source of every number)
└── reports/
    ├── figures/              ← charts
    ├── tables/               ← CSV tables
    └── memo.md               ← two-page consultant memo
```

## Data note

This analysis uses the publicly available **H&M Personalized Fashion Recommendations** dataset from Kaggle. It is **not** data from any employer. Revenue figures are presented as an **index** (base 100) or as a **share of total**, because prices in the dataset are anonymised.


