.PHONY: help all sample m0 m1 m2 m3 m4 m5 m6 m7 m8 m9 test clean

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
SAMPLE_FLAG :=

help:  ## Show available targets
	@echo ""
	@echo "  Fashion-Retailer Retention Analysis"
	@echo "  ──────────────────────────────────────────────"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'
	@echo ""

# ── Full pipeline ───────────────────────────────────────────────────
all: m1 m2 m3 m4 m5 m6 m7 m8 m9  ## Run the full pipeline (requires data/)

sample: SAMPLE_FLAG := --sample
sample: m1 m2 m3 m4 m5 m6 m7 m8 m9  ## Run the full pipeline on a 10 %% customer sample

# ── Milestones ──────────────────────────────────────────────────────
m0: ## M0  Scaffold (you are here)
	@echo "Scaffold complete."

m1: ## M1  Ingest CSVs → Parquet + schema report
	$(PYTHON) src/ingest.py $(SAMPLE_FLAG)

m2: m1 ## M2  Data-quality report
	$(PYTHON) src/quality.py $(SAMPLE_FLAG)

m3: m2 ## M3  Descriptive analytics + charts
	$(PYTHON) src/descriptives.py $(SAMPLE_FLAG)

m4: m3 ## M4  RFM segmentation + cohort retention
	$(PYTHON) src/segments.py $(SAMPLE_FLAG)
	$(PYTHON) src/cohorts.py $(SAMPLE_FLAG)

m5: m4 ## M5  Leakage-safe train / val / test snapshots
	$(PYTHON) src/snapshots.py $(SAMPLE_FLAG)

m6: m5 ## M6  Lapse model (baseline → XGBoost)
	$(PYTHON) src/model.py $(SAMPLE_FLAG)

m7: m3 ## M7  Basket-affinity analysis
	$(PYTHON) src/affinity.py $(SAMPLE_FLAG)

m8: m6 ## M8  Opportunity sizing
	$(PYTHON) src/sizing.py $(SAMPLE_FLAG)

m9: m8 m7 ## M9  Memo, README, polish
	$(PYTHON) src/memo.py
	@echo "Run: make test  to verify."

test: ## Run all tests
	$(PYTEST) tests/ -v

clean: ## Remove generated outputs (keeps raw data)
	rm -rf results/*.json reports/figures/*.png reports/tables/*.csv
	rm -rf data/*.parquet
	@echo "Cleaned generated outputs."
