# Decision Log

Every non-obvious judgement call is recorded here with the evidence that justified it.

---

## Format

Each entry follows:

> **Decision:** What was decided
> **Evidence:** What data or reasoning supported it
> **Effect:** Impact on the analysis (e.g. rows dropped, scope change)

---

*Entries will be added as the analysis progresses through each milestone.*


## M2: Data-Quality Cleaning Decisions

1. **Duplicate transaction rows**: Found 149,963 rows that are exact duplicates across all columns. Treatment: **kept**, because a customer can legitimately purchase multiple units of the same item in the same transaction. There is no unique transaction ID to distinguish them.

2. **Implausible ages**: 0 customers have age < 15 or > 100 (range [16, 99]). Treatment: **set to NULL**. A fashion retailer's customer base is unlikely to include infants or centenarians at scale; these are likely data entry errors. Median age = 31.

3. **Customers with no transactions**: 464 (0.7% of customer table). Treatment: **excluded from all analysis**. Without purchase history, these customers cannot be segmented or modelled.

4. **Price outliers**: Prices are anonymised/normalised. Distribution: min=0.0001, p25=0.0158, median=0.0254, p75=0.0339, p99=0.0950, max=0.5915. Treatment: **no removal**. We cannot judge which prices are implausible without knowing the normalisation scheme.

**Net effect**: 464 customer records flagged for cleaning. No transaction rows were dropped.


## M5: Label Window Justification

**Decision:** Label window set to 12 weeks (84 days).

**Evidence:** Inter-purchase gap distribution: median = 22d, p75 = 58d, p90 = 124d, p95 = 188d, mean = 48.2d. A 12-week window (84d) is well beyond the 75th percentile inter-purchase gap, meaning most active customers would have made at least one purchase in this window if they intend to continue buying.

**Effect:** This is used to define the binary lapse label for modelling.


## M2: Data-Quality Cleaning Decisions

1. **Duplicate transaction rows**: Found 299,319 rows that are exact duplicates across all columns. Treatment: **kept**, because a customer can legitimately purchase multiple units of the same item in the same transaction. There is no unique transaction ID to distinguish them.

2. **Implausible ages**: 0 customers have age < 15 or > 100 (range [16, 99]). Treatment: **set to NULL**. A fashion retailer's customer base is unlikely to include infants or centenarians at scale; these are likely data entry errors. Median age = 31.

3. **Customers with no transactions**: 971 (0.7% of customer table). Treatment: **excluded from all analysis**. Without purchase history, these customers cannot be segmented or modelled.

4. **Price outliers**: Prices are anonymised/normalised. Distribution: min=0.0001, p25=0.0158, median=0.0254, p75=0.0339, p99=0.0961, max=0.5915. Treatment: **no removal**. We cannot judge which prices are implausible without knowing the normalisation scheme.

**Net effect**: 971 customer records flagged for cleaning. No transaction rows were dropped.


## M5: Label Window Justification

**Decision:** Label window set to 12 weeks (84 days).

**Evidence:** Inter-purchase gap distribution: median = 22d, p75 = 58d, p90 = 124d, p95 = 187d, mean = 48.2d. A 12-week window (84d) is well beyond the 75th percentile inter-purchase gap, meaning most active customers would have made at least one purchase in this window if they intend to continue buying.

**Effect:** This is used to define the binary lapse label for modelling.
