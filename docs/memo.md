# Memorandum: Targeted Customer Retention Strategy

**To:** Chief Commercial Officer  
**From:** Strategy Analytics Team  
**Date:** 2026-10-01  
**Subject:** Sizing and targeting the retention opportunity  

---

## 1. The Problem: Retention Volatility and Revenue Concentration

Our analysis of the transaction data reveals a highly concentrated revenue base coupled with steep early-tenure attrition. 

**Revenue Concentration:** The "Champions" segment—our most frequent and highest-spending customers—accounts for just 24.6% of the active customer base, yet generates **66.3%** of total revenue. This top-heavy distribution exposes the business to significant risk if these core customers lapse.

**Retention Baseline:** Customer retention drops sharply in the early lifecycle. On average, only **16.0%** of a cohort remains active six months after their first purchase. The majority of newly acquired customers fail to form a habit, leaving a substantial proportion of marketing acquisition spend unrecovered.

Currently, retention efforts are broadly applied or rely on simple recency rules, resulting in inefficient spend.

## 2. The Opportunity: Predictive Targeting and Value at Stake

By applying a machine-learning approach to predict lapsing behavior, we can target interventions much more precisely.

**Predictive Power:** Our calibrated XGBoost lapse model significantly outperforms simple recency rules. In a challenging environment with a high baseline lapse rate (59.1%), raw recall metrics are mathematically capped. Instead, we evaluate precision: the model achieves a **92.7% accuracy** in its top risk decile. In other words, over 9 out of 10 customers flagged in this highest-risk group will genuinely lapse if no action is taken. This captures **14.9%** of all lapsers, representing **88%** of the theoretical maximum possible in a single decile.

**Value at Stake:** We isolated a high-priority target group: customers who belong to high-value segments (Champions, Loyal, Promising) but are identified by the model as being in the highest risk decile for lapsing in the next 12 weeks. 
- **Target Audience:** 14,271 high-value, high-risk customers.
- **Value at Risk:** We estimate the expected revenue loss from this specific group lapsing is **1.57 index points** per quarter (where total quarterly revenue = 100.0).

**Scenario Sizing:** A targeted intervention program pays for itself rapidly. If an intervention achieves a conservative **5 percentage point uplift** in retention for this target group, it would recover **0.12 index points** in revenue. Under this scenario, the break-even cost is **5.0%** of the average retained customer's quarterly spend, leaving ample margin for promotional offers or dedicated outreach.

## 3. The Action Plan: Next Best Action and Interventions

We recommend shifting from generic retention campaigns to highly targeted, predictive interventions for the 14,271 high-risk, high-value customers. 

**Intervention Strategy:**
1. **Targeted Deployment:** Deploy the lapse prediction model to score the customer base weekly.
2. **Early Intervention:** Engage customers as they enter the top risk decile—*before* they have completely lapsed (the median inter-purchase gap is around 3-4 weeks; waiting longer significantly reduces recovery chances).
3. **Cross-Category Recommendations (Basket Affinity):** When deploying targeted offers, use proven basket affinities to encourage broader catalog discovery, which increases stickiness. For example, our affinity analysis shows that customers buying a **Bikini top** are **12.1x** more likely to also purchase a **Swimwear bottom**. Promoting these natural complements (rather than discounting items they would buy anyway) provides a strong value proposition.

**Next Steps:** We propose a controlled A/B test of this predictive targeting strategy over a 6-week window, offering complementary affinity-based products to the target group.
