# Model card — version 2.0.0

## Purpose

Estimate association with the IBM telecom sample's churn label and rank profiles for human review. This is a reproducible benchmark and local decision-support prototype. It is not a validated production forecast, automated customer-treatment system, or estimate of the effect of a retention offer.

## Selection protocol

1. Verify the dataset hash, row count, customer-ID uniqueness, feature values, and labels.
2. Use a stratified seed-42 split: 60% training, 20% validation, 20% test. No customer ID overlaps partitions. `artifacts/splits.json` records assignment.
3. Compare four complete pipelines with four-fold stratified CV on training customers only. Scoring is average precision (AP). Preprocessing is fitted separately inside each fold.
4. Select the candidate with highest mean CV AP, then fit it on the 4,225 training customers.
5. On the 1,409 validation customers, examine thresholds 0.05–0.95 at steps of 0.01. Maximize F2, which weights recall more than precision. If numerically tied, prefer the higher threshold. The selected threshold is 0.13.
6. Freeze the pipeline and threshold. Evaluate once on the 1,409 test customers. Do not refit on validation/test, use test scores to select a winner, or change the policy after seeing test results.

Re-running the same experiment to reproduce it is different from using its test metrics to tune a new candidate. Future tuning requires a new final evaluation strategy; this test set should be considered seen.

## Candidates

| Model | Training CV AP mean | Fold SD |
| --- | ---: | ---: |
| Prior-only dummy | 0.265 | 0.000 |
| Logistic regression, C=0.1 | 0.658 | 0.038 |
| Logistic regression, C=1 | 0.660 | 0.038 |
| Histogram gradient boosting | 0.663 | 0.023 |

Gradient boosting wins this selection rule, but its CV advantage over logistic regression is small. This comparison does not establish a statistically significant superiority. Logistic regression provides a useful interpretable reference.

Selected configuration: 150 maximum iterations, 15 maximum leaves, learning rate 0.05, L2 regularization 5, random state 42. Other settings use scikit-learn 1.7.2 defaults. Logistic configurations use max_iter=2000. No class weighting, oversampling, or explicit probability recalibration is applied.

## Test performance

ROC-AUC 0.8287; AP 0.6204; Brier score 0.1431. Churn prevalence is 26.54%. AP measures precision integrated over changes in recall; it is not accuracy and is not identical to trapezoidal PR-AUC. The prior-only baseline's AP is approximately prevalence.

| Threshold | Accuracy | Precision | Recall | F1 | F2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.13, selected on validation | 0.654 | 0.428 | 0.906 | 0.581 | 0.741 |
| 0.50, conventional reference | 0.786 | 0.623 | 0.487 | 0.547 | 0.509 |

At 0.13: TN=582, FP=453, FN=35, TP=339. The policy contacts 56.2% of test customers and produces many false positives. F2 reflects a design preference for recall, not a measured cost function. The conventional 0.5 threshold improves precision and accuracy while missing more positive labels. Neither cutoff is a universal business optimum.

For capacity-constrained outreach, the top 20% selects 282 customers, includes 178 observed churners, has 63.1% precision, and captures 47.6% of all 374 churners. Lift is 0.6312 / 0.2654 = 2.38. Ties retain input order; an alternative tie policy could change a boundary selection. These are retrospective ranking statistics, not retention outcomes.

The dashboard shows reliability bins based on test predicted probabilities (eight quantile bins). Brier score and reliability plots characterize probability quality; they do not certify calibration in a new population.

## Uncertainty and interpretation

300 seed-42 bootstrap resamples of the fixed test predictions produce percentile 95% intervals: ROC-AUC 0.809–0.851; AP 0.569–0.674. They describe sampling variation conditional on this split and fitted model. They exclude the full uncertainty from model selection, data collection, temporal drift, and demographic or service-population shifts.

Feature importance uses five repeats of permutation on validation data with AP as the scoring function. The display shows mean AP loss after shuffling one original feature. This is global predictive reliance, not an individual explanation or causal effect. Correlated features can share predictive information and reduce one another's measured importance. Standard deviations for all features are retained in the report.

## Limits and deployment requirements

- The source is a fictional benchmark. No operational external validation is available.
- No observation timestamps: temporal separation, feature availability before churn, and a future outcome horizon cannot be established.
- No causal inference: lowering a charge, changing a contract, or making a call is not proven to reduce churn by these scores.
- No business-impact claim: intervention response, margin, offer cost, and opportunity cost are absent. A controlled experiment is required for savings or uplift claims.
- No fairness guarantee: excluding demographic fields does not remove correlated proxies. Subgroup analysis requires an appropriate evaluation plan and lawful data access.
- No drift monitor or scheduled retraining is implemented. A version string and artifact hash identify this build; they are not a full model registry.
- No authentication or production request controls. This build is intended for local use.

Only trusted, locally trained joblib artifacts should be loaded. A checksum verifies consistency with the report, not the trustworthiness of arbitrary serialized code. Record dataset, environment, code revision, and model report for any future deployment.

## Reproduction

Install the locked Python 3.12 environment, run `scripts/download_data.py`, then `python -m churn.train`. The report records dataset and model hashes, library version, seed, scores, and split sizes. Joblib bytes and training timestamps can vary between runs/platforms even where predictions reproduce; compare metrics and split assignments with sensible floating-point tolerance.
