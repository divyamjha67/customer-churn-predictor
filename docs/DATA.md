# Data provenance and feature contract

## Source

- Provider: IBM, `telco-customer-churn-on-icp4d` repository.
- Revision: `d5371f5d83a446ad5673cbcca3b814b926491f8a`.
- Original file: `data/Telco-Customer-Churn.csv`.
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
- Dimensions: 7,043 customer rows, 21 original columns.
- Label: `Churn`, mapped from Yes/No to 1/0. There are 1,869 positive labels.
- Provenance record: `data/source.json`; upstream license: `data/UPSTREAM_LICENSE.txt`.

IBM's [sample description](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) describes a fictional telecom company and a churn label indicating customers who left within the last month. The classic CSV used here has no event timestamps or feature-availability timestamps. This project does not claim that the inputs precede every departure or establish a validated next-month prediction window.

The download script verifies the pinned SHA-256 before overwriting the raw file. Training verifies it again, checks the row count, requires unique customer IDs, and rejects unexpected labels. Eleven blank TotalCharges values are represented as missing and handled by training-fitted median imputation. Numeric conversion itself learns nothing from the population.

## Inputs

The scoring model uses 15 columns. Names are case-sensitive. CSV files require all 15 headers, plus an optional `customerID`. `TotalCharges` may be blank; the other values are required. JSON may omit `TotalCharges` or send null.

| Field | Values / unit | Meaning |
| --- | --- | --- |
| tenure | Whole months, 0–1200 | Months with the provider |
| MonthlyCharges | Finite number, 0–1000 | Monthly charge in sample currency (USD) |
| TotalCharges | Finite number, 0–1000000, or missing | Cumulative charge |
| Contract | Month-to-month; One year; Two year | Contract term |
| InternetService | DSL; Fiber optic; No | Internet type |
| OnlineSecurity | Yes; No; No internet service | Security subscription |
| TechSupport | Yes; No; No internet service | Support subscription |
| OnlineBackup | Yes; No; No internet service | Backup subscription |
| DeviceProtection | Yes; No; No internet service | Protection subscription |
| PaymentMethod | Electronic check; Mailed check; Bank transfer (automatic); Credit card (automatic) | Payment channel |
| PaperlessBilling | Yes; No | Electronic billing |
| PhoneService | Yes; No | Phone subscription |
| MultipleLines | Yes; No; No phone service | Multiple phone lines |
| StreamingTV | Yes; No; No internet service | TV streaming subscription |
| StreamingMovies | Yes; No; No internet service | Movie streaming subscription |

Numeric bounds are defensive application limits, not the benchmark's observed training range or a promise of reliability throughout those intervals. Inputs far outside the training distribution can yield unreliable estimates. Production scoring needs distribution checks and monitoring.

If InternetService is No, all six internet add-on fields must be No internet service; otherwise that value is forbidden. PhoneService and MultipleLines must likewise agree. The form updates dependent defaults when the service changes. It still validates the final submitted profile.

`customerID` is only a display/ranking identifier. It never enters preprocessing or the estimator. Batch IDs must be unique and at most 80 characters. Blank/missing IDs receive a row label. Gender, SeniorCitizen, Partner, and Dependents are excluded from the model. Churn is never accepted as a scoring input. The full original CSV therefore cannot be uploaded directly; use the provided unlabeled example or a file with the scoring schema.

## Transformations

Numeric columns use median imputation followed by standardization. Categorical columns use dense one-hot encoding. These transforms are fitted within each cross-validation training fold and inside the final training-only pipeline. No separate inference implementation recreates feature engineering. Unknown categorical values are rejected at the input boundary and by the encoder.

No label encoding based on outcomes, SMOTE, or post-churn fields are used. HistGradientBoosting also performs its internal binning/early-stopping logic within the training data supplied to it. All external validation and test customers stay outside that fitting process.

## Privacy and retention

Training inputs and example IDs belong to a fictional public sample. Scoring requests stay in application memory; there is no application database or raw-input logging. The server's normal access logs include route/status information. Browser exports are downloaded only when requested. A hosted environment may have additional infrastructure logs and policies; those are outside this local application's guarantees.
