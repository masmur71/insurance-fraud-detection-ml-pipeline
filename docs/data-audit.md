# Data audit

Verified with `python scripts/audit_data.py` on Python 3.13.18 / pandas 2.2.3.
The service and CI use Python 3.11; these audit figures record the original run.

Synthetic data, seed 20261028: 150,300 raw rows, 150,000 unique claims,
3.0707% observed fraud among unique claims. The manifest's 3.0725% rate uses
raw rows including duplicates; these figures have different denominators.
Train / validation / test: 79,329 / 15,555 / 55,116 before availability filtering.

Passed primary-key, identical-duplicate, foreign-key, policy-owner, truth-coverage,
target-consistency, feature-shape, date-range, split-boundary and manifest checks.
The four documented leakage columns are present intentionally; truth columns are absent
from the stored feature table. Feature allowlisting is mandatory before training.

There are 450 unique negative-delay claims, versus 451 raw rows. The EDA summary
now distinguishes these counts. Generator weights are odds multipliers, not direct
logit coefficients; both data READMEs have been corrected.

There are 977 train records reported after the train cutoff and 955 validation
records reported after the validation cutoff. Training excludes these from
the respective fitting/selection sets and reports the resulting counts.
Historical counts must be rebuilt using report availability, with no same-day peers.
Label maturity cannot be verified because label-observation timestamps are absent.

No data was regenerated, and notebook outputs were not rerun.
The executable audit checks data consistency before model training.
