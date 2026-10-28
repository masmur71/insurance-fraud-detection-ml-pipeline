"""Read-only foundation checks; run from repository root."""
import json
from pathlib import Path

import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    tables = {name: pd.read_parquet(data / f"{name}.parquet") for name in
              ("customers", "policies", "claims", "workshops", "claims_truth", "claims_features")}
    manifest = json.loads((data / "manifest.json").read_text())
    config = json.loads((data / "generator_config.json").read_text())
    claims = tables["claims"]
    unique = claims.drop_duplicates("claim_id", keep="first")
    require(claims.drop_duplicates().shape[0] == len(unique), "Conflicting claim duplicates")
    for table, key in (("customers", "customer_id"), ("policies", "policy_id"),
                       ("workshops", "workshop_id"), ("claims_truth", "claim_id"),
                       ("claims_features", "claim_id")):
        require(tables[table][key].notna().all() and tables[table][key].is_unique,
                f"Invalid primary key: {table}")
    for column, parent in (("policy_id", "policies"), ("customer_id", "customers"),
                           ("workshop_id", "workshops")):
        require(claims[column].isin(tables[parent][column]).all(), f"Orphan {column}")
    policies = tables["policies"]
    require(policies.customer_id.isin(tables["customers"].customer_id).all(), "Orphan policy customer")
    owners = policies.set_index("policy_id").customer_id
    require(claims.policy_id.map(owners).eq(claims.customer_id).all(), "Policy owner mismatch")
    features = tables["claims_features"]
    for table in ("claims_truth", "claims_features"):
        require(set(tables[table].claim_id) == set(unique.claim_id), f"Coverage mismatch: {table}")
    require(unique.is_fraud.isin([0, 1]).all(), "Nonbinary target")
    require(features.set_index("claim_id").is_fraud.sort_index().equals(
        unique.set_index("claim_id").is_fraud.sort_index()), "Feature target mismatch")
    require(not {"true_fraud", "detected", "fraud_probability", "ring_era"}.intersection(features),
            "Truth leaked into stored features")
    require({"claim_status", "payout_amount", "siu_investigation_flag", "days_to_settlement"}
            .issubset(features), "Documented leakage trap missing")
    dates = features.incident_date
    expected = pd.Series("test", index=features.index)
    expected.loc[dates < "2024-07-01"] = "validation"
    expected.loc[dates < "2024-01-01"] = "train"
    require(expected.eq(features.split).all(), "Split boundary mismatch")
    require(dates.between("2021-01-01", "2025-12-31").all(), "Incident outside configured period")
    counts = {"customers": len(tables["customers"]), "policies": len(policies),
              "claims_including_duplicates": len(claims), "claims_unique": len(unique),
              "workshops": len(tables["workshops"])}
    require(counts == manifest["row_counts"] == config["row_counts"], "Row count mismatch")
    require(len(features) == manifest["feature_rows"] and
            len(features.columns) == manifest["feature_columns"], "Feature schema mismatch")
    require(abs(claims.is_fraud.mean() - manifest["observed_fraud_rate"]) < 1e-12,
            "Manifest prevalence mismatch")
    print("PASS: keys, identical duplicates, joins, target, truth exclusion, dates, splits, manifest")
    print(f"Synthetic data; seed {manifest['seed']}; unique claims {len(unique)}; "
          f"observed fraud {unique.is_fraud.mean():.4%}")
    print("Split counts:", features.split.value_counts().to_dict())
    print("Negative report delays:", int((unique.report_date < unique.incident_date).sum()))
    for split, cutoff in (("train", "2024-01-01"), ("validation", "2024-07-01")):
        subset = features.loc[features.split.eq(split)]
        print(f"{split}: reports unavailable at cutoff: {int((subset.report_date >= cutoff).sum())}")


if __name__ == "__main__":
    main()
