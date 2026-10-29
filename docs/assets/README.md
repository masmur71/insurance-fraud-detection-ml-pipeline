# README screenshot provenance

These are browser screenshots of actual local outputs, captured on October 29,
2026. They contain synthetic experiment data only.

- `mlflow-run.png`: retained `frozen-model-reproduction` experiment, run
  `2491f9dbff8d4741a3a91f9f752e469c`, viewed in the local MLflow UI. This is a
  reproduction of the frozen recipe, not fresh model selection. The image is a
  labeled contact sheet of the Overview and Artifacts tabs from the same run;
  UI content is unmodified. See the
  [tracking workflow](../operations.md#local-mlflow).
- `drift-controlled-shift.png`: standalone HTML output from a fresh execution of
  `scripts/demo_drift.py --output-dir reports/readme-drift-demo`, at
  `reports/readme-drift-demo/controlled-shift/drift-report.html`.
  Reference and input each contain 78,352 training-derived rows. The deterministic
  fixture changes vehicle age missingness, vehicle value, and incident category;
  it does not read consumed holdout rows. See the [drift workflow](../data-drift.md).

The local tracking database, model files, reports, and browser profiles remain
excluded from Git. Screenshots are illustrative snapshots; commands in the linked
guides reproduce the workflows, and newly built artifact identifiers may differ.
