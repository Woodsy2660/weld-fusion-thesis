"""UT classifier training skeleton (Random Forest primary, XGBoost comparison).

Contract:
    - Features from configs/ut_train.yaml (six locked names).
    - Labels in the frame from configs/frame.yaml.
    - Splits MUST be GroupKFold grouped by coupon_id. Cell-level random
      splits are a bug (see evaluation.splits).
    - Fit isotonic (primary) or Platt (alternative) calibration on inner
      folds only; compute reliability alpha the same way.
    - Never train on DEV_ONLY synthetic features for reported results.

This module does not execute. Blocked on labelled A-scans with coupon_id.
"""

from __future__ import annotations

from typing import Any


def train_ut(config_path: str = "configs/ut_train.yaml", **kwargs: Any) -> dict:
    """Fit RF and XGBoost, return paths to joblib artefacts and a manifest."""
    raise NotImplementedError(
        "UT training is stubbed. Contract: GroupKFold by coupon_id, "
        "RandomForestClassifier (primary) and XGBClassifier (comparison), "
        "inner-fold isotonic calibration, reliability alpha, provenance "
        "manifest. Do not call this on DEV_ONLY synthetic features for "
        "anything that will be reported."
    )


def main() -> None:
    train_ut()


if __name__ == "__main__":
    main()
