from __future__ import annotations

import numpy as np
import pandas as pd


VALID_BT_DECISIONS = {"Text1", "Text2", 0, 1}


def bt_log_score_standard_errors(
    pairwise_df: pd.DataFrame,
    log_scores: pd.Series,
    decision_col: str,
) -> pd.Series:
    """Estimate BT log-score SEs from the constrained Fisher information."""
    required = {"item1", "item2", decision_col}
    missing = required - set(pairwise_df.columns)
    if missing:
        raise ValueError(f"Pairwise data is missing required columns: {missing}")
    if log_scores.index.has_duplicates:
        raise ValueError("log_scores must have one value per item.")
    if not np.isfinite(log_scores.to_numpy(dtype=float)).all():
        raise ValueError("log_scores must be finite.")

    valid_pairs = pairwise_df.loc[
        pairwise_df[decision_col].isin(VALID_BT_DECISIONS),
        ["item1", "item2"],
    ]
    if valid_pairs.empty:
        raise ValueError(f"No valid comparisons found in '{decision_col}'.")

    item_to_index = {item: index for index, item in enumerate(log_scores.index)}
    item_i = valid_pairs["item1"].map(item_to_index)
    item_j = valid_pairs["item2"].map(item_to_index)
    if item_i.isna().any() or item_j.isna().any():
        raise ValueError("Pairwise data contains items missing from log_scores.")

    item_i = item_i.to_numpy(dtype=int)
    item_j = item_j.to_numpy(dtype=int)
    score_values = log_scores.to_numpy(dtype=float)
    score_differences = score_values[item_i] - score_values[item_j]
    probabilities = np.exp(-np.logaddexp(0.0, -score_differences))
    information_weights = probabilities * (1.0 - probabilities)

    n_items = len(log_scores)
    information = np.zeros((n_items, n_items), dtype=float)
    np.add.at(information, (item_i, item_i), information_weights)
    np.add.at(information, (item_j, item_j), information_weights)
    np.add.at(information, (item_i, item_j), -information_weights)
    np.add.at(information, (item_j, item_i), -information_weights)

    information_rank = np.linalg.matrix_rank(information)
    if information_rank != n_items - 1:
        raise ValueError(
            "The comparison graph is disconnected; log-score SEs are not identified."
        )

    covariance = np.linalg.pinv(information, hermitian=True)
    variances = np.clip(np.diag(covariance), a_min=0.0, a_max=None)
    return pd.Series(
        np.sqrt(variances),
        index=log_scores.index,
        name="log_score_se",
    )


def z_standardize_estimates(
    scores: pd.Series,
    standard_errors: pd.Series,
) -> tuple[pd.Series, pd.Series, float, float]:
    """Z-standardize estimates and propagate SEs through the linear transform."""
    if not scores.index.equals(standard_errors.index):
        raise ValueError("scores and standard_errors must have identical indexes.")

    score_mean = float(scores.mean())
    score_sd = float(scores.std(ddof=1))
    if not np.isfinite(score_sd) or score_sd <= 0:
        raise ValueError("Scores must have a positive finite sample standard deviation.")

    z_scores = (scores - score_mean) / score_sd
    z_standard_errors = standard_errors / score_sd
    return z_scores, z_standard_errors, score_mean, score_sd


def score_pairadigm_models(
    pairadigm,
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit saved Pairadigm decisions as raw and standardized BT estimates."""
    all_scores = None
    scaling_parameters = []
    save_dir = pairadigm.save_dir
    pairadigm.save_dir = None

    try:
        for model_name in model_names:
            decision_col = f"decision_{model_name}"
            scored_df = pairadigm.score_items(
                normalization_scale="none",
                update_classObject=False,
                summarize=False,
                decision_col=decision_col,
            )
            pairadigm_score_col = f"Bradley_Terry_Score_{model_name}"
            if pairadigm_score_col not in scored_df.columns:
                raise ValueError(
                    f"Expected Bradley-Terry score column '{pairadigm_score_col}'."
                )

            log_scores = scored_df.set_index("item")[pairadigm_score_col]
            log_standard_errors = bt_log_score_standard_errors(
                pairadigm.pairwise_df,
                log_scores,
                decision_col,
            )
            z_scores, z_standard_errors, score_mean, score_sd = (
                z_standardize_estimates(log_scores, log_standard_errors)
            )

            model_scores = pd.DataFrame(
                {
                    "item": log_scores.index,
                    f"Bradley_Terry_Log_Score_{model_name}": log_scores.to_numpy(),
                    f"Bradley_Terry_Log_SE_{model_name}": log_standard_errors.to_numpy(),
                    f"Bradley_Terry_Z_Score_{model_name}": z_scores.to_numpy(),
                    f"Bradley_Terry_Z_SE_{model_name}": z_standard_errors.to_numpy(),
                }
            )
            scaling_parameters.append(
                {
                    "model": model_name,
                    "score_mean": score_mean,
                    "score_sd": score_sd,
                    "n_items": len(log_scores),
                }
            )

            if all_scores is None:
                legacy_score_columns = [
                    column
                    for column in scored_df.columns
                    if column.startswith(
                        (
                            "Bradley_Terry_Score_",
                            "Bradley_Terry_SE_",
                            "Bradley_Terry_Log_",
                            "Bradley_Terry_Z_",
                        )
                    )
                ]
                all_scores = scored_df.drop(columns=legacy_score_columns).copy()
            all_scores = all_scores.merge(model_scores, on="item", how="left")
    finally:
        pairadigm.save_dir = save_dir

    if all_scores is None:
        raise ValueError("model_names must contain at least one model.")
    return all_scores, pd.DataFrame(scaling_parameters)