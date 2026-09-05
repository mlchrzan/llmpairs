from __future__ import annotations

from pathlib import Path

import pandas as pd
import pairadigm as pdm

from bt_scaling import score_pairadigm_models


MODEL_NAMES = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.7-flash",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
]
TABLES = [
    "gilbert_meta_2",
    "gilbert_meta_7",
    "gilbert_meta_8",
    "gilbert_meta_102",
    "gilbert_meta_103",
    "gilbert_meta_104",
    "frac20",
]


def replace_bt_columns(
    existing: pd.DataFrame,
    replacements: pd.DataFrame,
) -> pd.DataFrame:
    """Replace all prior BT score columns while preserving other result columns."""
    bt_columns = [
        column for column in existing.columns if column.startswith("Bradley_Terry_")
    ]
    existing_without_bt = existing.drop(columns=bt_columns).copy()
    existing_without_bt["_item_key"] = existing_without_bt["item"].astype(str)
    replacement_values = replacements.drop(columns="item").copy()
    replacement_values["_item_key"] = replacements["item"].astype(str)
    result = existing_without_bt.merge(
        replacement_values,
        on=["table", "_item_key"],
        how="left",
        validate="one_to_one",
    ).drop(columns="_item_key")
    new_bt_columns = [
        column for column in replacements.columns if column.startswith("Bradley_Terry_")
    ]
    if result[new_bt_columns].isna().any().any():
        raise ValueError("Some items were not matched to regenerated BT scores.")
    return result


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    results_dir = repository_root / "results"
    aggregate_path = results_dir / "all_tables_item_scores.csv"
    aggregate = pd.read_csv(aggregate_path)
    score_updates = []
    scaling_parameters = []

    for table in TABLES:
        pairadigm_dir = results_dir / table / "pairadigm_results"
        print(f"Rescoring saved Pairadigm object: {table}")
        pairadigm = pdm.load_pairadigm(str(pairadigm_dir))
        scored_df, table_scaling = score_pairadigm_models(pairadigm, MODEL_NAMES)
        pairadigm.scored_df = scored_df
        pairadigm.save()

        score_columns = [
            column
            for column in scored_df.columns
            if column.startswith("Bradley_Terry_")
        ]
        table_scores = scored_df[["item", *score_columns]].copy()
        table_scores.insert(0, "table", table)
        score_updates.append(table_scores)

        table_scaling.insert(0, "table", table)
        scaling_parameters.append(table_scaling)

        table_results_path = results_dir / table / f"{table}_all_item_scores.csv"
        if table_results_path.exists():
            table_results = pd.read_csv(table_results_path)
            table_replacements = table_scores.drop(columns="table").copy()
            table_bt_columns = [
                column
                for column in table_results.columns
                if column.startswith("Bradley_Terry_")
            ]
            table_results = table_results.drop(columns=table_bt_columns).copy()
            table_results["_item_key"] = table_results["item"].astype(str)
            table_replacement_values = table_replacements.drop(columns="item").copy()
            table_replacement_values["_item_key"] = table_replacements["item"].astype(str)
            table_results = table_results.merge(
                table_replacement_values,
                on="_item_key",
                how="left",
                validate="one_to_one",
            ).drop(columns="_item_key")
            if table_results[score_columns].isna().any().any():
                raise ValueError(f"Some {table} items were not matched during rescoring.")
            table_results.to_csv(table_results_path, index=False)

    replacements = pd.concat(score_updates, ignore_index=True)
    updated_aggregate = replace_bt_columns(aggregate, replacements)
    updated_aggregate.to_csv(aggregate_path, index=False)
    pd.concat(scaling_parameters, ignore_index=True).to_csv(
        results_dir / "all_BT_scaling_parameters.csv",
        index=False,
    )
    print(f"Updated {len(updated_aggregate)} aggregate item rows without LLM calls.")


if __name__ == "__main__":
    main()