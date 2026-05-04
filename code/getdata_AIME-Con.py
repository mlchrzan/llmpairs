import dotenv
import os
import sys
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import pandas as pd
import irw
import polars as pl
import openai
import pairadigm as pdm
import mirt
dotenv.load_dotenv()

CGCoT_PROMPTS = [
    "Summarize the core task, question, behavioral statement, or scenario presented in this item. Item text: {text}",
    "Identify the primary knowledge domain, skill, clinical symptom, attitudinal belief, or psychological construct this item is attempting to measure and give a concise justification. Text: {text}",
    "Analyze the demand or severity of this item. For cognitive/ability items, does it require recall, multi-step reasoning, or synthesis? For psychological/attitudinal items, does endorsing it require a severe level of a symptom, a strongly held belief, or deep introspection? CONCISELY identify any structural or semantic elements—such as complex phrasing, tricky distractors, conditional logic, or strong emotional wording—that increase the cognitive load or the threshold for endorsement. Text: {text}",
    "Based on your analysis, describe the overall level of the underlying latent trait (e.g., high cognitive ability, severe psychological symptomology, strong attitude) required for a respondent to successfully answer or strongly endorse this item. Previous analysis: {text}"
]

# MODEL_NAMES=["gemini-3-flash-preview", 'gpt-5.4-mini']
MODEL_NAMES=["gemini-3-flash-preview", 'gemini-3.1-pro-preview', 
             'gpt-5.4-mini','gpt-5.4']
API_KEYS=[os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY"), 
          os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_API_KEY")]
BASE_URLS=[None, "https://us.api.openai.com/v1"]

np.random.seed(1234)  # For reproducibility of any random processes

# Pull all IRW tables that have item text available, age range is <=18, construct is cognitive/educational, measurement tool is test, item format is Likert/constructed response, and language is English
irw_edtexts = irw.fetch([
    'gilbert_meta_102',
    'gilbert_meta_103',
    'gilbert_meta_104',
    'gilbert_meta_2' 
    'frac20',
    # 'gilbert_meta_23', no item text available for this table
    # 'gilbert_meta_26', # Removed for being longitudinal
    # 'preschool_sel_akt', # Removed because the items also imply a picture was used which is not available to us'preschool_sel_akt'
])

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_log_path():
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
    return logs_dir / f"analysis_AIME-Con_{log_timestamp}.log"

def fit_2pl_mirt(
    df,
    id_col="id",
    item_col="item",
    resp_col="resp",
    n_quadpts=21,
    max_iter=500,
    tol=1e-4,
    score_method="EAP",
    use_rust=False,
    verbose=False,
):
    # Accept polars or pandas input
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    x = df[[id_col, item_col, resp_col]].dropna().copy()
    x[resp_col] = x[resp_col].astype(int)

    bad = set(x[resp_col].unique()) - {0, 1}
    if bad:
        raise ValueError(f"{resp_col} must be binary (0/1). Found: {sorted(bad)}")

    # Guard against duplicate person-item rows
    dup_mask = x.duplicated([id_col, item_col], keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        raise ValueError(
            f"Found {n_dup} duplicate person-item rows. "
            "Resolve duplicates before fitting mirt."
        )

    # Stable coding so joins/merges are predictable
    person_codes, person_labels = pd.factorize(x[id_col], sort=True)
    item_codes, item_labels = pd.factorize(x[item_col], sort=True)

    n_persons = len(person_labels)
    n_items = len(item_labels)

    # mirt expects 2D matrix: rows=persons, cols=items; missing coded as -1
    responses = np.full((n_persons, n_items), -1, dtype=np.int32)
    responses[person_codes, item_codes] = x[resp_col].to_numpy(dtype=np.int32)

    fit_result = mirt.fit_mirt(
        responses,
        model="2PL",
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        item_names=item_labels.astype(str).tolist(),
        use_rust=use_rust,
    )

    # Person scores (theta + SE)
    score_result = mirt.fscores(
        fit_result,
        responses,
        method=score_method,
        person_ids=person_labels.tolist(),
        n_quadpts=n_quadpts,
    )

    # mirt may return pandas or polars depending on backend
    scores_df = score_result.to_dataframe()
    if hasattr(scores_df, "to_pandas"):  # polars DataFrame
        persons_df = scores_df.to_pandas()
    else:  # pandas DataFrame
        persons_df = scores_df.copy()

    # Keep person id as a normal column
    if "person" in persons_df.columns:
        persons_df = persons_df.rename(columns={"person": id_col})
    elif id_col not in persons_df.columns:
        persons_df = persons_df.reset_index()
        if "index" in persons_df.columns:
            persons_df = persons_df.rename(columns={"index": id_col})

    # Item parameters
    params = fit_result.model.parameters
    a = np.asarray(params["discrimination"]).ravel()
    b = np.asarray(params["difficulty"]).ravel()

    items_df = pd.DataFrame(
        {
            item_col: item_labels.tolist(),
            "a": a,
            "b": b,
        }
    )

    return persons_df, items_df, fit_result, score_result, responses


def fit_1pl_mirt(
    df,
    id_col="id",
    item_col="item",
    resp_col="resp",
    n_quadpts=21,
    max_iter=500,
    tol=1e-4,
    score_method="EAP",
    use_rust=False,
    verbose=False,
):
    """Fit a 1PL (Rasch) IRT model. Same interface as fit_2pl_mirt."""
    # Accept polars or pandas input
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    x = df[[id_col, item_col, resp_col]].dropna().copy()
    x[resp_col] = x[resp_col].astype(int)

    bad = set(x[resp_col].unique()) - {0, 1}
    if bad:
        raise ValueError(f"{resp_col} must be binary (0/1). Found: {sorted(bad)}")

    # Guard against duplicate person-item rows
    dup_mask = x.duplicated([id_col, item_col], keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        raise ValueError(
            f"Found {n_dup} duplicate person-item rows. "
            "Resolve duplicates before fitting mirt."
        )

    # Stable coding so joins/merges are predictable
    person_codes, person_labels = pd.factorize(x[id_col], sort=True)
    item_codes, item_labels = pd.factorize(x[item_col], sort=True)

    n_persons = len(person_labels)
    n_items = len(item_labels)

    # mirt expects 2D matrix: rows=persons, cols=items; missing coded as -1
    responses = np.full((n_persons, n_items), -1, dtype=np.int32)
    responses[person_codes, item_codes] = x[resp_col].to_numpy(dtype=np.int32)

    fit_result = mirt.fit_mirt(
        responses,
        model="1PL",
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        item_names=item_labels.astype(str).tolist(),
        use_rust=use_rust,
    )

    # Person scores (theta + SE)
    score_result = mirt.fscores(
        fit_result,
        responses,
        method=score_method,
        person_ids=person_labels.tolist(),
        n_quadpts=n_quadpts,
    )

    # mirt may return pandas or polars depending on backend
    scores_df = score_result.to_dataframe()
    if hasattr(scores_df, "to_pandas"):  # polars DataFrame
        persons_df = scores_df.to_pandas()
    else:  # pandas DataFrame
        persons_df = scores_df.copy()

    # Keep person id as a normal column
    if "person" in persons_df.columns:
        persons_df = persons_df.rename(columns={"person": id_col})
    elif id_col not in persons_df.columns:
        persons_df = persons_df.reset_index()
        if "index" in persons_df.columns:
            persons_df = persons_df.rename(columns={"index": id_col})

    # Item parameters (1PL only has difficulty parameter)
    params = fit_result.model.parameters
    b = np.asarray(params["difficulty"]).ravel()

    items_df = pd.DataFrame(
        {
            item_col: item_labels.tolist(),
            "b": b,
        }
    )

    return persons_df, items_df, fit_result, score_result, responses


def fit_bipartite_bt(
    df,
    id_col="id",
    item_col="item",
    resp_col="resp",
    max_iter=2000,
    tol=1e-8,
    eps=1e-12,
):
    """
    Fit a Bradley-Terry model on a person-item bipartite response graph.

    Encoding:
    - resp == 1: respondent wins against item
    - resp == 0: item wins against respondent
    """
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    x = df[[id_col, item_col, resp_col]].dropna().copy()
    x[resp_col] = x[resp_col].astype(int)

    bad = set(x[resp_col].unique()) - {0, 1}
    if bad:
        raise ValueError(f"{resp_col} must be binary (0/1). Found: {sorted(bad)}")

    # Aggregate duplicated person-item rows so repeated administrations contribute as multiple matches.
    agg = x.groupby([id_col, item_col], as_index=False).agg(
        wins_person=(resp_col, "sum"),
        n_obs=(resp_col, "count"),
    )
    agg["wins_item"] = agg["n_obs"] - agg["wins_person"]

    person_labels = pd.Index(sorted(agg[id_col].unique()))
    item_labels = pd.Index(sorted(agg[item_col].unique()))

    n_person = len(person_labels)
    n_item = len(item_labels)
    n_nodes = n_person + n_item

    person_idx = {k: i for i, k in enumerate(person_labels)}
    item_idx = {k: i + n_person for i, k in enumerate(item_labels)}

    i_idx = agg[id_col].map(person_idx).to_numpy(dtype=np.int64)
    j_idx = agg[item_col].map(item_idx).to_numpy(dtype=np.int64)
    w_ij = agg["wins_person"].to_numpy(dtype=np.float64)
    w_ji = agg["wins_item"].to_numpy(dtype=np.float64)

    # w_sum[k] is total wins of node k.
    w_sum = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(w_sum, i_idx, w_ij)
    np.add.at(w_sum, j_idx, w_ji)

    p = np.ones(n_nodes, dtype=np.float64)
    converged = False
    n_iter = 0

    for it in range(max_iter):
        n_iter = it + 1
        p_old = p.copy()

        denom_i = w_ij + w_ji
        frac_i = denom_i / (p[i_idx] + p[j_idx] + eps)

        d = np.zeros(n_nodes, dtype=np.float64)
        np.add.at(d, i_idx, frac_i)
        np.add.at(d, j_idx, frac_i)

        p = w_sum / np.maximum(d, eps)
        p = np.maximum(p, eps)

        # Identifiability: normalize geometric mean strength to 1.
        p /= np.exp(np.mean(np.log(p)))

        if np.max(np.abs(np.log(p) - np.log(p_old))) < tol:
            converged = True
            break

    scores = np.log(p)
    scores -= scores.mean()

    persons_df = pd.DataFrame(
        {
            id_col: person_labels.tolist(),
            "BT_Score_Bipartite": scores[:n_person],
            "BT_Strength_Bipartite": p[:n_person],
            "BT_Total_Wins": w_sum[:n_person],
        }
    )

    items_df = pd.DataFrame(
        {
            item_col: item_labels.tolist(),
            "BT_Score_Bipartite": scores[n_person:],
            "BT_Strength_Bipartite": p[n_person:],
            "BT_Total_Wins": w_sum[n_person:],
        }
    )

    diagnostics = {
        "converged": converged,
        "n_iter": n_iter,
        "n_iter_max": max_iter,
        "tol": tol,
        "n_person": n_person,
        "n_item": n_item,
        "n_edges": int(len(agg)),
    }

    return persons_df, items_df, diagnostics

def main():
    all_results_metadata = []
    all_tables_item_scores = []

    for table in irw_edtexts:

        # Make a directory for the current table's results
        table_dir = f"../results/{table}"
        os.makedirs(table_dir, exist_ok=True)

        # Get item response data for the current table
        df = irw_edtexts[table]
        df = pl.DataFrame(df)

        # Bipartite BT requires binary responses; compute once so item and respondent measures are always available.
        bt_persons = None
        bt_items = None
        bt_diagnostics = None

        # Filter out any items with >95% correct responses or <5% correct responses to avoid issues with perfect separation in the BT model and to ensure meaningful variability in item responses for the pairwise comparisons and IRT analyses.
        print(f"Filtering items in {table} based on response rates...")
        if 'resp' in df.columns:
            item_response_rates = df.groupby("item").agg(
                pl.col("resp").mean().alias("pct_correct"),
                pl.count("resp").alias("n_responses")
            )
            items_to_keep = item_response_rates.filter(
                (pl.col("pct_correct") <= 0.95) & 
                (pl.col("pct_correct") >= 0.05) &
                (pl.col("n_responses") >= 30)  # Ensure at least 30 responses per item for stable estimates
            )["item"].to_list()
            print(f"    Keeping {len(items_to_keep)} items out of {item_response_rates.shape[0]} based on response rates and minimum response count.")
            df = df.filter(pl.col("item").is_in(items_to_keep))

        # Get the number of unique values in df['item']
        unique_items = df['item'].n_unique()

        # Identify if the responses are binary or likert/ordinal based on the number of unique values in df['resp']
        unique_responses = df['resp'].n_unique()
        if unique_responses == 2:
            response_type = "binary"
        elif unique_responses > 2:
            response_type = "likert/ordinal"
        else:            
            response_type = "unknown"

        print(f"Analyzing {table}; Shape: {df.shape[0]} respondents, {df.shape[1]} columns; Unique Items: {unique_items}; Response Type: {response_type}")

        if response_type == "binary":
            print(f"    Fitting bipartite BT model for {table}...")
            bt_persons, bt_items, bt_diagnostics = fit_bipartite_bt(
                df,
                id_col="id",
                item_col="item",
                resp_col="resp",
                max_iter=2000,
                tol=1e-8,
            )
            print(f"    Bipartite BT diagnostics: {bt_diagnostics}")
            bt_persons.to_csv(f"../results/{table_dir}/{table}_bipartite_bt_persons.csv", index=False)
            bt_items.to_csv(f"../results/{table_dir}/{table}_bipartite_bt_items.csv", index=False)
        else:
            print(f"    Skipping bipartite BT for {table}: responses are {response_type}, not binary.")

        # Get item text
        text = irw.itemtext(table)
        text = pl.DataFrame(text)
        print(f"    Item text shape: {text.shape[0]} rows, {text.shape[1]} columns")

        # Aggregate item response data to the item level
        required_columns = ["item", "section_prompt", 
                            "item_text"]
        missing_columns = [col for col in required_columns if col not in text.columns]

        if missing_columns:
            print(f"    Warning: Missing columns in item text for {table}: {missing_columns}; skipping analysis for this table.")
            continue

        # Aggregate items based on the presence of option_text column, which is only present for some tables. If option_text is present, aggregate it by joining non-null, non-"NA" options with newlines, and include it in the combined text. If not present, just combine section_prompt and item_text.
        if "option_text" in text.columns:
            text_agg = text.group_by("item").agg(
                pl.col("section_prompt").first(),
                pl.col("item_text").first(),
                pl.col("option_text").filter(pl.col("option_text").is_not_null() & (pl.col("option_text") != "NA")).str.join("\n").alias("options")
            ).with_columns(
                pl.col("section_prompt").replace("NA", None),
                pl.col("item_text").replace("NA", None),
                pl.when(pl.col("options") == "").then(None).otherwise(pl.col("options")).alias("options")
            ).with_columns(
                pl.concat_str(
                    [
                        pl.concat_str([pl.col("section_prompt"), pl.col("item_text")], separator=": ", ignore_nulls=True),
                        pl.col("options")
                    ],
                    separator="\n",
                    ignore_nulls=True
                ).alias("combined_text")
            ).drop("options")
        else: 
            text_agg = text.group_by("item").agg(
                pl.col("section_prompt").first(),
                pl.col("item_text").first()
            ).with_columns(
                pl.col("section_prompt").replace("NA", None),
                pl.col("item_text").replace("NA", None)
            ).with_columns(
                pl.concat_str(
                    [
                        pl.col("section_prompt"), 
                        pl.col("item_text")
                    ],
                    separator=": ",
                    ignore_nulls=True
                ).alias("combined_text")
            )

        text_agg.write_csv(f"../results/{table_dir}/{table}_itemtext.csv")
        num_items_text = text_agg.shape[0]
        print(f"    Aggregated item text shape: {text_agg.shape[0]} rows, {text_agg.shape[1]} columns")
        
        if num_items_text != unique_items:
            print(f"    Warning: Number of unique items in response data ({unique_items}) does not match number of rows in aggregated item text ({num_items_text}) for {table}. This may indicate issues with the item text data or the aggregation process.")

        # Convert to pandas DataFrame for use with pairadigm
        text_agg_pd = text_agg.to_pandas()
        text_agg_pd['combined_text'] = text_agg_pd['combined_text'].astype(str)

        p = pdm.Pairadigm(
            data=text_agg_pd, 
            item_id_name='item',
            text_name='combined_text',
            cgcot_prompts=CGCoT_PROMPTS, 
            model_name=MODEL_NAMES,
            api_key=API_KEYS,
            base_url=BASE_URLS,
            target_concept='underlying level of the targeted latent trait, ability, symptom severity, or attitude', 
            save_dir=f"{table_dir}/pairadigm_results"
        )

        client_test_results = p.test_clients_connection()
        print(f"    Client test results: {client_test_results}")
        
        # If any values in the dict client_test_results are False, raise ValueError with a message indicating which clients failed the connection test
        if any(value is False for value in client_test_results.values()):
            failed_clients = [client for client, result in client_test_results.items() if result is False]
            raise ValueError(f"    Error: Clients {failed_clients} failed the connection test for {table}. Review pairadigm configuration.")
        
        # Run pairadigm analysis
        p.generate_breakdowns(max_workers=32)

        num_pairs_per_item = min(15, unique_items - 1)  # Generate up to 15 pairs per item, or fewer if there are not enough items

        print(f"    Generating pairings with num_pairs_per_item={num_pairs_per_item} for {table}")
        p.generate_pairings(num_pairs_per_item=num_pairs_per_item,
                            breakdowns=True)
        p.generate_pairwise_annotations(max_workers=32)

        # Print annotator quality and save the dataframes to the table directory
        irr_reliabilities = p.irr()
        ds_reliabilities = p.dawid_skene_annotator_ranking(random_seed=1234)

        irr_reliabilities.to_csv(f"../results/{table_dir}/{table}_irr_reliabilities.csv", index=False)
        ds_reliabilities.to_csv(f"../results/{table_dir}/{table}_ds_reliabilities.csv", index=False)

        # Get and concatenate scored items for all models
        all_scores = []
        index = 0
        for model in MODEL_NAMES:
            print(f"    Scoring items using decisions from {model} for {table}...")            
            scored_df = p.score_items(
                normalization_scale=(-3, 3), 
                decision_col=f"decision_{model}"
            )  
            if index == 0:
                all_scores = scored_df.copy()
            else:
                scored_df = scored_df[[
                    'item', 
                    f'Bradley_Terry_Score_{model}', f'Bradley_Terry_SE_{model}']]
                all_scores = all_scores.merge(scored_df, on='item', how='left')
            index += 1
            
        print(f"    Completed scoring for {table}. Saving to pairadigm object...")
        p.scored_df = all_scores
        p.save()
        print(f"    Saved final pairadigm object with scored items for {table}.")

        ### IRT ANALYSIS ###

        # Only fit IRT models when responses are binary.
        if response_type == "binary":
            print(f"    Fitting 2PL IRT model for {table}...")
            persons_2pl, items_2pl, fit_res_2pl, score_res_2pl, resp_matrix = fit_2pl_mirt(
                df,
                id_col="id",
                item_col="item",
                resp_col="resp",
                n_quadpts=21,
                max_iter=500,
                tol=1e-4,
                score_method="EAP",
                use_rust=False
            )

            print(fit_res_2pl.summary())
            print(items_2pl.head())
            print(persons_2pl.head())

            # Fit 1PL model
            print(f"    Fitting 1PL IRT model for {table}...")
            persons_1pl, items_1pl, fit_res_1pl, score_res_1pl, _ = fit_1pl_mirt(
                df,
                id_col="id",
                item_col="item",
                resp_col="resp",
                n_quadpts=21,
                max_iter=500,
                tol=1e-4,
                score_method="EAP",
                use_rust=False
            )

            print(fit_res_1pl.summary())
            print(items_1pl.head())
            print(persons_1pl.head())

            # Rename 1PL columns to distinguish from 2PL
            items_1pl_renamed = items_1pl.rename(columns={"b": "b_1PL"})
            
            # Identify theta and SE columns in persons_1pl (typically first two columns after id)
            person_cols = list(persons_1pl.columns)
            person_cols.remove("id")
            theta_col_1pl = person_cols[0] if len(person_cols) > 0 else "theta"
            se_col_1pl = person_cols[1] if len(person_cols) > 1 else "se"
            
            persons_1pl_renamed = persons_1pl.rename(
                columns={
                    theta_col_1pl: "theta_1PL",
                    se_col_1pl: "se_1PL"
                }
            )

            # Identify theta and SE columns in persons_2pl
            person_cols_2pl = list(persons_2pl.columns)
            person_cols_2pl.remove("id")
            theta_col_2pl = person_cols_2pl[0] if len(person_cols_2pl) > 0 else "theta"
            se_col_2pl = person_cols_2pl[1] if len(person_cols_2pl) > 1 else "se"
            
            persons_2pl_renamed = persons_2pl.rename(
                columns={
                    theta_col_2pl: "theta_2PL",
                    se_col_2pl: "se_2PL"
                }
            )
            
            # Also rename 2PL item parameters for clarity
            items_2pl_renamed = items_2pl.rename(columns={"a": "a_2PL", "b": "b_2PL"})

            # Combine 2PL and 1PL person estimates
            persons = persons_2pl_renamed.merge(persons_1pl_renamed[["id", "theta_1PL", "se_1PL"]], on="id", how="left")
            
            # Combine 2PL and 1PL item parameters
            items = items_2pl_renamed.merge(items_1pl_renamed, on='item', how='left')

            # Save combined IRT results
            items.to_csv(f"../results/{table_dir}/{table}_irt_items.csv", index=False)
            persons.to_csv(f"../results/{table_dir}/{table}_irt_persons.csv", index=False)
            
            all_item_scores = p.scored_df.merge(items, on='item').copy()
            if bt_items is not None:
                all_item_scores = all_item_scores.merge(
                    bt_items.rename(
                        columns={
                            "BT_Score_Bipartite": "BT_Item_Score_Bipartite",
                            "BT_Strength_Bipartite": "BT_Item_Strength_Bipartite",
                            "BT_Total_Wins": "BT_Item_Total_Wins_Bipartite",
                        }
                    ),
                    on='item',
                    how='left'
                )

            if bt_persons is not None:
                bt_persons_with_irt = bt_persons.merge(persons, on="id", how="left")
                bt_persons_with_irt.to_csv(
                    f"../results/{table_dir}/{table}_bipartite_bt_persons_with_irt.csv",
                    index=False,
                )
            irt_item_params_summary = items.describe().to_dict()
            irt_person_scores_summary = persons.describe().to_dict()
        else:
            print(f"    Skipping IRT models for {table}: responses are {response_type}, not binary.")
            all_item_scores = p.scored_df.copy()
            irt_item_params_summary = None
            irt_person_scores_summary = None

        print(f"    Completed analysis for {table}. Saving results...")
        
        all_item_scores['table'] = table
        all_item_scores.to_csv(f"../results/{table_dir}/{table}_all_item_scores.csv", index=False)
        
        try:
            all_results_metadata.to_csv(f"../results/{table_dir}/{table}_results_metadata.csv", index=False)
        except Exception as e:
            print(f"Error saving metadata for {table}: {e}")
            # Try saving metadata as a JSON file instead
            try:
                import json
                with open(f"../results/{table_dir}/{table}_results_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(all_results_metadata, f, indent=4)
            except Exception as e:
                print(f"Error saving metadata as JSON for {table}: {e}")
                print("Metadata will be included in the aggregated results across all tables instead.")

        all_results_metadata.append({
            "table": table,
            "client_test_results": client_test_results,
            "num_respondents": df.shape[0],
            "num_items": unique_items,
            "num_pairs_per_item": num_pairs_per_item,
            "bipartite_bt": bt_diagnostics,
            "irt_item_params": irt_item_params_summary,
            "irt_person_scores": irt_person_scores_summary
        })

        all_tables_item_scores.append(all_item_scores)
        print(f"    Saved results for {table}.")
    

    # After processing all tables, save the aggregated metadata and item scores
    print("Saving results across all tables...")
    try:
        metadata_df = pd.DataFrame(all_results_metadata)
        metadata_df.to_csv("../results/all_tables_results_metadata.csv", index=False)
        
    except Exception as e:
        print(f"Error saving aggregated results: {e}")
        print("Saving individual table results instead.")
        for i, metadata in enumerate(all_results_metadata):
            table_name = metadata['table']
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_csv(f"../results/{table_name}_metadata.csv", index=False)
    
    all_item_scores_df = pd.concat(all_tables_item_scores, ignore_index=True)
    all_item_scores_df.to_csv("../results/all_tables_item_scores.csv", index=False)
    print("Saved results across all tables.")

if __name__ == "__main__":  
    log_path = get_log_path()
    with log_path.open("a", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print(f"Logging output to: {log_path}")
            main()