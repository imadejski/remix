import argparse
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm


def read_data(cosine_path, labels_path, label_type="auto"):
    """
    Reads in csvs with cosine similarity scores, ground-truth labels, and split
    """
    cosine_df = pd.read_csv(cosine_path)
    labels_df = pd.read_csv(labels_path)

    if label_type == "labeled":
        split_df = None
    elif label_type == "combined":
        split_df = pd.read_csv("/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv")
    else:  # auto, raw, convirt
        split_df = pd.read_csv("/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv")

    return cosine_df, labels_df, split_df


def filter_split_type_data(split_df, labels_df, split_type, label_type="auto"):
    """
    Filters split df so that only specified split type exists
    """
    if label_type == "labeled":
        return labels_df  # No filtering needed for labeled test set

    split_type_df = split_df[split_df["split"] == split_type]
    split_type_study_ids = split_type_df["study_id"].unique()
    split_type_labels_df = labels_df[labels_df["study_id"].isin(split_type_study_ids)]

    return split_type_labels_df


def transform_cosine_df(cosine_df, labels, embedding_types, label_type="auto"):
    """
    Transforms cosine similarity df for processing by accuracy function
    """
    data = []
    for label in labels:
        if label_type == "raw":
            column_name = f"{label} cosine_similarity"
            if column_name in cosine_df.columns:
                temp_df = cosine_df[
                    ["subject_id", "study_id", "dicom_id", column_name]
                ].copy()
                temp_df.rename(columns={column_name: "cosine_similarity"}, inplace=True)
                temp_df["label"] = label
                temp_df["cosine_similarity"] = pd.to_numeric(
                    temp_df["cosine_similarity"], errors="coerce"
                )
                data.append(temp_df)
            else:
                raise ValueError(f"Column {column_name} does not exist in cosine_df")
        else:
            for emb in embedding_types:
                column_name = f"{label} {emb}"
                if column_name in cosine_df.columns:
                    temp_df = cosine_df[
                        ["subject_id", "study_id", "dicom_id", column_name]
                    ].copy()
                    temp_df.rename(
                        columns={column_name: "cosine_similarity"}, inplace=True
                    )
                    temp_df["label"] = label
                    temp_df["embedding_type"] = emb
                    temp_df["cosine_similarity"] = pd.to_numeric(
                        temp_df["cosine_similarity"], errors="coerce"
                    )
                    data.append(temp_df)
                else:
                    raise ValueError(
                        f"Column {column_name} does not exist in cosine_df"
                    )

    combined_df = pd.concat(data, ignore_index=True)
    return combined_df


def count_positive_cases(labels, split_type_labels_df):
    """
    Finds total number of positive cases, represented by 1, in df w labels
    """
    n_counts = {}
    for label in labels:
        label_positives = split_type_labels_df[split_type_labels_df[label] == 1]
        unique_positive_study_ids = label_positives["study_id"].unique()
        n_counts[label] = len(unique_positive_study_ids)

    return n_counts


def aggregate_cosine(df, method: Literal["max", "mean"]):
    """
    Aggregates cosine similarity scores by taking the max or average
    """
    if method == "max":
        aggregated_df = (
            df.groupby(["study_id", "label", "embedding_type"])["cosine_similarity"]
            .max()
            .reset_index()
        )
    elif method == "mean":
        aggregated_df = (
            df.groupby(["study_id", "label", "embedding_type"])["cosine_similarity"]
            .mean()
            .reset_index()
        )
    else:
        raise ValueError("Aggregation method must be 'max' or 'mean'")

    # Debugging: Print the size of the aggregated DataFrame
    print(f"Aggregated DataFrame size for method '{method}': {aggregated_df.shape}")
    return aggregated_df


def calculate_dcg(relevance_scores, k=None):
    """
    Calculate Discounted Cumulative Gain (DCG) for a list of relevance scores.

    Args:
        relevance_scores: List of relevance scores (1 for relevant, 0 for non-relevant)
        k: If specified, only consider the first k items

    Returns:
        DCG score
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]

    if len(relevance_scores) == 0:
        return 0.0

    dcg = relevance_scores[0]  # First item has discount factor of 1
    for i in range(1, len(relevance_scores)):
        dcg += relevance_scores[i] / np.log2(i + 1)

    return dcg


def calculate_ndcg(relevance_scores, k=None):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).

    Args:
        relevance_scores: List of relevance scores (1 for relevant, 0 for non-relevant)
        k: If specified, only consider the first k items

    Returns:
        NDCG score
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]

    dcg = calculate_dcg(relevance_scores, k)

    # Calculate ideal DCG (IDCG) by sorting relevance scores in descending order
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_relevance, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_metrics(
    labels,
    n_counts,
    cosine_max,
    cosine_mean,
    split_type_labels_df,
    k_values,
):
    """
    Calculate all metrics: accuracy@n, top-k accuracy, DCG@n/NDCG@n, and DCG@k/NDCG@k
    """
    results = {}
    top_k_results = {}
    dcg_results = {}

    for label in labels:
        n = n_counts[label]
        label_results = {}
        label_results_k = {}
        label_results_dcg = {}

        filtered_embedding_types = [
            "average_cosine_similarity",
            "max_cosine_similarity",
        ]

        for emb in filtered_embedding_types:
            max_df = cosine_max[
                (cosine_max["label"] == label) & (cosine_max["embedding_type"] == emb)
            ]

            mean_df = cosine_mean[
                (cosine_mean["label"] == label) & (cosine_mean["embedding_type"] == emb)
            ]

            # Check for empty DataFrames
            if max_df.empty or mean_df.empty:
                print(
                    f"Warning: No data available for label '{label}' and embedding '{emb}'. Skipping."
                )
                # Set all metrics to NaN for this combination
                label_results[f"{emb}_max_accuracy"] = np.nan
                label_results[f"{emb}_mean_accuracy"] = np.nan
                label_results[f"{emb}_max_dcg_n"] = np.nan
                label_results[f"{emb}_mean_dcg_n"] = np.nan
                label_results[f"{emb}_max_ndcg_n"] = np.nan
                label_results[f"{emb}_mean_ndcg_n"] = np.nan

                for k in k_values:
                    label_results_k[f"{emb}_max_accuracy_top_{k}"] = np.nan
                    label_results_k[f"{emb}_mean_accuracy_top_{k}"] = np.nan
                    label_results_dcg[f"{emb}_max_dcg_{k}"] = np.nan
                    label_results_dcg[f"{emb}_mean_dcg_{k}"] = np.nan
                    label_results_dcg[f"{emb}_max_ndcg_{k}"] = np.nan
                    label_results_dcg[f"{emb}_mean_ndcg_{k}"] = np.nan
                continue

            # Get label positives
            label_positives = split_type_labels_df[split_type_labels_df[label] == 1][
                "study_id"
            ]

            # Sort by cosine similarity (descending)
            max_df_sorted = max_df.sort_values("cosine_similarity", ascending=False)
            mean_df_sorted = mean_df.sort_values("cosine_similarity", ascending=False)

            # Check if we have enough positive cases for meaningful metrics
            num_positives = len(label_positives)

            # Adjust n to be at most the number of available rows
            n_adjusted = min(n, len(max_df_sorted), len(mean_df_sorted))

            # Calculate top_n accuracy and DCG@n (precision at n positive cases)
            if n_adjusted > 0 and num_positives > 0:
                top_n_max = max_df_sorted.head(n_adjusted)["study_id"]
                top_n_mean = mean_df_sorted.head(n_adjusted)["study_id"]

                label_results[f"{emb}_max_accuracy"] = (
                    top_n_max.isin(label_positives).sum() / n_adjusted
                )
                label_results[f"{emb}_mean_accuracy"] = (
                    top_n_mean.isin(label_positives).sum() / n_adjusted
                )

                # DCG@n and NDCG@n calculations
                # Create relevance vectors for top-n items
                max_relevance_n = [
                    1 if study_id in label_positives.values else 0
                    for study_id in top_n_max
                ]
                mean_relevance_n = [
                    1 if study_id in label_positives.values else 0
                    for study_id in top_n_mean
                ]

                # Calculate DCG@n and NDCG@n
                label_results[f"{emb}_max_dcg_n"] = calculate_dcg(max_relevance_n)
                label_results[f"{emb}_mean_dcg_n"] = calculate_dcg(mean_relevance_n)
                label_results[f"{emb}_max_ndcg_n"] = calculate_ndcg(max_relevance_n)
                label_results[f"{emb}_mean_ndcg_n"] = calculate_ndcg(mean_relevance_n)
            else:
                # Set to NaN if no positive cases or no data
                if num_positives == 0:
                    print(
                        f"Warning: No positive cases for label '{label}'. Setting accuracy@n and DCG@n to NaN."
                    )
                label_results[f"{emb}_max_accuracy"] = (
                    np.nan if num_positives == 0 else 0
                )
                label_results[f"{emb}_mean_accuracy"] = (
                    np.nan if num_positives == 0 else 0
                )
                label_results[f"{emb}_max_dcg_n"] = np.nan if num_positives == 0 else 0
                label_results[f"{emb}_mean_dcg_n"] = np.nan if num_positives == 0 else 0
                label_results[f"{emb}_max_ndcg_n"] = np.nan if num_positives == 0 else 0
                label_results[f"{emb}_mean_ndcg_n"] = (
                    np.nan if num_positives == 0 else 0
                )

            # Calculate top_k metrics and DCG@k for each k in k_values
            for k in k_values:
                # Check if we have enough positive cases for meaningful top-k metrics
                # Require at least k positive cases for meaningful metrics
                if (
                    num_positives >= k
                    and len(max_df_sorted) >= k
                    and len(mean_df_sorted) >= k
                ):
                    # Top-k accuracy - always use exactly k samples
                    top_k_max = max_df_sorted.head(k)["study_id"]
                    top_k_mean = mean_df_sorted.head(k)["study_id"]

                    label_results_k[f"{emb}_max_accuracy_top_{k}"] = (
                        top_k_max.isin(label_positives).sum() / k
                    )
                    label_results_k[f"{emb}_mean_accuracy_top_{k}"] = (
                        top_k_mean.isin(label_positives).sum() / k
                    )

                    # DCG@k and NDCG@k calculations
                    # Create relevance vectors for top-k items
                    max_relevance_k = [
                        1 if study_id in label_positives.values else 0
                        for study_id in top_k_max
                    ]
                    mean_relevance_k = [
                        1 if study_id in label_positives.values else 0
                        for study_id in top_k_mean
                    ]

                    # Calculate DCG@k and NDCG@k
                    label_results_dcg[f"{emb}_max_dcg_{k}"] = calculate_dcg(
                        max_relevance_k
                    )
                    label_results_dcg[f"{emb}_mean_dcg_{k}"] = calculate_dcg(
                        mean_relevance_k
                    )
                    label_results_dcg[f"{emb}_max_ndcg_{k}"] = calculate_ndcg(
                        max_relevance_k
                    )
                    label_results_dcg[f"{emb}_mean_ndcg_{k}"] = calculate_ndcg(
                        mean_relevance_k
                    )
                else:
                    # Set to NaN if insufficient positive cases or total samples
                    if num_positives < k:
                        print(
                            f"Warning: Not enough positive cases ({num_positives} < {k}) for meaningful top-{k} metrics for label '{label}'. Setting to NaN."
                        )
                    elif len(max_df_sorted) < k or len(mean_df_sorted) < k:
                        print(
                            f"Warning: Not enough total samples (max:{len(max_df_sorted)}, mean:{len(mean_df_sorted)} < {k}) for top-{k} metrics for label '{label}'. Setting to NaN."
                        )

                    label_results_k[f"{emb}_max_accuracy_top_{k}"] = np.nan
                    label_results_k[f"{emb}_mean_accuracy_top_{k}"] = np.nan
                    label_results_dcg[f"{emb}_max_dcg_{k}"] = np.nan
                    label_results_dcg[f"{emb}_mean_dcg_{k}"] = np.nan
                    label_results_dcg[f"{emb}_max_ndcg_{k}"] = np.nan
                    label_results_dcg[f"{emb}_mean_ndcg_{k}"] = np.nan

        results[label] = label_results
        top_k_results[label] = label_results_k
        dcg_results[label] = label_results_dcg

    return results, top_k_results, dcg_results


def save_results(results_df, output_path):
    """
    Save results in output df.
    """
    results_df.to_csv(output_path, index=False)


def calculate_confidence_intervals(data):
    """
    Calculate mean and 95% confidence intervals for resampling data.
    """
    mean = data.mean()
    se = data.std(ddof=1) / np.sqrt(len(data))  # Standard error
    ci_lower = mean - 1.96 * se  # Lower bound of the 95% CI
    ci_upper = mean + 1.96 * se  # Upper bound of the 95% CI
    return mean, ci_lower, ci_upper


def resample_and_calculate_metrics(
    labels,
    n_counts,
    cosine_max,
    cosine_mean,
    split_type_labels_df,
    k_values,
    num_iterations=1000,
):
    """
    Perform resampling and calculate all metrics with confidence intervals.
    """
    all_results = []

    # Determine the unique study IDs from the combined cosine_max data
    unique_study_ids = cosine_max["study_id"].unique()

    # Ensure there are enough study IDs to sample from
    if len(unique_study_ids) == 0:
        print("Warning: No unique study IDs available for resampling. Exiting.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Unique study IDs available for resampling: {len(unique_study_ids)}")

    for i in range(num_iterations):
        # Determine the sample size as half of the unique study IDs
        sample_size = max(1, (len(unique_study_ids) // 2))
        if i == 0:  # Only print once to avoid spam
            print(f"Sample size: {sample_size}")

        # Randomly sample study IDs
        sampled_ids = np.random.choice(
            unique_study_ids, size=sample_size, replace=False
        )

        # Resample the data for the current subset
        sampled_cosine_max = cosine_max[cosine_max["study_id"].isin(sampled_ids)]
        sampled_cosine_mean = cosine_mean[cosine_mean["study_id"].isin(sampled_ids)]
        sampled_split_type_labels_df = split_type_labels_df[
            split_type_labels_df["study_id"].isin(sampled_ids)
        ]

        # Recalculate the number of positive cases in the sampled data
        n_counts_sampled = count_positive_cases(labels, sampled_split_type_labels_df)

        # Ensure that sampled data is not empty
        if sampled_cosine_max.empty or sampled_cosine_mean.empty:
            print(
                f"Warning: No valid data for resampled set in iteration {i}. Skipping this iteration."
            )
            continue

        # Calculate all metrics for the resample
        results, top_k_results, dcg_results = calculate_metrics(
            labels,
            n_counts_sampled,
            sampled_cosine_max,
            sampled_cosine_mean,
            sampled_split_type_labels_df,
            k_values,
        )

        if i % 100 == 0:  # Print progress every 100 iterations
            print(f"Iteration {i} calculated")

        # Store all results for each metric type
        all_metric_dicts = [results, top_k_results, dcg_results]

        for metric_dict in all_metric_dicts:
            for label in metric_dict:
                for key, value in metric_dict[label].items():
                    all_results.append(
                        {
                            "iteration": i,
                            "label": label,
                            "metric": key,
                            "value": value,
                        }
                    )

    # Convert all results into a DataFrame
    all_results_df = pd.DataFrame(all_results)

    # Calculate mean and confidence intervals for all metrics
    mean_ci_results = []

    # Get all unique metrics
    unique_metrics = all_results_df["metric"].unique()

    for label in labels:
        for metric in unique_metrics:
            subset = all_results_df[
                (all_results_df["label"] == label)
                & (all_results_df["metric"] == metric)
            ]

            if not subset.empty:
                subset_values = pd.to_numeric(subset["value"], errors="coerce")
                # Remove NaN values for calculation
                subset_values = subset_values.dropna()

                if len(subset_values) > 0:
                    mean, ci_lower, ci_upper = calculate_confidence_intervals(
                        subset_values
                    )
                    mean_ci_results.append(
                        {
                            "label": label,
                            "metric": metric,
                            "mean": mean,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "n_samples": len(subset_values),
                        }
                    )

    mean_ci_results_df = pd.DataFrame(mean_ci_results)

    return mean_ci_results_df, all_results_df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate accuracy, top-k, DCG, and NDCG metrics for cosine similarity data."
    )
    parser.add_argument(
        "-c",
        "--cosine-path",
        type=str,
        required=True,
        help="Path to the cosine similarity CSV file.",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        required=True,
        help="Path to the ground-truth labels CSV file.",
    )
    parser.add_argument(
        "-o",
        "--output-results-path",
        type=str,
        required=True,
        help="Path to save the accuracy results CSV file.",
    )
    parser.add_argument(
        "-t",
        "--output-top-k-results-path",
        type=str,
        required=True,
        help="Path to save the top-k accuracy results CSV file.",
    )
    parser.add_argument(
        "-d",
        "--output-dcg-results-path",
        type=str,
        help="Path to save the DCG/NDCG results CSV file.",
    )
    parser.add_argument(
        "-s",
        "--split-type",
        type=str,
        required=True,
        help="Data split type (e.g., 'validate', 'train').",
    )
    parser.add_argument(
        "-n",
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of resampling iterations. Default is 1000.",
    )
    parser.add_argument(
        "-r",
        "--resampling",
        action="store_true",
        help="Flag to enable resampling for metric calculation.",
    )
    parser.add_argument(
        "-l",
        "--label-type",
        type=str,
        choices=["auto", "labeled", "raw", "convirt", "combined"],
        default="auto",
        help="Type of labels to use: auto (CheXpert), labeled (test set), raw, convirt, or combined reports.",
    )

    args = parser.parse_args()

    # Define labels based on label type
    if args.label_type == "convirt":
        labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Edema",
            "Fracture",
            "No Finding",
            "Pleural Effusion",
            "Pneumonia",
            "Pneumothorax",
        ]
    elif args.label_type == "labeled":
        labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Airspace Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]
    else:  # auto, raw, combined
        labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]

    # Define embedding types based on label type
    if args.label_type == "convirt":
        embedding_types = [
            "cosine_similarity_1",
            "cosine_similarity_2",
            "cosine_similarity_3",
            "cosine_similarity_4",
            "cosine_similarity_5",
            "average_cosine_similarity",
            "max_cosine_similarity",
        ]
    elif args.label_type == "raw":
        embedding_types = ["cosine_similarity"]  # Not used, but kept for consistency
    else:  # auto, labeled, combined
        embedding_types = [
            "cosine_similarity_1",
            "cosine_similarity_2",
            "cosine_similarity_3",
            "cosine_similarity_4",
            "average_cosine_similarity",
            "max_cosine_similarity",
        ]

    # Read and process data
    cosine_df, labels_df, split_df = read_data(
        args.cosine_path, args.labels_path, args.label_type
    )
    split_type_labels_df = filter_split_type_data(
        split_df, labels_df, args.split_type, args.label_type
    )
    cosine_df_transformed = transform_cosine_df(
        cosine_df, labels, embedding_types, args.label_type
    )
    n_counts = count_positive_cases(labels, split_type_labels_df)
    cosine_max = aggregate_cosine(cosine_df_transformed, "max")
    cosine_mean = aggregate_cosine(cosine_df_transformed, "mean")

    # Define k values for top_k calculations
    k_values = [5, 10, 20]

    if args.resampling:
        # Perform resampling and calculate all metrics
        mean_ci_results_df, all_results_df = resample_and_calculate_metrics(
            labels,
            n_counts,
            cosine_max,
            cosine_mean,
            split_type_labels_df,
            k_values,
            args.num_iterations,
        )

        # Save mean and CI results with label type in filename
        save_results(
            mean_ci_results_df,
            args.output_results_path.replace(
                ".csv", f"_resampling_{args.label_type}.csv"
            ),
        )

        # Save all resampling results in separate files
        save_results(
            all_results_df,
            args.output_results_path.replace(
                ".csv", f"_all_resampling_{args.label_type}.csv"
            ),
        )

        print(f"All metrics with resampling results saved for {args.label_type} labels")

    else:
        # Calculate all metrics without resampling
        results, top_k_results, dcg_results = calculate_metrics(
            labels,
            n_counts,
            cosine_max,
            cosine_mean,
            split_type_labels_df,
            k_values,
        )

        # Convert results to DataFrame and save
        results_df = (
            pd.DataFrame(results)
            .T.reset_index()
            .melt(id_vars="index", var_name="metric", value_name="value")
        )
        results_df.columns = ["label", "metric", "value"]
        save_results(results_df, args.output_results_path)

        # Save top_k_results
        top_k_results_df = (
            pd.DataFrame(top_k_results)
            .T.reset_index()
            .melt(id_vars="index", var_name="metric", value_name="value")
        )
        top_k_results_df.columns = ["label", "metric", "value"]
        save_results(top_k_results_df, args.output_top_k_results_path)

        # Save DCG results if path is provided
        if args.output_dcg_results_path:
            dcg_results_df = (
                pd.DataFrame(dcg_results)
                .T.reset_index()
                .melt(id_vars="index", var_name="metric", value_name="value")
            )
            dcg_results_df.columns = ["label", "metric", "value"]
            save_results(dcg_results_df, args.output_dcg_results_path)

        print(f"All metrics saved for {args.label_type} labels")


if __name__ == "__main__":
    main()
