#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import yaml

from usal_echo import usr_dir
from usal_echo.d00_utils.log_utils import setup_logging
from usal_echo.d00_utils.db_utils import dbReadWriteMeasurement
from usal_echo.d06_visualisation.confusion_matrix import plot_confusion_matrix

logger = setup_logging(__name__, __name__)


def evaluate_meas(folder):
    # Get ground truth and calculated measurements for files in folder.
    io_measurement = dbReadWriteMeasurement()
    ground_truths_df = io_measurement.get_table("ground_truths")
    ground_truths_df = ground_truths_df.drop(columns=["ground_truth_id"])
    calculations_df = io_measurement.get_table("calculations")
    calculations_df = calculations_df.drop(columns=["calculation_id", "date_run"])

    with open(os.path.join(usr_dir,"conf","path_parameters.yml")) as f:
        paths = yaml.safe_load(f)
    path = os.path.expanduser(paths["dcm_dir"])
    file_names = [fn.split(".")[0] for fn in os.listdir(f"{path}/{folder}/raw")]
    ground_truths_df = ground_truths_df[ground_truths_df["file_name"].isin(file_names)]
    calculations_df = calculations_df[calculations_df["file_name"].isin(file_names)]
    # Combine ground truth and calculated measurements for evaluation.
    cols = ["study_id", "instance_id", "file_name", "measurement_name"]
    merge_df = ground_truths_df.merge(
        calculations_df, on=cols, suffixes=["_gt", "_calc"]
    )
    merge_df = merge_df[merge_df["measurement_value_calc"] != ""]
    merge_df = merge_df.drop_duplicates(keep="last")

    # Evaluate volumes and ejection fractions with absolute/relative differences.
    numeric_df = merge_df[merge_df["measurement_name"] != "recommendation"].copy()
    numeric_df["measurement_value_gt"] = pd.to_numeric(
        numeric_df["measurement_value_gt"]
    )
    numeric_df["measurement_value_calc"] = pd.to_numeric(
        numeric_df["measurement_value_calc"]
    )

    abs_diff_df = numeric_df[cols].copy()
    abs_diff_df["score_type"] = "absolute_difference"
    abs_diff_df["score_value"] = np.abs(
        numeric_df["measurement_value_gt"] - numeric_df["measurement_value_calc"]
    )
    abs_diff_df.head()

    rel_diff_df = abs_diff_df.copy()
    rel_diff_df["score_type"] = rel_diff_df["score_type"].str.replace(
        "absolute", "relative"
    )
    rel_diff_df["score_value"] /= numeric_df["measurement_value_gt"].astype(float)
    rel_diff_df["score_value"] *= 100

    # Evaluate recommendations with accuracies.
    rec_df = merge_df[merge_df["measurement_name"] == "recommendation"].copy()

    acc_df = rec_df[cols].copy()
    acc_df["score_type"] = "accuracy"
    acc_df["score_value"] = (
        rec_df["measurement_value_gt"] == rec_df["measurement_value_calc"]
    ).astype(int)

    # Write evaluations to schema.
    evaluations_df = abs_diff_df.append(rel_diff_df).append(acc_df)
    # Add serial id.
    old_evaluations_df = io_measurement.get_table("evaluations")
    start = len(old_evaluations_df)
    evaluation_id = pd.Series(start + evaluations_df.index)
    evaluations_df.insert(0, "evaluation_id", evaluation_id)
    all_evaluations_df = old_evaluations_df.append(evaluations_df)
    io_measurement.save_to_db(all_evaluations_df, "evaluations")

    # Plot confusion matrix for recommendations, with classes in given order.
    classes = ["normal", "greyzone", "abnormal"]
    class_to_index = {v: k for k, v in enumerate(classes)}

    y_true = rec_df["measurement_value_gt"].map(class_to_index)
    y_calc = rec_df["measurement_value_calc"].map(class_to_index)

    fig, ax = plot_confusion_matrix(
        y_true, y_calc, classes=classes, title="Confusion matrix, without normalization"
    )

    fig.savefig(f"results/cm_{folder}", bbox_inches="tight")
    fig.savefig(f"results/cm_{folder}.pdf", format="pdf", bbox_inches="tight")

    fig, ax = plot_confusion_matrix(
        y_true,
        y_calc,
        classes=classes,
        normalize=True,
        title="Normalized confusion matrix",
    )

    fig.savefig(f"results/norm_cm_{folder}", bbox_inches="tight")
    fig.savefig(f"results/norm_cm_{folder}.pdf", format="pdf", bbox_inches="tight")

    # Produce dataframe like Table 2 in Zhang et al:
    # "Comparison Between Fully Automated and Manual Measurements Derived From 2-Dimensional Echocardiography"
    measurement_names = ["VTD(MDD-ps4)", "VTS(MDD-ps4)", "FE(MDD-ps4)"]

    measurements = [
        "Left ventricular diastolic volume, mL",
        "Left ventricular systolic volume, mL",
        "Left ventricular ejection fraction, %",
    ]

    meas_counts = numeric_df["measurement_name"].value_counts()
    numbers = [meas_counts[measurement_name] for measurement_name in measurement_names]

    value_describes = [
        numeric_df[numeric_df["measurement_name"] == measurement_name][
            "measurement_value_gt"
        ]
        .describe()
        .astype(int)
        for measurement_name in measurement_names
    ]
    median_iqr_strs = [
        f"{describe['50%']} ({describe['25%']}-{describe['75%']})"
        for describe in value_describes
    ]

    abs_diff_describes = [
        abs_diff_df[abs_diff_df["measurement_name"] == measurement_name]["score_value"]
        .describe()
        .astype(int)
        for measurement_name in measurement_names
    ]
    rel_diff_describes = [
        rel_diff_df[rel_diff_df["measurement_name"] == measurement_name]["score_value"]
        .describe()
        .astype(int)
        for measurement_name in measurement_names
    ]
    abs_dev_median_strs = [
        f"{abs_diff_describe['50%']} ({rel_diff_describe['50%']})"
        for abs_diff_describe, rel_diff_describe in zip(
            abs_diff_describes, rel_diff_describes
        )
    ]

    df = pd.DataFrame.from_dict(
        {
            "Measurements": measurements,
            "Number": numbers,
            "Median Value (IQR)": median_iqr_strs,
            "Median Absolute Deviation (% of Manual)": abs_dev_median_strs,
        }
    )
    df.to_csv(f"results/measurement_comparison_{folder}.csv", index=False)
    logger.info("Successfully evaluated measurements")
    return df
