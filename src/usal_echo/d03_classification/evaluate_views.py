#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import numpy as np

from sklearn.metrics import multilabel_confusion_matrix

from usal_echo.d00_utils.db_utils import dbReadWriteViews, dbReadWriteClassification
from usal_echo.d00_utils.log_utils import setup_logging

logger = setup_logging(__name__, __name__)


def _groundtruth_views():

    # Get ground truth labels via views.instances_w_labels table
    io_views = dbReadWriteViews()
    io_class = dbReadWriteClassification()

    groundtruth = io_views.get_table("instances_w_labels")
    groundtruth.rename(
        columns={"filename": "file_name", "studyidk": "study_id", "view": "view_true"},
        inplace=True,
    )
    groundtruth["file_name"] = (
        "a_"
        + groundtruth["study_id"].astype(str)
        + "_"
        + groundtruth["file_name"].astype(str)
    )
    groundtruth.drop(columns=["sopinstanceuid", "instanceidk"], inplace=True)
    predictions = io_class.get_table("predictions")

    # Merge tables df_new and labels_df
    predict_truth = predictions.merge(groundtruth, on=["file_name", "study_id"])

    return predict_truth


def evaluate_view_map(img_dir, model_name, date_run, view_mapping, study_filter=None):

    predict_truth = _groundtruth_views()

    df = predict_truth.loc[
        (predict_truth["img_dir"] == img_dir)
        & (predict_truth["model_name"] == model_name)
        & (pd.to_datetime(predict_truth["date_run"]).dt.date == date_run),
        :,
    ]

    if type(study_filter) == dict:
        df = df[df["study_id"].isin(list(study_filter.values())[0])]
        study_filter = list(study_filter.keys())[0]

    mcm = multilabel_confusion_matrix(
        y_pred=df[view_mapping],
        y_true=df["view_true"],
        labels=["a2c", "a4c", "plax", "other"],
    )

    df_mcm = pd.DataFrame(
        np.reshape(mcm, (4, 4)) / np.sum(mcm[0]),
        columns=["tn", "fp", "fn", "tp"],
        index=["a2c", "a4c", "plax", "other"],
    )

    eval_out = df_mcm.rename_axis("view").reset_index()

    eval_out["model_name"] = model_name
    eval_out["img_dir"] = img_dir
    eval_out["date_run"] = df["date_run"]  # [0] #TODO: debug error, no KeyValue 0
    eval_out["view_mapping"] = view_mapping
    eval_out["study_filter"] = study_filter

    cols = list(eval_out.columns[5::]) + list(eval_out.columns[:5])
    eval_out = eval_out[cols]

    return eval_out


def evaluate_views(
    img_dir, model_name, date_run=datetime.date.today(), study_filter=None, if_exists="append"
):
    """Filters and then evaluates classification.predictions table
    
    The functions applies evaluate_view_map which filters the 
    classification.predictions table on img_dir, model_name and date_run.
    
    :param img_dir: directory with jpg echo images for classification
    :param model_name: name of model used for making predictions
    :param date_run: date on which predictions were made
    :param study_filter: dictionary mapping filter name to list of study_ids
    :param if_exists (str): write action if table exists, must be 'replace' or 'append'    
    
    """
    for view_mapping in ["view4_dev", "view4_seg"]:
        eval_out = evaluate_view_map(
            img_dir, model_name, date_run, view_mapping, study_filter=None
        )

        io_class = dbReadWriteClassification()
        io_class.save_to_db(eval_out, "evaluations", if_exists)

        logger.info(
            "Evaluated {} {} {} {})".format(img_dir, model_name, date_run, view_mapping)
        )
