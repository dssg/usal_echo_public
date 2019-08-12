# coding: utf-8

import os
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread

from d00_utils.db_utils import dbReadWriteClassification
from d03_classification import vgg
from d00_utils.log_utils import setup_logging

logger = setup_logging(__name__, __name__)

view_classes = [
    "plax_far",
    "plax_plax",
    "plax_laz",
    "psax_az",
    "psax_mv",
    "psax_pap",
    "a2c_lvocc_s",
    "a2c_laocc",
    "a2c",
    "a3c_lvocc_s",
    "a3c_laocc",
    "a3c",
    "a4c_lvocc_s",
    "a4c_laocc",
    "a4c",
    "a5c",
    "other",
    "rvinf",
    "psax_avz",
    "suprasternal",
    "subcostal",
    "plax_lac",
    "psax_apex",
]


def classify(img_dir, feature_dim, label_dim, model_path):
    """Classifies echo images in img_dir.

    :param img_dir: directory with jpg echo images for classification
    :param feature_dim:
    :param label_dim: number of classes on which model has been trained
    :param model_path: path to trained model for making predictions

    """
    # Initialise tensorflow
    tf.reset_default_graph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    model = vgg.Network(0.0, 0.0, feature_dim, label_dim, False)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # Classify views
    predictions = {}
    for filename in os.listdir(img_dir):
        image = imread(os.path.join(img_dir, filename), flatten=True).astype("uint8")
        img_data = [image.reshape((224, 224, 1))]
        predictions[filename] = np.around(
            model.probabilities(sess, img_data), decimals=3
        ).tolist()[0]

    return predictions


def run_classify(img_dir, model_path, if_exists, feature_dim=1):
    """Writes classification predictions to database.

    :param img_dir: directory with jpg echo images for classification
    :param model_path: path to trained model for making predictions
    :param if_exists (str): write action if table exists
    :param feature_dim: default=1

    """
    label_dim = len(view_classes)
    model_name = os.path.basename(model_path)

    predictions = classify(img_dir, feature_dim, label_dim, model_path)

    df_columns = ["output_" + x for x in view_classes]
    df = (
        pd.DataFrame.from_dict(predictions, columns=df_columns, orient="index")
        .rename_axis("file_name")
        .reset_index()
    )
    df["file_name"] = df["file_name"].apply(lambda x: x.split(".")[0])
    df["study_id"] = df["file_name"].apply(lambda x: x.split("_")[1])
    df["model_name"] = model_name
    df["date_run"] = datetime.datetime.now()
    cols = ["study_id", "file_name", "model_name", "date_run"] + df_columns
    df = df[cols]

    io_classification = dbReadWriteClassification()
    io_classification.save_to_db(df, "prediction_frames", if_exists)

    logger.info(
        "{} prediction on frames with model {} (feature_dim={})".format(
            os.path.basename(img_dir), model_name, feature_dim
        )
    )


def agg_predictions(if_exists="replace"):
    """Aggregates predictions by study and instance.
    
    Fetches predictions from classification.prediction_frames and 
    calculates the mean.
    
    :param if_exists (str): write action if table exists
    """

    io_classification = dbReadWriteClassification()

    prediction_frames = io_classification.get_table("prediction_frames")
    prediction_frames["file_name"] = prediction_frames["file_name"].apply(
        lambda x: x.rsplit("_", 1)[0]
    )

    mean_cols = ["output_" + x for x in view_classes]
    agg_cols = dict(zip(mean_cols, ["mean"] * len(mean_cols)))
    agg_cols["prediction_frame_id"] = "count"

    predictions = (
        prediction_frames.groupby(["study_id", "file_name", "model_name", "date_run"])
        .agg(agg_cols)
        .reset_index(drop=False)
    )
    predictions.rename(columns={"prediction_frame_id": "frame_count"}, inplace=True)

    io_classification.save_to_db(predictions, "predictions", if_exists)
    logger.info('Wrote aggregated predictions to db with method "{}"'.format(if_exists))
