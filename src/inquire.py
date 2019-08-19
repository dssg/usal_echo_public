from PyInquirer import prompt
import os
import yaml

from d01_data.ingestion_dcm import ingest_dcm
from d01_data.ingestion_xtdb import ingest_xtdb
from d02_intermediate.clean_dcm import clean_dcm_meta
from d02_intermediate.clean_xtdb import clean_tables
from d02_intermediate.filter_instances import filter_all
from d02_intermediate.download_dcm import (
    s3_download_decomp_dcm,
    dcmdir_to_jpgs_for_classification,
)
from d03_classification.predict_views import (
    run_classify,
    agg_probabilities,
    predict_views,
)
from d03_classification.evaluate_views import evaluate_views
from d02_intermediate.create_seg_view import create_seg_view
from d04_segmentation.segment_view import run_segment
from d04_segmentation.generate_masks import generate_masks
from d04_segmentation.evaluate_masks import evaluate_masks
from d05_measurement.retrieve_meas import retrieve_meas
from d05_measurement.calculate_meas import calculate_meas
from d05_measurement.evaluate_meas import evaluate_meas

with open("./conf/local/path_parameters.yml") as f:
    paths = yaml.safe_load(f)

bucket = paths["bucket"]
dcm_dir = os.path.expanduser(paths["dcm_dir"])
img_dir = os.path.expanduser(paths["img_dir"])
model_dir = os.path.expanduser(paths["model_dir"])
classification_model = paths["classification_model"]


def _print_welcome_message():
    print("##################################################################")
    print("####   CLI USAL-ECHO | DSSG - Imperial College London 2019    ####")
    print("##################################################################")
    print("")


def _format_answer(option):
    """
    Process the option selected on the CLI, passes it to lower case, change spaces for underscore and eliminates the single quote used in contractions
    :param option: options selected on CLI
    :return: processed option
    """
    return option.lower().replace(" ", "_").replace("'", "")


def ingest_metadata():
    """
    Asks for selection on ingesting DICOM files option
    :return: selected option
    """
    ingest_dicom = {
        "type": "list",
        "name": "ingest_metadata",
        "message": "Do you want to ingest dicom metadata?",
        "choices": ["Ingest DICOM metadata", "Don't Ingest DICOM metadata"],
    }
    answers = prompt(ingest_dicom)

    return answers


def ingest_xcelera():
    """
    Asks for selection on ingesting XCelera options
    :return:
    """
    ingest_dicom = {
        "type": "list",
        "name": "ingest_xcelera",
        "message": "Do you want to ingest Xcelera data?",
        "choices": ["Ingest Xcelera", "Don't Ingest Xcelera"],
    }
    answers = prompt(ingest_dicom)

    return answers


def download_files():
    """
    Asks for selection on downloading files options
    :return: selected_option
    """
    download = {
        "type": "list",
        "name": "download_file",
        "message": "Do you want to download DICOM files?",
        "choices": ["Download", "Don't Download"],
    }
    answers = prompt(download)

    return answers


def _download_files_args():
    """
    Asks for the arguments required for downloading DICOM files
    :return: selected option
    """
    questions = [
        {
            "type": "input",
            "name": "download_file_train_test_ratio",
            "message": "Train test ratio (float, range(0, 1)):",
        },
        {
            "type": "input",
            "name": "download_file_downsample_ratio",
            "message": "Down sample ratio (float, range(0, 1)):",
        },
    ]

    answers = prompt(questions)

    return answers


def modules():
    """
    Asks for modules selection options
    :return: module selected
    """
    modules = {
        "type": "checkbox",
        "name": "module",
        "message": "Which modules do you want to execute?",
        "choices": [
            {"name": "classification", "checked": True},
            {"name": "segmentation"},
            {"name": "measurements"},
        ],
    }
    answers = prompt(modules)

    return answers


def _pipeline_args():
    """
    Asks for parameters required by classification, segmentation and measurement
    :return: arguments defined
    """
    questions = [
        {
            "type": "input", 
            "name": "dir_name", 
            "message": "Directory with echo images:"}
    ]
    answers = prompt(questions)

    return answers


def process_choices(options):
    """
    Depending on the selection made by the user a specific pipeline is triggered
    :param options: dictionary with options selected
    :return:
    """
    if _format_answer(options["ingest_metadata"]).startswith("ingest"):
        ingest_dcm(bucket)
        clean_dcm_meta()
    if _format_answer(options["ingest_xcelera"]).startswith("ingest"):
        ingest_xtdb(bucket)
        clean_tables()
        filter_all()
    if _format_answer(options["download_file"]).startswith("download"):
        train_test_ratio = float(options["download_file_train_test_ratio"])
        downsample_ratio = float(options["download_file_downsample_ratio"])
        s3_download_decomp_dcm(
            train_test_ratio, downsample_ratio, dcm_dir, bucket=bucket
        )
    if "classification" in options["module"]:
        print("Starting classification.")
        dir_name = options["dir_name"]
        img_dir_path = os.path.join(img_dir, dir_name)
        dcmdir_to_jpgs_for_classification(dcm_dir, img_dir_path)
        run_classify(img_dir_path, os.path.join(model_dir, classification_model))
        agg_probabilities()
        predict_views()
        evaluate_views(img_dir_path, classification_model)
    if "segmentation" in options["module"]:
        dir_name = options["dir_name"]
        dcm_dir_path = os.path.join(dcm_dir, dir_name)
        run_segment(dcm_dir_path, model_dir)
        create_seg_view()
        generate_masks(dcm_dir_path)
        evaluate_masks()
    if "measurements" in options["module"]:
        dir_name = options["dir_name"]
        retrieve_meas()
        calculate_meas(dir_name)
        evaluate_meas(dir_name)


def cli():
    _print_welcome_message()

    options = ingest_metadata()
    options.update(ingest_xcelera())
    options.update(download_files())

    if _format_answer(options["download_file"]).startswith("download"):
        options.update(_download_files_args())

    options.update(modules())
    if len(options["module"]) > 0:
        options.update(_pipeline_args())

    process_choices(options)


if __name__ == "__main__":
    cli()
    print("Pipeline execution complete. Check log files for potential errors.")
