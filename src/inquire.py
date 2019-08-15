from __future__ import print_function, unicode_literals
from pprint import pprint
from PyInquirer import prompt
import os

from d01_data.ingestion_dcm import ingest_dcm
from d01_data.ingestion_xtdb import ingest_xtdb
from d02_intermediate.clean_dcm import clean_dcm_meta
from d02_intermediate.clean_xtdb import clean_tables
from d02_intermediate.filter_instances import filter_all
from d02_intermediate.download_dcm import s3_download_decomp_dcm, dcmdir_to_jpgs_for_classification
from d03_classification.predict_views import run_classify, agg_probabilities, predict_views
from d03_classification.evaluate_views import evaluate_views

#from d05_measurement.retrieve_meas import retrieve_meas
#import d05_measurement.calculate_meas
#from d05_measurement.evaluate_meas import evaluate_meas

dcm_dir = os.path.expanduser('~/data/01_raw')
img_dir = os.path.expanduser('~/data/02_intermediate')
model_path = os.path.expanduser('~/models/model.ckpt-6460')


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


def download_files_args():
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


def classification_args():
    """
    Asks for classifications parameters
    :return: arguments defined
    """
    questions = [
        {
            "type": "input",
            "name": "classification_img_dir",
            "message": "Directory with echo images:",
        }
    ]

    answers = prompt(questions)

    return answers


def segmentation_args():
    """
    Asks for classifications parameters
    :return: arguments defined
    """
    questions = [
        {"type": "input", "name": "_a1", "message": "Directory with echo images:"},
        {"type": "input", "name": "_a2", "message": "Path to trained model:"},
    ]

    answers = prompt(questions)

    return answers


def measurements_args():
    """
    Asks for classifications parameters
    :return: arguments defined
    """
    questions = [
        {"type": "input", "name": "_a1", "message": "Directory with echo images:"},
        {"type": "input", "name": "_a2", "message": "Path to trained model:"},
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
        ingest_dcm()
        clean_dcm_meta()
    elif _format_answer(options["ingest_xcelera"]).startswith("ingest"):
        ingest_xtdb()
        clean_tables()
        filter_all()
    elif _format_answer(options["download_file"]).startswith("download"):
        train_test_ratio = float(options["download_file_train_test_ratio"])
        downsample_ratio = float(options["download_file_downsample_ratio"])
        dir_name = s3_download_decomp_dcm(train_test_ratio, downsample_ratio, dcm_dir)
    elif "classification" in options["module"]:
        print('Starting classification.')
        dir_name = options["classification_img_dir"]
        dcmdir_to_jpgs_for_classification(dcm_dir, os.path.join(img_dir, dir_name))
        run_classify(os.path.join(img_dir, dir_name), model_path)
        agg_probabilities()
        predict_views()
        evaluate_views(os.path.join(img_dir, dir_name), os.path.basename(model_path))
    elif "segmentation" in options["module"]:
        pass
    elif "measurements" in options["module"]:
#        retrieve_meas()
#        evaluate_meas()
        pass


def cli():
    _print_welcome_message()

    options = ingest_metadata()
    options.update(ingest_xcelera())
    options.update(download_files())

    if _format_answer(options["download_file"]).startswith("download"):
        options.update(download_files_args())

    options.update(modules())
    if "classification" in options["module"]:
        options.update(classification_args())
    elif "segmentation" in options["module"]:
        options.update(segmentation_args())
    elif "measurements" in options["module"]:
        options.update(measurements_args())

    process_choices(options)


if __name__ == "__main__":
    cli()
    print('Pipeline executed successfully')
