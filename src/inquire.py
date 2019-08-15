"""
* Checkbox question example
* run example by typing `python example/checkbox.py` in your console
"""
from __future__ import print_function, unicode_literals
from pprint import pprint
from PyInquirer import prompt, Separator

from d01_data.ingestion_dcm import ingest_dcm
from d01_data.ingestion_xtdb import ingest_xtdb
from d02_intermediate.download_dcm import s3_download_decomp_dcm
#from d02_intermediate.clean_dcm import clean_dcm
from d02_intermediate.filter_instances import filter_all
from d02_intermediate.clean_xtdb import clean_tables
from d03_classification.predict_view import run_classify, agg_probabilities


def _print_welcome_message():
    print("##################################################################")
    print("####   CLI CIBERCV-USAL | DSSG - Imperial College London 2019 ####")
    print("##################################################################")
    print("")



def _format_answer(option):
    """
    Process the option selected on the CLI, passes it to lower case, change spaces for underscore and eliminates the single quote used in contractions
    :param option: options selected on CLI
    :return: processed option
    """
    return option.lower().replace(" ", "_").replace("'","")


def ingest_metadata():
    """
    Asks for selection on ingesting DICOM files option
    :return: selected option
    """
    ingest_dicom = {
        'type': 'list',
        'name': 'ingest_metadata',
        'message': 'Do you want to ingest dicom metadata?',
        'choices': ['Ingest DICOM metadata', "Don't Ingest DICOM metadata"]
    }
    answers = prompt(ingest_dicom)

    return answers

def ingest_xcelera():
    """
    Asks for selection on ingesting XCelera options
    :return:
    """
    ingest_dicom = {
        'type': 'list',
        'name': 'ingest_xcelera',
        'message': 'Do you want to ingest XCelera data?',
        'choices': ['Ingest XCelera', "Don't Ingest XCelera"]
    }
    answers = prompt(ingest_dicom)

    return answers

def download_files():
    """
    Asks for selection on downloading files options
    :return: selected_option
    """
    download = {
        'type': 'list',
        'name': 'download_file',
        'message': 'Do you want to download DICOM files?',
        'choices': ['Download', "Don't Download"]
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
            'type': 'input',
            'name': 'download_file_a1',
            'message': 'DICOM directory:'
        },
        {
            'type': 'input',
            'name': 'download_file_a2',
            'message': 'Table name:'
        },
        {
             'type': 'input',
             'name': 'download_file_a3',
             'message': 'Train test ratio (float):'
        },
        {
            'type': 'input',
            'name': 'download_file_a4',
            'message': 'Down sample ratio (float):'
        }
    ]

    answers = prompt(questions)

    return answers


def modules():
    """
    Asks for modules selection options
    :return: module selected
    """
    modules = {
        'type': 'list',
        'name': 'module',
        'message': 'Which module do you want to execute?',
        'choices': ['Classification', "Segmentation", "Measurements", "All"]
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
            'type': 'input',
            'name': 'classification_a1',
            'message': 'Image directory:'
        },
        {
            'type': 'input',
            'name': 'classification_a2',
            'message': 'Model path:'
        }
    ]

    answers = prompt(questions)

    return answers


def process_choices(options):
    """
    Depending on the selection made by the user a specific pipeline is triggered
    :param options: dictionary with options selected
    :return:
    """
    if _format_answer(options['ingest_metadata']).startswith('ingest'):
        ingest_dcm()
        # TODO metadata_path_file is missing!
        #clean_dcm()
        #filter_all()
    elif _format_answer(options['ingest_xcelere']).startswith('ingest'):
        ingest_xtdb()
        clean_tables()
        filter_all()
    elif _format_answer(options['download_file']).startswith('download'):
        a1 = options['download_file_a1']
        a2 = options['downlaod_file_a2']
        a3 = options['download_file_a3']
        a4 = options['download_file_a4']
        s3_download_decomp_dcm(a1, a2, a3, a4)
    elif (_format_answer(options['module']) == 'classification'):
        a1 = options['classification_a1']
        a2 = options['classification_a2']
        run_classify(a1, a2)
        agg_probabilities()
        # TODO add the predict_views function from test_predict branch

def main():
    _print_welcome_message()
    options = ingest_metadata()
    options.update(ingest_xcelera())
    options.update(download_files())
    if (_format_answer(options['download_file']).startswith('download')):
        options.update(download_files_args())
    options.update(modules())
    if (_format_answer(options['module']) == 'classification '):
        options.update(classification_args())
    elif (_format_answer(options['module']) == 'segmentation'):
        pass
    elif (_format_answer(options['module']) == 'measurements'):
        pass
    else:
        pass

    process_choices(options)


if __name__ == '__main__':
    main()

