"""
* Checkbox question example
* run example by typing `python example/checkbox.py` in your console
"""
from __future__ import print_function, unicode_literals
from pprint import pprint
from PyInquirer import prompt, Separator

#from d01_ data.ingestion_dcm import ingest_dcm


def _print_welcome_message():
    print("##################################################################")
    print("####   CLI CIBERCV-USAL | DSSG - Imperial College London 2019 ####")
    print("##################################################################")
    print("")
    print(" Usage: ")
    print(" + Get metadata requires n arguments")


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

        :return:
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
    Asks for the arguments required for downloaading DICOM files
    :return:
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

    :return:
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

    :return:
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
    pprint(options)


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
    elif (_format_answer(options['module']) == 'segmentaton'):
        pass
    elif (_format_answer(options['module']) == 'measurements'):
        pass
    else:
        pass

    process_choices(options)


if __name__ == '__main__':
    main()

