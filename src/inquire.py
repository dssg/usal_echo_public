"""
* Checkbox question example
* run example by typing `python example/checkbox.py` in your console
"""
from __future__ import print_function, unicode_literals

from PyInquirer import prompt, Separator


print("##################################################################")
print("####   CLI CIBERCV-USAL | DSSG - Imperial College London 2019 ####")
print("##################################################################")
print("")

questions = [
    {
        'type': 'checkbox',
        'message': 'Select options',
        'name': 'options',
        'choices': [
            Separator('====== Download DICOM image ======'),
            {
                'name': 'Download image'
            },
            {
                'name': "Don't download image",
                'checked': 'true'
            },
            Separator('====== Get metadata ======'),
            {
                'name': 'Ingest DICOM'
            },
            {
                'name': "Don't ingest DICOM"
            },
            Separator('====== Ingest XCelera ======'),
            {
                'name': 'Ingest Xcelara'
            },
            {
                'name': "Don't ingest Xcelera"
            },
            Separator('====== Select module ======'),
            {
                'name': 'Classification'
            },
            {
                'name': 'Segmentation'
            },
            {
                'name': 'Measurements'
            },
            {
                'name': 'All'
            },
            Separator('====== Evaluation ======'),
            {
                'name': 'Evaluate'
            },
            {
                'name': "Don't evaluate"
            }
        ],
        'validate': lambda answer: 'You must choose at least one topping.' \
            if len(answer) == 0 else True
    }
]

answers = prompt(questions)


# TODO there are different possible combinations, we need to define the conditional statements

## download
if (answers['options'][0].lower().replace(" ",'_').replace("'", '') == 'download_image'):
    print("### ", answers['options'][0].lower().replace(" ",'_').replace("'", ''))
    # TODO call the correct function

## metadata
if (answers['options'][1].lower().replace(" ",'_').replace("'", '') == 'ingest_dicom'):
    print("### ", answers['options'][1].lower().replace(" ",'_').replace("'", ''))
    # TODO call the correct function

## metadata
if (answers['options'][2].lower().replace(" ",'_').replace("'", '') == 'ingest_xcelera'):
    print("### ", answers['options'][2].lower().replace(" ",'_').replace("'", ''))
    # TODO call the correct function

## modules
if (answers['options'][3].lower().replace(" ", '_') == 'classification'):
    print("###", answers['options'][0].lower().replace(' ', '_').replace("'", ''))
    # TODO call the correct function

## evaluation
if (answers['options'][4].lower().replace(' ', '_').replace("'",'') == 'evaluate'):
    print("###", answers['options'][4].lower().replace(' ', '_').replace("'", ''))
    # TODO call the correct function



