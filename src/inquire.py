"""
* Checkbox question example
* run example by typing `python example/checkbox.py` in your console
"""
from __future__ import print_function, unicode_literals
from pprint import pprint

from PyInquirer import style_from_dict, Token, prompt, Separator


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


## modules
if (answers['options'][3].lower().replace(" ", '_') == 'classification'):
    print("###", answers['options'][0].lower().replace(' ', '_').replace("'", ''))

## evaluation
if (answers['options'][4].lower().replace(' ', '_').replace("'",'') == 'evaluate'):
    print("evaluation")
else:
    print(answers['options'][4].lower().replace(' ', '_').replace("'",''))



