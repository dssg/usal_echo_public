'''
Script for running full pipeline
'''

import os

from d07_luigi.luigi_pipeline import Pipeline


if __name__ == '__main__':
    os.system('luigi --module luigi_pipeline Pipeline --local-scheduler') # path issues? i.e. Pipeline() defined in d07
