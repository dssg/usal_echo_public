#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from d01_data.ingestion_dcm import ingest_dcm
from d01_data.ingestion_xtdb import ingest_xtdb
from d02_intermediate.clean_dcm import clean_dcm_meta
from d02_intermediate.clean_xtdb import clean_tables
from d02_intermediate.filter_instances import filter_all
from d02_intermediate.download_dcm import decompress_dcm, s3_download_decomp_dcm, dcmdir_to_jpgs_for_classification
from d03_classification.predict_views import run_classify, agg_probabilities, predict_views
from d03_classification.evaluate_views import evaluate_views
from d04_segmentation.create_seg_view import create_seg_view
from d04_segmentation.segment_view import run_segment
from d04_segmentation.generate_masks import generate_masks
from d04_segmentation.evaluate_masks import evaluate_masks
from d05_measurement.retrieve_meas import retrieve_meas
from d05_measurement.calculate_meas import calculate_meas
from d05_measurement.evaluate_meas import evaluate_meas