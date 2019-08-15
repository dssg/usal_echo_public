#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import boto3
import os
from d00_utils.log_utils import setup_logging
logger = setup_logging(__name__, 'd00_utils.download_s3_objects')


# TODO specify aws s3 credentials


def get_matching_s3_objects(bucket, prefix="", suffix=""):
    """Generate objects in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        this suffix (optional).
        
    ** code from https://alexwlchan.net/2018/01/listing-s3-keys-redux/

    """
    s3 = boto3.client("s3")
    kwargs = {"Bucket": bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs["Prefix"] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)

        try:
            contents = resp["Contents"]
        except KeyError:
            return

        for obj in contents:
            key = obj["Key"]
            if key.startswith(prefix) and key.endswith(suffix):
                yield obj

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        except KeyError:
            break


def get_matching_s3_keys(bucket, prefix="", suffix=""):
    """Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    
    """
    for obj in get_matching_s3_objects(bucket, prefix, suffix):
        yield obj["Key"]
        
        
def download_s3_objects(bucket, outfile, prefix='', suffix='.dcm'):
    """Download all the objects with a specific suffix from a s3 bucket.

    :param bucket: Name of the S3 bucket.
    :param outfile: name of file download
    :param prefix: Only fetch objects whose key starts with this prefix (optional).
    :param suffix: Only fetch objects whose keys end with this suffix, default='.dcm'

    """
    outdir = os.path.dirname(outfile)
    os.makedirs(outdir, exist_ok=True)

    for key in get_matching_s3_keys(bucket, prefix, suffix):
        try:
            s3 = boto3.client('s3')
            s3.download_file(bucket, key, outfile)
            logger.info('{} Download '.format(key))
        except:
            logger.error('{} Download error'.format(key))

    return
