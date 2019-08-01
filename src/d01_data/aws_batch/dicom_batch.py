import os
import boto3

from ..dicom_metadata import get_dicom_metadata
from ...d00_utils.db_utils import save_to_db

db_table = "metadata"
credentials_file = os.path.join(os.path.expanduser("~"), ".psql_credentials.json")


def dicom_meta_to_db(bucket, suffix):
    """
    Get all the keys with a specific suffix from a s3 bucket.

    :param bucket: name of the bucket -without the s3-
    :param suffix: extension for just dicom files
    :return: list of keys
    """

    s3 = boto3.client("s3")

    keys = []
    kwargs = {"Bucket": bucket}

    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp["Contents"]:
            if obj["Key"].endswith(suffix):
                keys.append(obj["Key"])
                df = get_dicom_metadata(bucket, obj["Key"])
                save_to_db(df, db_table, credentials_file)

        try:
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        except KeyError:
            break

    return keys


if __name__ == "__main__":
    if not os.path.exists(credentials_file):
        print(
            "Create a database credentials file at ~/.psql_credentials.json . Then try again."
        )
    else:
        dicom_meta_to_db("cibercv", ".dcm")
        os.remove("temp.txt")
