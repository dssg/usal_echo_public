import pandas as pd
import boto3
import os
import tempfile
import psycopg2

from json import load
from sqlalchemy import create_engine


def _load_db_credentials():
    with open('credentials_db.json') as f:
        credentials = load(f)

    return credentials


def save_to_db(df):
    credentials = _load_db_credentials()
    connection_str =  "postgresql://{}:{}@{}/{}".format(credentials['user'],
                                                         credentials['password'],
                                                         credentials['host'],
                                                         credentials['database'])
    conn = create_engine(connection_str)
    df.to_sql('metadata', conn, if_exists='append', index=False)


def get_image_metadata(s3, bucket, file_name):
    '''

    :param file_name:
    :return:
    '''

    tmp = tempfile.NamedTemporaryFile()

    # Dump metadata of file to temp file
    s3.download_file(bucket, file_name, tmp.name)
    os.system('gdcmdump ' + tmp.name + ' > temp.txt')

    dir_name = file_name.split('/')[0]
    name_file = file_name.split('/')[1].split('.')[0]

    # Parse temp.txt file to extract tags
    temp_file = 'temp.txt'
    meta = []
    with open(temp_file, 'r') as f:
        line_meta = []
        for one_line in f:
            try:
                clean_line = one_line.replace(']', '').strip()
                if not clean_line:  # ignore empty lines
                    continue
                elif not clean_line.startswith('#'):  # ignore comment lines:
                    tag1 = clean_line[1:5]
                    tag2 = clean_line[6:10]
                    value = clean_line[16:clean_line.find('#')].strip()
                    line_meta = [dir_name, name_file, tag1, tag2, value]
                    meta.append(line_meta)
            except IndexError:
                break

    df = pd.DataFrame.from_records(meta, columns=['dirname', 'filename', 'tag1', 'tag2', 'value'])
    df_dedup = df.drop_duplicates(keep='first')
    df_dedup_goodvals = df_dedup[~df_dedup.value.str.contains('no value')]
    df_dedup_goodvals_short = df_dedup_goodvals[df_dedup_goodvals['value'].str.len() < 50]

    save_to_db(df_dedup_goodvals_short)


def get_objects_from_bucket(bucket, suffix):
    '''
    Get all the keys from the bucket

    :param bucket: name of the bucket -without the s3-
    :param suffix: extension for just dicom files
    :return: list of keys
    '''

    session = boto3.Session(profile_name='usal')
    s3 = session.client('s3')

    keys = []
    kwargs = {'Bucket': bucket}

    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            if obj['Key'].endswith(suffix):
                keys.append(obj['Key'])
                get_image_metadata(s3, bucket, obj['Key'])

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

    return keys





def process_image(bucket, suffix):
    '''

    :param bucket:
    :param suffix:
    :return:
    '''
    get_objects_from_bucket(bucket, suffix)




if __name__ == '__main__':
    process_image('cibercv', '.dcm')