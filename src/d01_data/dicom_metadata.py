import pandas as pd
import os
import boto3
import tempfile

def get_dicom_metadata(bucket, object_path):
    
    """Get all dicom tags for file in object_path.
    
    This function uses gdcmdump to retrieve the metadata tags of the file in object_bath.
    The tags are as a pandas dataframe.
    
    ** Requirements 
    libgdcm: 
        Unix install with `sudo apt-get install libgdcm-tools`
        Mac install with `brew install gdcm`
    .aws/credentials file with s3 access details saved as default profile
    
    Parameters:
        bucket (str): s3 bucket
        object_path (str): string to append to metadata file name 'dicom_metadata.csv', default=None
        
    Output:
        pandas DataFrame object with columns=['dirname','filename','tag1','tag2','value']
    """
    
    s3 = boto3.client('s3')
    tmp = tempfile.NamedTemporaryFile()

    # Dump metadata of file to temp file
    s3.download_file(bucket, object_path, tmp.name)
    os.system('gdcmdump '+ tmp.name +' > temp.txt')

    dir_name = object_path.split('/')[0]
    file_name = object_path.split('/')[1].split('.')[0]

    # Parse temp.txt file to extract tags
    temp_file='temp.txt'
    meta = []
    with open(temp_file, 'r') as f:
        line_meta = []
        for one_line in f:            
            try:
                clean_line = one_line.replace(']','').strip()
                if not clean_line: # ignore empty lines
                    continue
                elif not clean_line.startswith('#'): # ignore comment lines:
                    tag1 = clean_line[1:5]
                    tag2 = clean_line[6:10]
                    value = clean_line[16:clean_line.find('#')].strip()
                    line_meta=[dir_name, file_name, tag1, tag2, value]
                    meta.append(line_meta)
            except IndexError:
                break
                    
    df = pd.DataFrame.from_records(meta, columns=['dirname','filename','tag1','tag2','value'])
    df_dedup = df.drop_duplicates(keep='first')
    df_dedup_goodvals = df_dedup[~df_dedup.value.str.contains('no value')]
    df_dedup_goodvals_short = df_dedup_goodvals[df_dedup_goodvals['value'].str.len()<50]
    
    return df_dedup_goodvals_short


def write_dicom_metadata(df, metadata_file_name=None):
    
    """Save metadata
    
    Writes the output of 'get_dicom_metadata()' to a csv file.
    
    Parameters:
        df (pandas.DataFrame): output of 'get_dicom_metadata()'
        metadata_file_name (str): string to append to metadata file name 'dicom_metadata.csv', default=None
        
    Output:
        file (csv): saves to ~/data_usal/02_intermediate/dicom_metadata.csv
        """

    data_path = os.path.join(os.path.expanduser('~'),'data_usal','02_intermediate')
    os.makedirs(os.path.expanduser(data_path), exist_ok=True)
    if metadata_file_name is None:
        dicom_meta_path = os.path.join(data_path,'dicom_metadata.csv')
    else:
        dicom_meta_path = os.path.join(data_path,'dicom_metadata_'+str(metadata_file_name)+'.csv')
    if not os.path.isfile(dicom_meta_path): # create new file if it does not exist
        print('Creating new metadata file')
        df.to_csv(dicom_meta_path, index=False)
    else: # if file exists append
        df.to_csv(dicom_meta_path, mode='a', index=False, header=False)
               
    print('dicom metadata saved for study {}, directory {}'.format(df.iloc[0,0], df.iloc[0,1]))
    

def get_matching_s3_objects(bucket, prefix='', suffix=''):
    """
    Generate objects in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        this suffix (optional).
        
    ** code from https://alexwlchan.net/2018/01/listing-s3-keys-redux/

    """
    s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)

        try:
            contents = resp['Contents']
        except KeyError:
            return

        for obj in contents:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield obj

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def get_matching_s3_keys(bucket, prefix='', suffix=''):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    
    ** code from https://alexwlchan.net/2018/01/listing-s3-keys-redux/
    
    """
    for obj in get_matching_s3_objects(bucket, prefix, suffix):
        yield obj['Key']
        
        
if __name__ == '__main__':
    check = input("Do you want to fetch all dicom metadata? This will take ~48 hours. Type YES to continue. Any other input will stop the process.")
    
    if check.lower() == 'yes':
        for key in get_matching_s3_keys('cibercv','','.dcm'): 
            df = get_dicom_metadata('cibercv', key)
            write_dicom_metadata(df)
        os.remove('temp.txt')
        
    else:
        print('Exciting the dicom metadata retrieval process. Rerun the script and type YES when prompted if this was a mistake.')