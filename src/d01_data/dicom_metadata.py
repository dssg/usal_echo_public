import pandas as pd
import os
import boto3
import tempfile

from ..d00_utils.s3_utils import get_matching_s3_keys
from ..d01_utils.db_utils import save_to_db

def get_dicom_metadata(bucket, file_path, description=False):
    
    """Get all dicom tags for file in file_path.
    
    This function uses gdcmdump to retrieve the metadata tags of the file in object_bath.
    The tags are as a pandas dataframe.
    
    ** Requirements 
    libgdcm: 
        Unix install with `sudo apt-get install libgdcm-tools`
        Mac install with `brew install gdcm`
    .aws/credentials file with s3 access details saved as default profile
    
    :param bucket (str): s3 bucket
    :param object_path (str): path to dicom file
    :param description (bool): include dicom tag descriptions instead of values, default=False        
    :return: pandas DataFrame object with columns=['dirname','filename','tag1','tag2','value']
    """
    
    s3 = boto3.client('s3')
    tmp = tempfile.NamedTemporaryFile()

    # Dump metadata of file to temp file
    s3.download_file(bucket, file_path, tmp.name)
    os.system('gdcmdump '+ tmp.name +' > temp.txt')

    dir_name = file_path.split('/')[0]
    file_name = file_path.split('/')[1].split('.')[0]

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
                    if description == False:
                        value = clean_line[16:clean_line.find('#')].strip()
                    elif description == True:
                        value = clean_line[clean_line.find('#')+2:].strip()
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
    
    """Write the output of 'get_dicom_metadata()' to a csv file.
    
    :param df (pandas.DataFrame): output of 'get_dicom_metadata()'
    :param metadata_file_name (str): string to append to metadata 
        file name 'dicom_metadata.csv', default=None
    :return: csv file; saves to ~/data_usal/02_intermediate/dicom_metadata.csv
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

    
    
def get_meta_lite(dicom_tags, save_to_db=False, credentials_file=None, db_table=None):

    """Select subset of dicom tags and save to database.
    
    :param dicom_tags (dict): dict of dicom tag descriptions and tag tuple
    :param save_to_db (bool): if True saves tag subset to postgres database; default=False
    :param credentials_file (str): path to credentials file; required if save_to_db=True
    :param db_table (str): name of database table to write to; required if save_to_db=True
    :return: pandas dataframe with filtered metadata
    """    
    
    datalist = []
    # Read metadata in chunks to avoid kernel crashing due to large data volume.
    for chunk in pd.read_csv(metadata_path, chunksize=1000000, dtype={'dirname':'category','filename':'category',
                                                                      'tag1':'category','tag2':'category'}):
        chunk['tags'] = list(zip(chunk['tag1'],chunk['tag2']))
        filtered_chunk = chunk[chunk['tags'].isin(dicom_tags.values())]
        if save_to_db is True:
            try:
                save_to_db(filtered_chunk, db_table, credentials_file)
            except:
                raise
            print('saved chunks to db')
        datalist.append(filtered_chunk)
    
    meta = pd.concat(datalist)
    
    return meta
    
        
        
if __name__ == '__main__':
    check = input("Do you want to fetch all dicom metadata? This will take ~48 hours. Type YES to continue. Any other input will stop the process.")
    
    if check.lower() == 'yes':
        for key in get_matching_s3_keys('cibercv','','.dcm'): 
            df = get_dicom_metadata('cibercv', key)
            write_dicom_metadata(df)
        os.remove('temp.txt')
        
    else:
        print('Exiting the dicom metadata retrieval process. Rerun the script and type YES when prompted if this was a mistake.')