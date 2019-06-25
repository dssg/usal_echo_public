import pandas as pd
import os

def get_dicom_metadata(dirpath):
    """Get all dicom tags for files in dirpath.
    
    This function uses gdcmdump to retrieve the metadata tags of all files in dirpath.
    The tags are written to a csv file with header=['dirname','filename','tag1','tag2','value']
    
    ** Requires libgdcm: 
        Unix install with `sudo apt-get install libgdcm-tools`
        Mac install with `brew install gdcm`
    
    Parameters:
        dirpath (str): directory path
        
    Output:
        file (csv): saves to ~/data_usal/02_intermediate/dicom_metadata.csv
    """
    
    # Get list of all files in directory
    all_files = os.listdir(dirpath)
    dicom_files = [dcm for dcm in all_files if dcm.endswith('.dcm')]
    dicom_files.sort()    
    
    # Dump metadata of all files in study directory to temp.txt
    os.system('gdcmdump '+ dirpath +' > temp.txt')
    dir_name = os.path.basename(dirpath)

    # Parse temp.txt file to extract tags
    temp_file='temp.txt'
    meta = []
    file_iterator = -1 # needed to associate filename with metadata
    with open(temp_file, 'r') as f:
        line_meta = []
        for one_line in f:            
            try:
                file_name = dicom_files[file_iterator]
                clean_line = one_line.replace(']','').strip()
                if "# Dicom-File-Format" in clean_line: # check for new header
                    file_iterator += 1
                    continue
                if clean_line.startswith('#'): # ignore comment lines
                    continue
                if not clean_line: # ignore empty lines
                    continue
                else:
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
    

    # Save metadata as csv file
    data_path = os.path.join(os.path.expanduser('~'),'data_usal','02_intermediate')
    os.makedirs(os.path.expanduser(data_path), exist_ok=True)
    dicom_meta_path = os.path.join(data_path,'dicom_metadata.csv')
    if not os.path.isfile(dicom_meta_path): # create new file if it does not exist
        print('Creating new metadata file')
        df_dedup_goodvals_short.to_csv(dicom_meta_path, index=False)
    else: # if file exists append
        df_dedup_goodvals_short.to_csv(dicom_meta_path, mode='a', index=False, header=False)
        
    os.remove('temp.txt')
        
    print('dicom metadata saved for {}'.format(dir_name))

def iterate_through_studies(parentdir):
    """
    This function iterates through all study directories in parentdir and processes
    the dicom metadata of all files contained in the study.

    Parameters:
        parentdir (str): directory path (contains study directories)
        
    Output:
        file (csv): saves to ~/data_usal/02_intermediate/dicom_metadata.csv    
    
    """
    for obj in os.listdir(parentdir):
        if obj.startswith('.'):
            pass
        else:
            get_dicom_metadata(os.path.join(parentdir,obj))