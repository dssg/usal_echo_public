import pandas as pd
import glob
import os

def get_dicom_metadata(dirpath):
    """Get all dicom tags for files in dirpath.
    
    This function uses gdcmdump to retrieve the metadata tags of all files in dirpath.
    The tags are written to a csv file with header=['dirname','filename','tag1','tag2','value']
    
    Parameters:
        dirpath (str): directory path
        
    Output:
        file (csv): saves to ~/data_usal/dicom_metadata.csv
    """
    
    # Dump metadata of all files in study directory to temp.txt
    os.system('gdcmdump '+ dirpath +' > temp.txt')
    dir_name = os.path.basename(dirpath)
    
    # Get list of all files in directory
    all_files = os.listdir(dir_name)
    dicom_files = [dcm for dcm in all_files if dcm.endswith('.dcm')]
    dicom_files.sort()
    
    temp_file='temp.txt'
    meta = []
    file_iterator = -1 # needed to associate filename with metadata
    with open(temp_file, 'r') as f:
        line_meta = []
        try:
            for one_line in f:
                file_name = dicom_files[file_iterator]
                clean_line = one_line.replace(']','').strip()
                if "# Dicom-File-Format" in clean_line: # check for new header
                    file_iterator += 1 # increase if header is encountered
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
        except Exception as e:
            print(e, dir_name, file_name, one_line)
                    
    df = pd.DataFrame.from_records(meta, columns=['dirname','filename','tag1','tag2','value'])

    os.makedirs(os.path.expanduser('~/data_usal'), exist_ok=True)
    if not os.path.isfile(file_name): # create new file if it does not exist
        df.to_csv('~/data_usal/dicom_metadata.csv', index=False)
    else: # if file exists append
        df.to_csv('~/data_usal/dicom_metadata.csv', mode='a', index=False, header=False)
        
    return('dicom metadata saved for {}'.format(dir_name))
