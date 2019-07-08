# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:14:55 2019

@author: court
"""
#TODO import open_config_file from D00_utils
#TODO import create_connection from D00_utils

def clean_age_height_weight_bmi(df_summary_table):
    '''
    Takes the summary table in a dataframe 
    returns cleaned_summary_table
    '''
    # clean age, patientweight, patientheight column
    column_names_to_clean = ['age', 'patientweight', 'patientheight']
    for column in column_names_to_clean:
        df_summary_table[column] = df_summary_table[column].replace('', 1) #Replace blanks in the column with 1
        df_summary_table[column] = df_summary_table[column].str.replace(',', '.') #Replace comma in the column with decimal points
        df_summary_table[column] = df_summary_table[column].fillna(1)
        #print('Column: {} has been cleaned'.format(column))
    
    #convert each column to correct type
    if df_summary_table['age'].dtype != 'int64':
         df_summary_table['age'] = df_summary_table['age'].astype('int64')
    if df_summary_table['patientweight'].dtype != 'float64':
         df_summary_table['patientweight'] = df_summary_table['patientweight'].astype('float64')
    if df_summary_table['patientheight'].dtype != 'float64':
         df_summary_table['patientheight'] = df_summary_table['patientheight'].astype('float64')
    
    #Remove outliers based on boxplot
    column_names_to_clean = ['age', 'patientweight', 'patientheight']
    for column in column_names_to_clean:
        boxplot = plt.boxplot(df_summary_table[column])
        outlier_min, outlier_max = [item.get_ydata()[0] for item in boxplot['caps']]
        df_summary_table[column] = df_summary_table[column].apply(lambda x: 1 if x > outlier_max else x)
        df_summary_table[column] = df_summary_table[column].apply(lambda x: 1 if x < outlier_min else x)

    #create BMI column (formula from https://www.cdc.gov/nccdphp/dnpao/growthcharts/training/bmiage/page5_1.html)
    df_summary_table['bmi'] = df_summary_table.apply(lambda x: ((x.patientweight/x.patientheight/x.patientheight)*10000), axis=1)
    
    #clean BMI column outliers
    boxplot = plt.boxplot(df_summary_table['bmi']);
    outlier_min, outlier_max = [item.get_ydata()[0] for item in boxplot['caps']]
    df_summary_table['bmi'] = df_summary_table['bmi'].apply(lambda x: 1 if x > outlier_max else x)
    df_summary_table['bmi'] = df_summary_table['bmi'].apply(lambda x: 1 if x < outlier_min else x)
    
    return df_summary_table

def clean_gender(df_summary_table):
    '''
    Takes the summary table in a dataframe 
    returns cleaned_summary_table with all blank rows replaced with 'U' for unsure
    '''
    df_summary_table['gender'] = df_summary_table['gender'].replace('', 'U')
    return df_summary_table

def clean_findingcodes(df_summary_table):
    '''
    Takes the summary table in a dataframe 
    returns cleaned_summary_table with findingcode string converted to a list
    '''
    df_summary_table['findingcode'] = df_summary_table['findingcode'].apply(lambda x: x.split(","))
    return df_summary_table

def clean_all(df_summary_table):
    '''
    Produces the clean summary table
    '''
    cleaned_age_height_weight_bmi_summary_table = clean_age_height_weight_bmi(df_summary_table)
    cleaned_age_height_weight_bmi_gender_summary_table = clean_gender(cleaned_age_height_weight_bmi_summary_table)
    cleaned_age_height_weight_bmi_gender_findingcodes_summary_table = clean_findingcodes(cleaned_age_height_weight_bmi_gender_summary_table)
    return cleaned_age_height_weight_bmi_gender_findingcodes_summary_table

if __name__ == '__main__':
    # Open config file and create connections
    configuration = open_config_file()
    connection = create_connection(configuration);
    
    #get dataframe
    table_name ='DM_Spain_VIEW_study_summary'
    query = ("""
        select * 
        from {};
        """).format(table_name)
    
    df_summary_table = pd.read_sql(query, connection)
    clean_all(df_summary_table)
