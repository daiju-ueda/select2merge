import os
import pandas as pd
import lib.split2merge as lib
from tqdm import tqdm


#target data setting. 1 should be better.
target_date_1 = '2022-03-01-05-12-29' #This should be higher performance than 2.
target_date_2 = '2022-02-28-13-17-32'
splitted_institution = 'Morimoto'
l_institution = ['Kashiwara','OCU','Habikino']

n_repeat = 100
cutoff_severity_roc = 0.9
cutoff_severity_demographics = 0.9

positive_value = 1.0
negative_value = 0.0

documents_dir = '../documents'

splitted_dir = os.path.join(documents_dir, 'splitted')
selected_dir = os.path.join(documents_dir, 'selected')
merged_dir = os.path.join(documents_dir, 'merged')
source_dir = os.path.join(documents_dir, 'source')
likelihood_csv = 'likelihood_train_val_test.csv'

path_to_likelihood_1 = os.path.join(splitted_dir, target_date_1, likelihood_csv)
path_to_likelihood_2 = os.path.join(splitted_dir, target_date_2, likelihood_csv)
path_source = os.path.join(source_dir, 'source.csv')

hyperparameter = lib.Hyperparameter(os.path.join(splitted_dir, target_date_1))
path_to_split_1 = os.path.join(splitted_dir, target_date_1, hyperparameter['splitted_csv'])
hyperparameter = lib.Hyperparameter(os.path.join(splitted_dir, target_date_2))
path_to_split_2 = os.path.join(splitted_dir, target_date_2, hyperparameter['splitted_csv'])

df_split_1 = pd.read_csv(path_to_split_1, usecols=['filename','id_uniq','age','Index_str','ANNONID','institution']).rename(columns={'Index_str':'Index'})
df_split_2 = pd.read_csv(path_to_split_2, usecols=['filename','id_uniq','age','Index_str','ANNONID','institution']).rename(columns={'Index_str':'Index'})
df_source = pd.read_csv(path_source, usecols=['Filename','Index','PatientID','PatientAge','PatientSex','Institution']).rename(columns={'PatientID':'ANNONID','Institution':'institution','Filename':'filename'}).drop_duplicates()
df_1 = pd.read_csv(path_to_likelihood_1)
df_2 = pd.read_csv(path_to_likelihood_2)

df_merge_1_pre = pd.merge(df_1, df_split_1, on ='id_uniq', how='left')
df_merge_1 = pd.merge(df_merge_1_pre, df_source, on =['filename', 'Index','ANNONID','institution'], how='left')
df_merge_1.loc[df_merge_1['age'].isna() & df_merge_1['PatientAge'].notna(), 'age'] = df_merge_1['PatientAge'].str.replace('Y','').astype('float')

df_merge_2_pre = pd.merge(df_2, df_split_2, on ='id_uniq', how='left')
df_merge_2 = pd.merge(df_merge_2_pre, df_source, on =['filename','Index','ANNONID','institution'], how='left')
df_merge_2.loc[df_merge_2['age'].isna() & df_merge_2['PatientAge'].notna(), 'age'] = df_merge_2['PatientAge'].str.replace('Y','').astype('float')


Libraries = lib.Library(df_merge_1)
df_all_roc_1 = Libraries.roc_calculator(df_merge_1)
Libraries = lib.Library(df_merge_2)
df_all_roc_2 = Libraries.roc_calculator(df_merge_2)

#Splitted institution
df_splitted_institution_1 = df_merge_1.query('institution == @splitted_institution').query('split != "test"')
df_splitted_institution_2 = df_merge_2.query('institution == @splitted_institution').query('split != "test"')
df_splitted_institution = pd.concat([df_splitted_institution_1, df_splitted_institution_2])
df_splitted_institution['split'] = 'test'
target_test = splitted_institution + '_test'
target_test_new = splitted_institution + '_test_new'
target_test_orig = splitted_institution + '_test_orig'


Libraries = lib.Library(df_splitted_institution)
df_splitted_institution_roc_crossvalid = Libraries.roc_calculator(df_splitted_institution)[[target_test]]
df_splitted_institution_roc_orig = df_all_roc_1[[target_test]].rename(columns={target_test:target_test_orig})
df_splitted_institution_comparison_roc = pd.concat([df_splitted_institution_roc_orig, df_splitted_institution_roc_crossvalid], axis=1).rename(columns={target_test:target_test_new})
df_splitted_institution_demographics = Libraries.demographics_calculator(df_splitted_institution)

dest_dir = os.path.join(selected_dir, splitted_institution, 'merged')
os.makedirs(dest_dir,exist_ok=True)
df_splitted_institution.to_csv(dest_dir + '/resplit.csv', index=False)
df_splitted_institution_comparison_roc.to_csv(dest_dir + '/roc_comparison.csv')
df_splitted_institution_demographics.to_csv(dest_dir + '/demographics.csv')

#Other institutions
##ROC
for institution in l_institution:
    print(institution)
    target_col = institution + '_test'
    target_col_orig = institution + '_test_orig'
    target_col_new = institution + '_test_new'
    df_institution_roc_orig = df_all_roc_1[[target_col]].rename(columns={target_col:target_col_orig})
    df_institution = df_merge_1.query('institution == @institution').drop(columns=['split'])
    num = 0
    while num < 5:
        df_institution_resplit = lib.institutional_resplit(df_institution)
        Libraries = lib.Library(df_institution_resplit)
        df_institution_resplit_roc = Libraries.roc_calculator(df_institution_resplit)[[target_col]].rename(columns={target_col:target_col_new})
        df_institution_comparison_roc = pd.concat([df_splitted_institution_roc_crossvalid, df_institution_roc_orig, df_institution_resplit_roc], axis=1)
        ##Demographics
        df_institution_resplit_demographics = Libraries.demographics_calculator(df_institution_resplit)
        df_institution_resplit_demographics['train/val'] = df_institution_resplit_demographics['train'] + df_institution_resplit_demographics['val']
        df_institution_resplit_demographics['testx9'] = df_institution_resplit_demographics['test'] * 9
        df_institution_resplit_demographics.at['mean age','testx9'] = df_institution_resplit_demographics.at['mean age','test'] *2
        df_institution_resplit_demographics['ratio'] = df_institution_resplit_demographics['testx9'] / df_institution_resplit_demographics['train/val']
        total_rocs = len(df_institution_comparison_roc)
        total_demographics = len(df_institution_resplit_demographics)
        cutoff_roc = total_rocs * cutoff_severity_roc
        cutoff_demographics = total_demographics * cutoff_severity_demographics
        ##csv
        diff_tests = df_institution_comparison_roc[target_test].astype('float') - df_institution_comparison_roc[target_col_new].astype('float')
        range_demographics = (df_institution_resplit_demographics['ratio'] < 1.2) & (df_institution_resplit_demographics['ratio'] > 0.8)
        if ((diff_tests > 0).sum() < cutoff_roc) & (range_demographics.sum() >= cutoff_demographics):
            num = num + 1
            str_num = str(num).zfill(4)
            dest_dir = os.path.join(selected_dir, institution, str_num)
            os.makedirs(dest_dir,exist_ok=True)
            df_institution_resplit.to_csv(dest_dir + '/resplit.csv', index=False)
            df_institution_comparison_roc.to_csv(dest_dir + '/roc_comparison.csv')
            df_institution_resplit_demographics.to_csv(dest_dir + '/demographics.csv')
        else:
            num = num