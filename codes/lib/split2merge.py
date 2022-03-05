import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

positive_value = 1.0

class Hyperparameter():
    def __init__(self, target_dir):
        hyperparameter_path = os.path.join(target_dir, 'hyperparameters.csv')
        df = pd.read_csv(hyperparameter_path).set_index('option')
        self.splitted_csv = df.at['csv_name','value']

    def __getitem__(self, x):
        splitted_csv = self.splitted_csv
        if x == 'splitted_csv':
            return splitted_csv

class Selected():
    def __init__(self, target_dir):
        eyechecked_path = os.path.join(target_dir, 'eyechecked.csv')
        df = pd.read_csv(eyechecked_path).set_index('institution')
        self.df = df

    def __getitem__(self, x):
        df = self.df
        selected_dir = df.at[x,'selected_dir']
        return selected_dir

class Library():

    def __init__(self, df):
        self.df = df
        self.l_label = list(df.columns[df.columns.str.startswith('output_')])
        self.l_input = list(df.columns[df.columns.str.startswith('input_')])
        self.l_etiology = list(df.columns[df.columns.str.startswith('etiology_')])
        self.l_pred = ['pred_' + s + '_' + str(positive_value) for s in self.l_label]
        self.l_institutions = list(df['institution'].unique())
        self.l_splits = list(df['split'].unique())

    def roc_calculator(self, df):
        l_institutions = self.l_institutions
        l_splits = self.l_splits
        l_label = self.l_label
        l_pred = self.l_pred
        d_pair_roc = dict(zip(l_label,l_pred))
        df_all_roc = pd.DataFrame()
        for institution in l_institutions:
            df_institution = df.query('institution == @institution')
            df_institution_roc = pd.DataFrame()
            for split in l_splits:
                df_split = df_institution.query('split == @split')
                s_label_roc = pd.Series(dtype='float64')
                for k,v in d_pair_roc.items():
                    y_true = df_split[k]
                    if y_true.sum() == 0:
                        s_tmp_roc = pd.Series([np.nan], index=[k])
                    else:
                        y_scores = df_split[v]
                        auc = roc_auc_score(y_true, y_scores)
                        str_auc = str(round(auc,2))
                        s_tmp_roc = pd.Series([str_auc], index=[k])
                        s_label_roc = pd.concat([s_label_roc, s_tmp_roc])
                df_institution_roc[institution + '_' + split] = s_label_roc
            df_all_roc = pd.concat([df_all_roc, df_institution_roc], axis=1)
        return df_all_roc

    def demographics_calculator(self, df):
        l_label = self.l_label
        l_input = self.l_input
        l_etiology = self.l_etiology
        l_splits = self.l_splits
        df_split_demographics = pd.DataFrame()
        for split in l_splits:
            df_split = df.query('split == @split')
    
            Examination = df_split['Index'].nunique()
            Patients = df_split['ANNONID'].nunique()
            Sex_M = df_split['PatientSex'].value_counts()['M']
            Sex_F = df_split['PatientSex'].value_counts()['F']
            Age_mean = round(df_split['age'].describe()['mean'])
            s_concat_base = pd.Series([Examination,Patients, Sex_M, Sex_F, Age_mean], index=['exam','patient','M','F','mean age'])
            s_concat_target = pd.Series()
            for target in l_label:
                s_concat_target_tmp = pd.Series()
                for i,v in df_split[target].value_counts().items():
                    s_concat_target_tmp[target + '_' + str(i)] = v
                s_concat_target = pd.concat([s_concat_target, s_concat_target_tmp])

            s_concat_input = pd.Series()
            for target in l_input:
                s_concat_input_tmp = pd.Series()
                if df_split[target].nunique() == 2:
                    for i,v in df_split[target].value_counts().items():
                        s_concat_input_tmp[target + '_' + str(i)] = v
                else:
                    s_concat_input_tmp[target + '_' + 'mean'] = df_split[target].describe()['mean']
                    s_concat_input_tmp[target + '_' + 'min'] = df_split[target].describe()['min']
                    s_concat_input_tmp[target + '_' + 'max'] = df_split[target].describe()['max']
                s_concat_input = pd.concat([s_concat_input, s_concat_input_tmp])        

            s_concat_disease = pd.Series()
            for disease in l_etiology:
                no_of_disease = (df_split[disease] == 'Yes').sum()
                s_disease = pd.Series([no_of_disease],index=[disease])
                s_concat_disease = pd.concat([s_concat_disease, s_disease])
            s_split = pd.concat([s_concat_base, s_concat_target, s_concat_input, s_concat_disease])
            df_split_demographics[split] = s_split
        return df_split_demographics

target_colum = 'ANNONID'
def institutional_resplit(df, validation_percent=.1, test_percent=.1, seed=None):
    np.random.seed(seed)
    df_pid_uniq = df[[target_colum]].drop_duplicates()
    perm = np.random.permutation(df_pid_uniq.index)
    m = len(df_pid_uniq.index)
    val_end = int(validation_percent * m)
    test_end = int(test_percent * m) + val_end
    val = df_pid_uniq.loc[perm[:val_end]]
    val['split'] = 'val'
    test = df_pid_uniq.loc[perm[val_end:test_end]]
    test['split'] = 'test'
    train = df_pid_uniq.loc[perm[test_end:]]
    train['split'] = 'train'
    df_concat = pd.concat([train, val, test])
    df_resplit = pd.merge(df, df_concat, on=target_colum, how='left')
    return df_resplit