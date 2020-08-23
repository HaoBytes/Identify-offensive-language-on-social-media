#!/usr/bin/env python3
#coding:utf-8


'''
data

'''



import os
import sys
import re
import pandas as pd

#os.environ["model= 'xxlarge' --do_train=1 --do_predict=0 --file_pre='a'"] = "0,1,2"

dat_path = '../ALbert/data/'

train_file = os.path.join(dat_path, 'training.csv')

tests = ['testset-levela.tsv.csv',
         'testset-levelb.tsv.csv',
         'testset-levelc.tsv.csv',
         ]


# Process Testset, 
# field：id,tweet
# 
def process_test_dat (filename):
    df = pd.read_csv(filename)
    df['label'] = 0
    df['tweet'] = df['tweet'].str.replace(r'@user', '') #, regex=True
    df = df[['tweet','label']]
    nfilename = os.path.splitext(filename)[0]
    df.to_csv(nfilename, header=None, sep='\t', index=False)
    print('save to file: %s' % nfilename)

# Process all test data
def ProcessAllTest ():
    for fn in tests:
        test_file = os.path.join(dat_path, fn)
        print('preprocess testset file: %s' % test_file)
        process_test_dat(test_file)


# Split dataset train:test:val = 8:1:1
def splitdataset (df, file_pre='dat' ):
    print('processing... file: ', file_pre)
    df = df.sample(frac=1.0)  # random

    cut_idx = int(round(0.2 * df.shape[0]))
    df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]

    #print('train records:', df_train.shape[0])
    df_train.to_csv(file_pre + '_train.tsv', header=None, sep='\t', index=False)

    #再拆分test和val
    cut_idx = int(round(0.5 * df_test.shape[0]))
    df_test, df_val = df_test.iloc[:cut_idx], df_test.iloc[cut_idx:]

    df_val.to_csv(file_pre + '_val.tsv', header=None, sep='\t', index=False)
    df_test.to_csv(file_pre + '_test.tsv', header=None, sep='\t', index=False)
    print('Records  train:test:val = %d:%d:%d ' % 
        (df_train.shape[0], df_test.shape[0], df_val.shape[0]))
    print('-'*40)

##Specifies that the column is mapped to the new column by the characteristic index value，newColumn
def MapNewColumn(df, oldcol, newcol, isdrop=1 , workpath = './'):
    A = df[oldcol].value_counts().argsort()
    print('[%s]Column value distribution:' % oldcol)
    dict_oldcol = {'index':A.index,'values':A.values}
    df_oldcol = pd.DataFrame(dict_oldcol)
    df_oldcol.to_csv( os.path.join(workpath, 'MapNewColumn_%s.csv' % oldcol) )
    print('[%s]Column value distribution saved' % oldcol)
    # -----
    df[newcol] = df[oldcol].map(A)
    if isdrop:
        df.drop(oldcol, axis=1, inplace=True)
    return df


# Processing training data
def ProcessAllTrain():
    
    # The field name： id,subtask_a,sbutask_b,subtask_c,tweet
    # Read CSV file
    nfilename = os.path.splitext(train_file)[0]
    
    print('process training file'.center(40,'-'))
    df_train = pd.read_csv(train_file)
    df_train['tweet'] = df_train['tweet'].str.replace(r'@user', '')
    #print(df_train.head())

    df_a = df_train[['tweet','subtask_a']].copy()
    df_a = MapNewColumn(df_a, 'subtask_a', 'subtask_a_id', isdrop=1 , workpath=dat_path)
    splitdataset(df_a, file_pre= nfilename+'_a')

    df_b = df_train[['tweet','sbutask_b']].copy()
    df_b.dropna(subset=['sbutask_b'], inplace=True)
    df_b = MapNewColumn(df_b, 'sbutask_b', 'subtask_b_id', isdrop=1 , workpath=dat_path)
    splitdataset(df_b, file_pre= nfilename+'_b')

    df_c = df_train[['tweet','subtask_c']].copy()
    df_c.dropna(subset=['subtask_c'], inplace=True)
    df_c = MapNewColumn(df_c, 'subtask_c', 'subtask_c_id', isdrop=1 , workpath=dat_path)
    splitdataset(df_c, file_pre= nfilename+'_c')

    print('All training file saved.')
if __name__ == '__main__':
    pass
    ProcessAllTest()
    ProcessAllTrain()
