#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:50:24 2019

@author: ftahmasebian
"""
import fixed_data as cfg 
import pandas as pd
import numpy as np
import re, os
import urllib.request
import shutil, requests
import re, time
import string
from pandas.api.types import is_string_dtype

import nltk
nltk.download('punkt')

#from nltk.corpus import stopwords
#import seaborn as sns
#import data_loader
from sklearn.model_selection import train_test_split



"""
The all unique words in description are: 53760
the shape of raw_data is (148955, 4)
number of unique company are :3669 
number of unique titles are :96 
test data and train data is splited
validation data and train data is splited
num of rows in training is :95331 
num of rows in validation is :23833  
num of rows in testing is :29791  
"""

def split_data(X,y):
    x_train_data, x_test_data, y_train, y_test = train_test_split(X, y,test_size= 0.2, random_state=101)
    return x_train_data, x_test_data, y_train, y_test 


def load_data_url(url, file_name):
    r = requests.get(url, allow_redirects=True)
    print (r.headers.get('content-type'))
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


def exploit_data(df):
    tmp_df = df[cfg.ACTIVITY_COLUMNS].str.split('.')
    df[cfg.ACTIVITY_COLUMNS] = tmp_df
    
    #tmp_df = df[cfg.ACTIVITY_COLUMNS].apply(lambda x: [x])
    #df[cfg.ACTIVITY_COLUMNS] = tmp_df
    
    #span = 2
    print("it is exploiting ............................................")
    span = 1
    
    
    #tmp_df = df[cfg.ACTIVITY_COLUMNS].str.join(".")
    #df[cfg.ACTIVITY_COLUMNS] = tmp_df
#    from time import sleep
##
#    l = len(df[cfg.ACTIVITY_COLUMNS])
#    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

#    for j in range(len(df[cfg.ACTIVITY_COLUMNS])):
#        df[cfg.ACTIVITY_COLUMNS].iloc[j] = [".".join(df[cfg.ACTIVITY_COLUMNS][j][i:i+span]) for i in range(0, len(df[cfg.ACTIVITY_COLUMNS][j]), span)]
#        sleep(0.1)
#        # Update Progress Bar
#        printProgressBar(j + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

    print("it is melting ............................................")
    """
    reshaped = \
    (df.set_index(df.columns.drop(cfg.ACTIVITY_COLUMNS,1).tolist())
       .activities.str.split('.', expand=True)
       .stack()
       .reset_index()
       .rename(columns={2:cfg.ACTIVITY_COLUMNS})
       .loc[:, df.columns]
    )
    """
    print(df[cfg.ACTIVITY_COLUMNS].iloc[0])

    df = df[cfg.ACTIVITY_COLUMNS].apply(pd.Series) \
        .merge(df, right_index = True, left_index = True) \
        .drop([cfg.ACTIVITY_COLUMNS], axis = 1) \
        .melt(id_vars = [cfg.TITLE_COLUMN,cfg.COMPNY_COLUMN], value_name = cfg.ACTIVITY_COLUMNS) \
        .drop("variable", axis = 1) \
        .dropna()
   
    df[cfg.ACTIVITY_COLUMNS].replace('', np.nan, inplace=True)
    df = df.dropna()
    print("+++++++++++++")
    print(df[cfg.ACTIVITY_COLUMNS].iloc[0])
    return df


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def main():
    
    orig_data = pd.read_csv(cfg.RAW_FILE)
    print("Shape of original data is {}".format(orig_data.shape))
    #print(orig_data.columns)
    #print(orig_data.info())
    #print(orig_data.describe())
    raw_data = orig_data[cfg.SELECTED_COLUMNS]

    for col in cfg.SELECTED_COLUMNS:
        raw_data = raw_data[pd.notnull(raw_data[col])]
    
    raw_data = raw_data.reset_index(drop=True)
    
    for col in cfg.SELECTED_COLUMNS:
        if is_string_dtype(raw_data[col]):
            raw_data[col] = raw_data[col].str.lower()
    
    
    replaces = [ "]", "[", "./li", ". e.","(. e)"]
    #import ast
    
    #tmp_df = raw_data[cfg.ACTIVITY_COLUMNS].apply(lambda x: ast.literal_eval(x))
    
    
    for elem in replaces:
        tmp_df = raw_data[cfg.Review_COLUMNS].apply(lambda x: str(x).replace(elem,""))
        raw_data[cfg.Review_COLUMNS] = tmp_df


    tmp_df = raw_data[cfg.Review_COLUMNS].apply(lambda x: str(x).replace("etc.","etc"))
    raw_data[cfg.Review_COLUMNS] = tmp_df
     
    tmp_df = raw_data[cfg.Review_COLUMNS].apply(lambda x: str(x).replace("e. g.","for example"))
    raw_data[cfg.Review_COLUMNS] = tmp_df
        
    tmp_df = raw_data[cfg.Review_COLUMNS].apply(lambda x: str(x).replace("Inc.","Inc"))
    raw_data[cfg.Review_COLUMNS] = tmp_df
    
    tmp_df = raw_data[cfg.Review_COLUMNS].apply(lambda x: str(x).replace("U. S.","US"))
    raw_data[cfg.Review_COLUMNS] = tmp_df
    
    tmp_df = raw_data[cfg.Review_COLUMNS].apply(lambda x: str(x).replace("vs.","vs"))
    raw_data[cfg.Review_COLUMNS] = tmp_df
    
#    tmp_df = raw_data[cfg.ACTIVITY_COLUMNS].apply(lambda x: str(x).replace("a.m.","am"))
#    raw_data[cfg.ACTIVITY_COLUMNS] = tmp_df
#    
#    tmp_df = raw_data[cfg.ACTIVITY_COLUMNS].apply(lambda x: str(x).replace("p.m.","pm"))
#    raw_data[cfg.ACTIVITY_COLUMNS] = tmp_df
    
    #(s) remove it 
#    
    tmp_df = raw_data[cfg.Review_COLUMNS].apply(lambda x: re.sub("[.]", "", str(x)))
    raw_data[cfg.Review_COLUMNS] = tmp_df
    
    
    tmp_df = raw_data[cfg.Review_COLUMNS].apply(lambda x: re.split('\",\"',str(x)))
    raw_data[cfg.Review_COLUMNS] = tmp_df
    tmp_df = raw_data[cfg.Review_COLUMNS].str.join(". ")
    raw_data[cfg.Review_COLUMNS] = tmp_df
    tmp_df = raw_data[cfg.Review_COLUMNS].apply(lambda x: str(x).strip('"'))
    raw_data[cfg.Review_COLUMNS] = tmp_df


    tmp_df = raw_data[cfg.Review_COLUMNS].apply(lambda x: x[0:-1] if x[-1] == '.' else x)
    raw_data[cfg.Review_COLUMNS] = tmp_df
    
    
    
#    from time import sleep
#
#    l = len(raw_data[cfg.ACTIVITY_COLUMNS])
#    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
#
#    for i in range(len(raw_data[cfg.ACTIVITY_COLUMNS])):
#        #tmp_df  = ""
#        #raw_data[cfg.ACTIVITY_COLUMNS][i] = re.split('\",\"' ,raw_data[cfg.ACTIVITY_COLUMNS][i])
#    #tmp_df = raw_data[cfg.ACTIVITY_COLUMNS].apply(lambda x: str(x).replace(','," "))
#        #tmp_df = ". ".join(raw_data[cfg.ACTIVITY_COLUMNS][i])
#        #tmp_df = tmp_df.strip('"')
#        if tmp_df[-1] == '.':
#            tmp_df = tmp_df[0:-1]
#        if len(tmp_df.split('.')) > 10: 
#            print(tmp_df)
#        raw_data[cfg.ACTIVITY_COLUMNS][i] = tmp_df
#        sleep(0.1)
#        # Update Progress Bar
#        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
#
#    print("--- %s seconds ---" % (time.time() - start_time))
#
    #raw_data[cfg.COMPNY_COLUMN] = raw_data[cfg.COMPNY_COLUMN].apply(lambda x: x.strip('"'))
    
    print("static before exploit data")
    
    #raw_data = exploit_data(raw_data)
    #gb = raw_data.groupby([cfg.TITLE_COLUMN, cfg.COMPNY_COLUMN]) 
    #groups = dict(list(gb))
    #print("before")
    #print(len(groups['Warehouse Worker (Transportation and Material Moving)', 'The Coca-Cola Company'][cfg.ACTIVITY_COLUMNS]))
    #print("set")
    #print(len(set(groups['Warehouse Worker (Transportation and Material Moving)', 'The Coca-Cola Company'][cfg.ACTIVITY_COLUMNS])))
    #set(groups['Warehouse Worker (Transportation and Material Moving)', 'The Coca-Cola Company']['activities'])
    for col in cfg.SELECTED_COLUMNS:
        if is_string_dtype(raw_data[col]):
            raw_data[col] = raw_data[col].apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
    
    
    
    
#    raw_data[cfg.TITLE_COLUMN] = raw_data[cfg.TITLE_COLUMN].apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
#    raw_data[cfg.COMPNY_COLUMN] = raw_data[cfg.COMPNY_COLUMN].apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
#    raw_data[cfg.ACTIVITY_COLUMNS] = raw_data[cfg.ACTIVITY_COLUMNS].apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
#
    
    len_all_data = len(raw_data)
    
    duplicateRowsDF = raw_data[raw_data.duplicated()]
    duplicateRowsDF.to_csv("duplicate_rows.csv")
    
    
    raw_data = raw_data.drop_duplicates()
    

    """
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(raw_data[cfg.ACTIVITY_COLUMNS])
    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    raw_data[cfg.ACTIVITY_COLUMNS] = "".join(stripped)
    
    tokens = word_tokenize(raw_data[cfg.COMPNY_COLUMN])
    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    raw_data[cfg.COMPNY_COLUMN] = "".join(stripped)


    tokens = word_tokenize(raw_data[cfg.TITLE_COLUMN])
    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    raw_data[cfg.TITLE_COLUMN] = "".join(stripped)

    """ 
    
    print("number of duplication is {}".format(len_all_data - len(raw_data)))
    
#    gb = raw_data.groupby([cfg.TITLE_COLUMN, cfg.COMPNY_COLUMN]) 
#    groups = dict(list(gb))
#    #print("after")
#    print(len(groups['Warehouse Worker (Transportation and Material Moving)', 'The Coca-Cola Company'][cfg.ACTIVITY_COLUMNS]))
    #print("set")
#    print(len(set(groups['Warehouse Worker (Transportation and Material Moving)', 'The Coca-Cola Company'][cfg.ACTIVITY_COLUMNS])))
    
    
    raw_data['cnt_words'] = raw_data[cfg.Review_COLUMNS].apply(lambda x: len(str(x).split(' ')))
    

    # Structure analysis
    """
    num_words = raw_data[cfg.ACTIVITY_COLUMNS].apply(lambda x: len(x.split()))
    num_words_mean, num_words_std = np.mean(num_words), np.std(num_words)
    print("The mean and std of words are {} and {}.".format(num_words_mean,num_words_std ))
    num_sentences = raw_data[cfg.ACTIVITY_COLUMNS].apply(lambda x: len(re.split( '~ ...' ,'~'.join(x.split('.')))))
    num_sentences_mean = np.mean(num_sentences)
    print("The mean number of sentences is {}.".format(num_sentences_mean))
    """
    #all unique word in dictionary
    all_words = list(raw_data[cfg.Review_COLUMNS].str.lower().str.split(' ', expand=True).stack().unique())
    print("The all unique words in description are: {}".format(len(all_words)))
    print("some words are:")
    print(all_words[1:20])
    
    #To remove all rows where column 'cnt_words' is < cut_off_word:
    raw_data.drop(raw_data[raw_data.cnt_words < cfg.CUT_OFF_WORDS].index, inplace = True)
    
    #print("the shape of raw_data is {}".format(raw_data.shape))
    #assert raw_data[cfg.COMPNY_COLUMN].isnull().count() == len(raw_data[cfg.COMPNY_COLUMN])
    #assert raw_data[cfg.TITLE_COLUMN].isnull().count() == len(raw_data[cfg.TITLE_COLUMN])

    print("number of unique company are :{} ".format(raw_data[cfg.COMPNY_COLUMN].nunique()))
    print("number of unique titles are :{} ".format(raw_data[cfg.CATEGORY_COLUMN].nunique()))

    
    x_train_data, x_test_data, y_train, y_test = split_data(raw_data[cfg.X_COLUMNS],raw_data[cfg.TARGET_COLUMN])
    
    print("test data and train data is splited")

    x_train_data, x_val_data, y_train, y_val = split_data(x_train_data, y_train)
    
    print("validation data and train data is splited")
    
    x_train_data.to_csv(cfg.TRAIN_X_FILE)
    y_train.to_csv(cfg.TRAIN_Y_FILE)
    
    x_val_data.to_csv(cfg.VAL_X_FILE)
    y_val.to_csv(cfg.VAL_Y_FILE)

    x_test_data.to_csv(cfg.TEST_X_FILE)
    y_test.to_csv(cfg.TEST_Y_FILE)
    


    print("num of rows in training is :{} ".format(len(x_train_data)))
    print("num of rows in validation is :{}  ".format(len(x_val_data)))
    print("num of rows in testing is :{}  ".format(len(x_test_data)))

    

    
    """
    #corpus = data_loader.WordCorpus(cfg.BASE_DIR, freq_cutoff=cfg.unk_threshold, verbose=True)
    #N = len(corpus.word_dict)    
    """
def statics_data(): 
    train_data = pd.read_csv(cfg.TRAIN_X_FILE)
    print("Shape of original data is {}".format(train_data.shape))
    print("columns", train_data.columns)
    print(train_data.info())
    print(train_data.describe())
    print("number of unique company are :{} ".format(train_data[cfg.COMPNY_COLUMN].nunique()))
    print("number of unique titles are :{} ".format(train_data[cfg.TITLE_COLUMN].nunique()))
    cmp_train = set(train_data[cfg.COMPNY_COLUMN].unique().tolist())
    ttl_train = set(train_data[cfg.TITLE_COLUMN].unique().tolist())
    
    train_data_Y = pd.read_csv(cfg.TRAIN_Y_FILE)
    
    
    train_num_words = list(train_data_Y[cfg.ACTIVITY_COLUMNS].str.lower().str.split(' ', expand=True).stack().unique())
    print("The all unique words in train activity are: {}".format(len(train_num_words)))

    
    test_data = pd.read_csv(cfg.TEST_X_FILE)
    test_data_Y = pd.read_csv(cfg.TEST_Y_FILE)

    print("Shape of original data is {}".format(test_data.shape))
    
    
    print("test data:number of unique company are :{} ".format(test_data[cfg.COMPNY_COLUMN].nunique()))
    print("test data:number of unique titles are :{} ".format(test_data[cfg.TITLE_COLUMN].nunique()))
    cmp_test = set(test_data[cfg.COMPNY_COLUMN].unique().tolist())
    ttl_test = set(test_data[cfg.TITLE_COLUMN].unique().tolist())



    train_num_words = list(train_data_Y[cfg.ACTIVITY_COLUMNS].str.lower().str.split(' ', expand=True).stack().unique())
    print("The all unique words in description are: {}".format(len(train_num_words)))


    test_num_words = list(test_data_Y[cfg.ACTIVITY_COLUMNS].str.lower().str.split(' ', expand=True).stack().unique())
    print("The all unique words in description are: {}".format(len(test_num_words)))
    
    
    print("The all unique words in description are: {}".format(len(set(train_num_words) - set(test_num_words))))
    

    company_intersect = set.intersection(cmp_train, cmp_test)
    title_intersect = set.intersection(ttl_test, ttl_train)
    print("len intersect company {} and title {}".format(len(company_intersect), len(title_intersect) ))
    
    print("difference of company name is:" )
    #print(list(cmp_train-cmp_test))
    
    gb_train = train_data.groupby([cfg.TITLE_COLUMN, cfg.COMPNY_COLUMN])
    counts = gb_train.size().to_frame(name='counts')
    counts.to_csv('train_static_cnt.csv')
    gb_test = test_data.groupby([cfg.TITLE_COLUMN, cfg.COMPNY_COLUMN])
    counts = gb_test.size().to_frame(name='counts')
    counts.to_csv('test_static_cnt.csv')

    """
    (counts
    .join(gb_raw.agg({'col3': 'mean'}).rename(columns={'col3': 'col3_mean'}))
    .join(gb_raw.agg({'col4': 'median'}).rename(columns={'col4': 'col4_median'}))
    .join(gb_raw.agg({'col4': 'min'}).rename(columns={'col4': 'col4_min'}))
    .reset_index()
    )
    """
def get_top_titles(): 
    df = pd.read_csv(cfg.TOP_TITILES_FILE, sep=",", header=None)
    df.columns = ["id", "title_name"]
    return df['title_name'].tolist()

def generate_test_top_titles():
    top_titles = get_top_titles()
    
    #test_data = pd.read_csv(cfg.TRAIN_X_FILE)
    test_data = pd.read_csv(cfg.TEST_X_FILE)

    test_data.rename( columns={'Unnamed: 0':'tid'}, inplace=True )


    activity_data = pd.read_csv(cfg.TEST_Y_FILE)
    activity_data.rename( columns={'Unnamed: 0':'tid'}, inplace=True )
    
    print("Shape of original data is {}".format(test_data.shape))
    print(test_data.info())
    print(test_data.describe())
    
    #gb_activity = activity_data.groupby(['tid'])
    #activity_groups = dict(list(gb_activity))

    gb_test = test_data.groupby([cfg.TITLE_COLUMN])
    groups = dict(list(gb_test))
    
    new_title_test = pd.DataFrame(columns=[cfg.COMPNY_COLUMN, cfg.TITLE_COLUMN])
    new_activity_test = pd.DataFrame(columns=[cfg.ACTIVITY_COLUMNS])
    
    
    for title in top_titles: 
        if title not in groups:
            print("not in testset {}".format(title))
        else:
            cnt_cmp = len(groups[title][cfg.COMPNY_COLUMN])
            sample_size = 4      # sample size
            if cnt_cmp > sample_size:
                df = groups[title][cfg.COMPNY_COLUMN]
                selected_cmp = df.sample(n=sample_size , random_state=7)
            else: 
                selected_cmp = groups[title][cfg.COMPNY_COLUMN]
            
            gb_tc = groups[title].groupby(cfg.COMPNY_COLUMN)
            ttlcmp_groups = dict(list(gb_tc))
            
            for cmp in selected_cmp:
                title_ids = ttlcmp_groups[cmp]['tid']
                idx = title_ids.index
                new_title_test = new_title_test.append(test_data.iloc[idx[0]][1:], ignore_index=True)
                new_activity_test= new_activity_test.append(activity_data.iloc[idx[0]][1:], ignore_index=True)        
    
    new_title_test.to_csv(cfg.sel_TEST_X_FILE, sep=',')
    new_activity_test.to_csv(cfg.sel_TEST_Y_FILE, sep=',')
    return
    

def preprocessing(df):
    pass
    """
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    print(words[:100])
    
    
    #result = re.sub('[\W_]+', '', ini_string) 
    
    # split into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    # stemming of words
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    print(stemmed[:100])
    """
    
if __name__ == '__main__':
    #url = 'https://careerbuilder.sharepoint.com/:x:/r/sites/datascience/Shared%20Documents/AI-Job-Resume/AI-Res%20sample%20data%202.csv?d=w75e95fdf8d1c4372931ed31843f51884&csf=1&e=Mm5Vep'
    #file_name = os.path.join(cfg.BASE_DIR,'airesume.csv')
    #print (is_downloadable(url))
    #load_data_url(url, file_name)
    #main()
    statics_data()
    #generate_test_top_titles()
    
#    df = pd.DataFrame([{'var1': 'myyyyyyyyyy', 'var2': 1, 'var3': 'XX'},
#                   {'var1': 'd,e,f,x,y', 'var2': 2, 'var3': 'ZZ'}])
#
#    print(df)
#    
#    print("++++++++")
#    df['var1_list'] = df['var1'].apply(lambda x: [x])
#    #df['var1'] = [df['var1'].str]
#    print(df)

    
#    reshaped = \
#    (df.set_index(df.columns.drop('var1',1).tolist())
#       .var1.str.split(',', expand=True)
#       .stack()
#       .reset_index()
#       .rename(columns={0:'var1'})
#       .loc[:, df.columns]
#    )
#    
#    print(reshaped)
    
 