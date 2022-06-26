import pandas as pd
from ltr import train_ltr
from evaluation import add_gold_column
import os
import argparse
from utils import find_query_copies, grade_by_age
from main import read_in_data

parser = argparse.ArgumentParser()
parser.add_argument("--model_save", required=True)
parser.add_argument("--features", required=True)
args = parser.parse_args()

source_filepath = '../data/data_dict.json'
curriculums = 'ICSE,CBSE,Cambridge,English,CCSS,NGSS,Scotland'
age_filepath = '../data/MASTER Reading levels and age filter settings (pwd 123456).xlsx'
data_dict = read_in_data(source_filepath)
query_copies = find_query_copies(data_dict)
age_to_grade = grade_by_age(curriculums,age_filepath)

for random_seed in [42,7,13]:

    ranking4training = f'../results/train_paraphrase-sbert-label-title-rankingloss-nodup_{random_seed}_top30_doc_title.csv'
    ranking4dev = f'../results/dev_paraphrase-sbert-label-title-rankingloss-nodup_{random_seed}_top30_doc_title.csv'

    if not os.path.isfile(ranking4training):
        raise Exception(f'{ranking4training} not found. Please make sure to add the predictions needed for training re-ranker')

    train_pred = pd.read_csv(ranking4training,
                            sep='\t', dtype= {'TARGET_ID': str,
                                            'TARGET_GRADEID': str,
                                            'SOURCE_ID': str,
                                            'SOURCE_GRADEID': str})
if 'GOLD' not in train_pred.columns:
    train_gold = pd.read_csv(f'../data/train_query_pairs_{random_seed}.csv', sep='\t', dtype= {'TARGET_ID': str,
                                                                   'TARGET_GRADEID': str,
                                                                   'SOURCE_ID': str,
                                                                   'SOURCE_GRADEID': str})
    train_pred = add_gold_column(train_pred, train_gold, query_copies)
    train_pred.to_csv(ranking4training, sep='\t', index=False)  # save predictions with gold column

if not os.path.isfile(ranking4dev):
    raise Exception(f'{ranking4dev} not found. Please make sure to add the predictions needed for training re-ranker')

dev_pred = pd.read_csv(ranking4dev,sep='\t', dtype={'TARGET_ID': str,
                                                    'TARGET_GRADEID': str,
                                                    'SOURCE_ID': str,
                                                    'SOURCE_GRADEID': str})

if 'GOLD' not in dev_pred.columns:
    dev_gold = pd.read_csv(f'../data/dev_query_pairs_{random_seed}.csv', sep='\t',
                                                dtype={'TARGET_ID': str,
                                              'TARGET_GRADEID': str,
                                              'SOURCE_ID': str,
                                              'SOURCE_GRADEID': str})
    dev_pred = add_gold_column(dev_pred, dev_gold, query_copies)
    dev_pred.to_csv(ranking4dev, sep='\t', index=False)  # save predictions with gold column

train_ltr(train_pred,
          dev_pred,
          data_dict,
          args.model_save,
          random_seed,
          age_to_grade,
          args.features,
          query_copies)