import os.path
import json
import pandas as pd
from utils import dump_to_json, generate_shared_docs_set, find_query_copies, grade_by_age, generate_train_test
from utils import find_age, target_subset, generate_doc_sums, generate_info_file
from data_exploration import generate_stats, target_set_stats, data_distribution
from generate_search_space import get_search_space
from match_learning_objectives import find_best_queries
from evaluation import eval_ranking, topn_recall, add_gold_column
import time
import os
from tfidf_baseline import tfidf_match
from collections import defaultdict
from finetune_sbert import fine_tune_bert
from ltr import train_ltr, ltr_infer
from sklearn.model_selection import train_test_split
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


def read_in_data(source_filepath):

    if os.path.isfile(source_filepath):

        if source_filepath != f'../data/data_dict.json':
            dump_to_json(source_filepath)

        with open(f'../data/data_dict.json') as json_file:
            data = json.load(json_file)

    else:
        print(f'{source_filepath} not found. Please make sure the source data is either the dump csv or a json file in the data folder of this repo')
        exit()

    return data


def read_in_target(target_filepath):

    if os.path.isfile(target_filepath):

        target_data = pd.read_csv(target_filepath, sep='\t')
        for column_id in ['TARGET_ID', 'SOURCE_ID', 'TARGET_GRADEID', 'SOURCE_GRADEID']:
            target_data[column_id] = target_data[column_id].astype(str)
            if column_id in {'TARGET_GRADEID', 'SOURCE_GRADEID'}:
                target_data[column_id] = target_data[column_id].map(lambda x: x.strip('.0'))

    else:
        raise Exception(f'{target_filepath} not found')

    return target_data

def save_results(predictions, results_filepath, k):

    # top 5 for inference results
    with open(results_filepath, 'w') as outfile:
        outfile.write(f'TARGET_CURRICULUM\tTARGET\tTARGET_ID\tTARGET_PATH\tSOURCE\tSOURCE_ID\tSOURCE_PATH\tSCORE\n')
        for target_cur, matches in predictions:
            for target_id, target_dict in matches.items():
                target_dict["path"] = target_dict["path"].replace("\n", "")
                target_dict['label'] = target_dict['label'].replace("\n","")
                for source_label, id, path, score in target_dict["scores"][:k]:
                    source_label = source_label.replace("\n","")
                    path = path.replace("\n", "")
                    outfile.write(f'{target_cur}\t{target_dict["label"]}\t{target_id}\t{target_dict["path"]}'
                                  f'\t{source_label}\t{id}\t{path}\t{score}\n')

def main():

    dump_filepath = f'../data/20220215-curriculum-data-export-report-production.csv'
    source_filepath = f'../data/data_dict.json'
    curriculums = 'ICSE,CBSE,Cambridge,English,CCSS,NGSS,Scotland'
    age_filepath = f'../data/MASTER Reading levels and age filter settings (pwd 123456).xlsx'
    target_filepath = f'../data/query_pairs_{curriculums}.csv'
    source_grades = ''
    source_subjects = ''
    filters = {'curriculums': curriculums, 'grades': source_grades, 'subjects': source_subjects}
    random_seeds = [42,13,7]
    k = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=False, default='cosine', choices=['cosine', 're-rank', 'classification'])
    parser.add_argument("--features", required=False, default="")
    parser.add_argument("--mode", required=False, default='test', choices=['dev','test'])
    parser.add_argument("--uncase", required=False, default="False", choices=['False', 'True'])
    parser.add_argument("--features_rerank", required=False, default="topic")
    args = parser.parse_args()

    if args.model not in ['tf-idf','sentence-transformers/paraphrase-MiniLM-L6-v2'] and not os.path.isfile(args.model):
        raise Exception(f'{args.model} not found. Please add the specified model to the /models folder and try again.')

    # exp_params = {'random_seed': [42,13],
    #               'model': [f'{MODEL_SAVE}/paraphrase-sbert-label-title-rankingloss-nodup_'],
    #               'features': ['doc_title'],
    #               'k': 30,
    #               # 'r': 100,
    #               'filter_age': False,
    #               'uncase': False,
    #               'method': 'cosine',
    #               'mode': ['train','dev','test'],
    #               'base_model': "",
    #               're-rank': True,
    #               'higher_layers': ['grade,subject,topic','grade,subject','topic','grade','subject'],
    #               'k2': 5}

    # exp_params = {'random_seed': [13],
    #               'model': ['sentence-transformers/paraphrase-MiniLM-L6-v2'],
    #               'features': [
    #                              # 'doc_title,topic',
    #                              'doc_title,grade'],
    #                              # 'doc_title,subject',
    #                              # 'doc_title,topic,grade',
    #                              # 'doc_title,topic,subject',
    #                              #'doc_title,grade,subject',
    #                              #'doc_title,topic,grade,subject'],
    #                 'k': 5,
    #                 # 'r': 100,
    #                 'filter_age': False,
    #                 'uncase': False,
    #                 'method': 'cosine',
    #                 'mode': 'test',
    #                 'base_model': "",
    #                 're-rank': False}

    # exp_params = {'random_seed': [13],
    #               'model': [f'{MODEL_SAVE}/model_weights.pth'],
    #               'features': ['doc_title,topic,grade,subject'], #'doc_title', 'doc_title,topic', 'doc_title,grade', 'doc_title,topic,grade', 'doc_title,grade,subject'
    #               'k': 5,
    #               # 'r': 100,
    #               'filter_age': False,
    #               'uncase': True,
    #               'method': 'classification',
    #               'mode': 'test',
    #               'base_model': "distilbert-base-uncased",
    #               're-rank': False,
    #               'higher_layers': None}

    # exp_params = {'random_seed': [42,7,13],
    #               'model': ['sentence-transformers/paraphrase-MiniLM-L6-v2',
    #                         f'{MODEL_SAVE}/paraphrase-sbert-label-title-rankingloss-nodup_',
    #                         f'{MODEL_SAVE}/paraphrase-sbert-label-title-sums-rankingloss-nodup_',
    #                         f'{MODEL_SAVE}/paraphrase-sbert-label-title-sumsnsents-rankingloss-nodup_'],
    #                         # f'tf-idf',
    #                         # f'{MODEL_SAVE}/cc.en.300.bin'
    #                         # 'sentence-transformers/paraphrase-MiniLM-L6-v2',
    #                         # f'{MODEL_SAVE}/paraphrase-sbert-label-rankingloss-nodup_'],
    #                         # f'{MODEL_SAVE}/paraphrase-sbert-label-title-rankingloss-nodup_',
    #                         # f'{MODEL_SAVE}/paraphrase-sbert-label-title-sums-rankingloss-nodup_',
    #                         # f'{MODEL_SAVE}/paraphrase-sbert-label-title-sumsnsents-rankingloss-nodup_'],
    #               'features': [''], #'doc_title', 'doc_title,topic', 'doc_title,grade', 'doc_title,topic,grade', 'doc_title,grade,subject'
    #               'k': 5,
    #               # 'r': 100,
    #               'filter_age': False,
    #               'uncase': False,
    #               'method': 'cosine',
    #               'mode': ['test'],
    #               'base_model': "",
    #               're-rank': False,
    #               'higher_layers': None}

    # fine_tune_args = {'pre_trained_model': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
    #                   'batch_size': int(12),
    #                   'epochs': int(3),
    #                   'train_loss': 'ranking',
    #                   'evaluator': 'information_retrieval',
    #                   'source': 'label,doc_titles',
    #                   'parent_nodes': 'grade,subject,topic'}

    print('Reading in data...')
    # read in database
    data_dict = read_in_data(source_filepath)

    # find which queries are copies
    query_copies = find_query_copies(data_dict)

    # generate json with each summaryText of each pinned doc for each query
    if not os.path.isfile(f'../data/doc_sums.csv'):
        generate_doc_sums(data_dict, curriculums)

    # generate csv with queries ids and features
    if not os.path.isfile(f'../data/data.csv'):
        age_to_grade = grade_by_age(curriculums,
                                age_filepath=age_filepath)
        doc_sums_df = pd.read_csv(f'../data/doc_sums.csv', sep='\t', dtype={'queryId': str})
        doc_sums = defaultdict(list)
        for query_id, docs in doc_sums_df.groupby(['queryId']):
            doc_sums[query_id] = [doc_sum for doc_sum in list(docs['sumText'])]
        generate_info_file(f'../data/data_dict.json', f'../data/data.csv', doc_sums, age_to_grade, curriculums)

    # generate train and test sets with search queries
    if not os.path.isfile(f'../data/query_pairs_{curriculums}.csv'):
        generate_shared_docs_set(dump_filepath, curriculums, query_copies)

    # run experiment
    for random_seed in random_seeds:

        if not os.path.isfile(f'../data/{args.mode}_query_pairs_{random_seed}.csv'):
            target_data = read_in_target(target_filepath)
            generate_train_test(target_data, random_seed)

        test = pd.read_csv(f'../data/{args.mode}_query_pairs_{random_seed}.csv', sep='\t', dtype= {'TARGET_ID': str,
                                                                                                'TARGET_GRADEID': str,
                                                                                                'SOURCE_ID': str,
                                                                                                'SOURCE_GRADEID': str})

        results_filepath = f'../results/{args.mode}_{args.model.replace(f"../models/","").replace("sentence-transformers/", "").replace("_"+ str(random_seed),"")}_{random_seed}_top{k}_{args.features}.csv'

        age_to_grade = grade_by_age(curriculums, age_filepath = age_filepath)

        doc_sums = None
        if 'doc_sum_1sent' in args.features or 'doc_sum_nsents' in args.features:
            doc_sums_df = pd.read_csv(f'../data/doc_sums.csv', sep='\t', dtype= {'queryId': str})
            doc_sums = defaultdict(list)
            for query_id, docs in doc_sums_df.groupby(['queryId']):
                doc_sums[query_id] = [doc_sum for doc_sum in list(docs['sumText'])]

        # MATCHING STARTS HERE
        matches_per_cur = []
        for index, cur_group in test.groupby(['TARGET_CURRICULUM']):

            cur_group_dict = cur_group.to_dict(orient='list')
            target_cur = cur_group_dict['TARGET_CURRICULUM'][0]

            start_time = time.perf_counter()

            target = cur_group.drop_duplicates(['TARGET_ID'])
            target = target.to_dict(orient='list')

            if args.method == 'classification':
                target = target_subset(target,50)

            print(f'\nTARGET CURRICULUM: {target_cur}\n'
                  f'N of TARGET LO: {len(target["TARGET"])}\n'
                  f'FILTERS: {filters}\n'
                  f'MODEL: {args.model.replace(f"../models/", "")}\n'
                  f'FEATURES: {args.features}\n'
                  f'K: {k}\n'
                  f'RANDOM_SEED: {random_seed}\n')

            if args.model == 'tf-idf':
                print('Matching queries...')
                matches = tfidf_match(target_cur, curriculums, target, k)

            else:
                print('Generating search space...')
                source = get_search_space(data_dict, filters, target_cur)

                print('Matching queries...')
                if args.method == 're-rank': k = 30
                if args.uncase == 'True': uncase = True
                else: uncase = False
                matches = find_best_queries(source,
                                            target,
                                            args.model,
                                            args.features,
                                            k,
                                            mode=args.method,
                                            age_to_grade=age_to_grade,
                                            doc_sums_dict=doc_sums,
                                            uncased=uncase)

            matches_per_cur.append((target_cur,matches))

            end_time = time.perf_counter()
            print(f"{len(set(cur_group_dict['TARGET_ID']))} search queries matched in {end_time - start_time:0.4f} seconds")

        print('Saving results of ranking...')
        save_results(matches_per_cur,results_filepath,k)

        if args.method == 're-rank':

            k2 = 5
            results_filepath = f'../results/rerank_ltr_{args.mode}_{random_seed}_{args.features_rerank}.csv'
            test = pd.read_csv(f'../results/{args.mode}_{args.model.replace("../models/", "").replace("sentence-transformers/", "")}_top{k}_{args.features}.csv',
                                sep='\t', dtype={'TARGET_ID': str,
                                'TARGET_GRADEID': str,
                                'SOURCE_ID': str,
                                'SOURCE_GRADEID': str})

            test_gold = pd.read_csv(f'../data/{args.mode}_query_pairs_{random_seed}.csv', sep='\t',
                                        dtype={'TARGET_ID': str,
                                               'TARGET_GRADEID': str,
                                               'SOURCE_ID': str,
                                               'SOURCE_GRADEID': str})
            if 'GOLD' not in test.columns:
                test = add_gold_column(test,test_gold,query_copies)
                test.to_csv(f'../results/{args.mode}_{args.model.replace(f"../models/", "").replace("sentence-transformers/", "")}_top{k}_{args.features}.csv',
                            sep='\t', index=False)
            ltr_infer(test,
                      data_dict,
                      k,
                      k2,
                      args.model,
                      age_to_grade,
                      args.features_rerank,
                      results_filepath,
                      random_seed)


    print(f'DONE!\n')