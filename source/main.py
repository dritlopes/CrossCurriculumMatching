import os.path
import json
import pandas as pd
from utils import dump_to_json, generate_shared_docs_set, find_query_copies, grade_by_age, generate_train_test,generate_doc_sums,generate_test_lebanon
from data_exploration import generate_stats, target_set_stats, data_distribution
from generate_search_space import get_search_space
from match_learning_objectives import find_best_queries
from match_higher_layers import rerank
from evaluation import eval_ranking, topn_recall, add_gold_column
import time
import os
from tfidf_baseline import tfidf_match
from collections import defaultdict
from finetune_sbert import fine_tune_bert
from ltr import train_ltr, ltr_infer
from sklearn.model_selection import train_test_split
import sys


def read_in_data(source_filepath, DATA_DIR):

    if os.path.isfile(source_filepath):

        if source_filepath != f'{DATA_DIR}/data_dict.json':
            dump_to_json(source_filepath)

        with open(f'{DATA_DIR}/data_dict.json') as json_file:
            data = json.load(json_file)

    else:
        print(f'{source_filepath} not found. Please make sure the source data is either the dump csv or a json file in the data folder of this repo')
        exit()

    return data

def read_in_target(target_filepath):

    if os.path.isfile(target_filepath):

        target_data = None

        if target_filepath == '../data/Lebanon Query Matching.xlsx':
            generate_test_lebanon(target_filepath)
        elif target_filepath == 'test_Lebanon.csv':
            pass

        else:
            target_data = pd.read_csv(target_filepath, sep='\t')
            for column_id in ['TARGET_ID', 'SOURCE_ID', 'TARGET_GRADEID', 'SOURCE_GRADEID']:
                target_data[column_id] = target_data[column_id].astype(str)
                if column_id in {'TARGET_GRADEID', 'SOURCE_GRADEID'}:
                    target_data[column_id] = target_data[column_id].map(lambda x: x.strip('.0'))

    else:
        raise Exception(f'{target_filepath} not found')

    return target_data


def save_results(predictions, results_filepath, results_filepath_4_eval,k):

    # top 100 for eval
    with open(results_filepath_4_eval, 'w') as outfile:
        outfile.write(f'TARGET_CURRICULUM\tTARGET\tTARGET_ID\tTARGET_PATH\tSOURCE\tSOURCE_ID\tSOURCE_PATH\tSCORE\n')
        for target_cur, matches in predictions:
            for target_id, target_dict in matches.items():
                for source_label, id, path, score in target_dict["scores"]:
                    target_dict["path"] = target_dict["path"].replace("\n", "")
                    path = path.replace("\n", "")
                    outfile.write(f'{target_cur}\t{target_dict["label"]}\t{target_id}\t{target_dict["path"]}'
                                  f'\t{source_label}\t{id}\t{path}\t{score}\n')

    # top 5 for inference results
    with open(results_filepath, 'w') as outfile:
        outfile.write(f'TARGET_CURRICULUM\tTARGET\tTARGET_ID\tTARGET_PATH\tSOURCE\tSOURCE_ID\tSOURCE_PATH\tSCORE\n')
        for target_cur, matches in predictions:
            for target_id, target_dict in matches.items():
                for source_label, id, path, score in target_dict["scores"][:k]:
                    target_dict["path"] = target_dict["path"].replace("\n", "")
                    path = path.replace("\n", "")
                    outfile.write(f'{target_cur}\t{target_dict["label"]}\t{target_id}\t{target_dict["path"]}'
                                  f'\t{source_label}\t{id}\t{path}\t{score}\n')

def main():

    DATA_DIR = sys.argv[1]
    MODEL_SAVE = sys.argv[2]
    EVAL_DIR = sys.argv[3]
    RESULTS_DIR = sys.argv[4]

    # TODO prepare test data Lebanon

    # DATA_DIR = '../data'
    # EVAL_DIR = '../eval'
    # MODEL_SAVE = '../models'
    # RESULTS_DIR = '../results'

    dump_filepath = f'{DATA_DIR}/20220215-curriculum-data-export-report-production.csv'
    source_filepath = f'{DATA_DIR}/data_dict.json'
    # k-12 curricula in database: 'ICSE, CBSE, Cambridge, English, Lebanon, CCSS, NGSS, CSTA, Scotland
    curriculums = 'ICSE,CBSE,Cambridge,English,CCSS,NGSS,Scotland'
    target_filepath = f'{DATA_DIR}/query_pairs_{curriculums}.csv'
    source_grades = ''
    source_subjects = ''
    filters = {'curriculums': curriculums, 'grades': source_grades, 'subjects': source_subjects}

    exp_params = {'random_seed': [13],
                  'model': f'{MODEL_SAVE}/model_weights.pth', # ../models/distilbert_doctitle,topic,subject,age_13.pth
                  'features': 'grade,subject,topic,doc_titles',
                  'k': 5,
                  'r': 100,
                  'filter_age': False,
                  'mode': 'dev',
                  'uncase': True,
                  'method': 'classification',
                  'base_model': "distilbert-base-uncased"}

    fine_tune_args = {'pre_trained_model': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
                      'batch_size': int(12),
                      'epochs': int(3),
                      'train_loss': 'ranking',
                      'evaluator': 'information_retrieval',
                      'source': 'label,doc_titles',
                      'parent_nodes': 'grade,subject,topic'}

    # setups = [
              # {'model': '../models/paraphrase-sbert-label-title-rankingloss-nodup_',
              #  'features': [[],['doc_title']]},
              # {'model': '../models/paraphrase-sbert-label-title-sums-rankingloss-nodup_',
              #  'features': [[],['doc_title','doc_sum']]},
              # {'model': '../models/paraphrase-sbert-label-title-sumsnsents-rankingloss-nodup_',
              #  'features': [[],['doc_title','doc_sum_nsents']]}
              # {'model': '../models/paraphrase-sbert-label-sums-rankingloss-nodup_',
              #  'features': [[],['doc_sum_1sent']]},
              # {'model': '../models/paraphrase-sbert-label-sumsnsents-rankingloss-nodup_',
              #  'features': [[],['doc_sum_nsents']]}]

    print('Reading in data...')
    # read in database
    data_dict = read_in_data(source_filepath, DATA_DIR)
    # generate_stats(data_dict,curriculums=curriculums+',Lebanon,CSTA')

    # find which queries are copies
    query_copies = find_query_copies(data_dict)

    # generate json with each summaryText of each pinned doc for each query
    if not os.path.isfile(f'{DATA_DIR}/doc_sums.csv'):
        generate_doc_sums(data_dict, curriculums)

    # generate train and test sets with search queries
    if not os.path.isfile(f'{DATA_DIR}/query_pairs_{curriculums}.csv'):
        generate_shared_docs_set(dump_filepath, curriculums, query_copies)

    for random_seed in exp_params['random_seed']:

        mode = exp_params['mode']

        if not os.path.isfile(f'{DATA_DIR}/{mode}_query_pairs_{random_seed}.csv'):
            target_data = read_in_target(target_filepath)
            generate_train_test(target_data, random_seed)

        if mode == 'train':
            fine_tune_bert(fine_tune_args)


        test = pd.read_csv(f'{DATA_DIR}/{mode}_query_pairs_{random_seed}.csv', sep='\t', dtype= {'TARGET_ID': str,
                                                                           'TARGET_GRADEID': str,
                                                                           'SOURCE_ID': str,
                                                                           'SOURCE_GRADEID': str})

        model_filepath = exp_params["model"]
        # if exp_params["model"] not in ['sentence-transformers/paraphrase-MiniLM-L6-v2','cc.en.300.bin', 'tf-idf']:
        #     model_filepath = f'{exp_params["model"]}{random_seed}'
        features = exp_params['features']
        k = exp_params['k']
        r = exp_params['r']
        uncase = exp_params['uncase']
        method = exp_params['method']
        base_model = exp_params['base_model']

        # if model_filepath == '../models/paraphrase-sbert-label-rankingloss-nodup': model_filepath = f'{model_filepath}_{random_seed}'

        results_filepath = f'{RESULTS_DIR}/{mode}_{model_filepath.replace("../models/","").replace("sentence-transformers/", "").replace("_"+ str(random_seed),"")}_{random_seed}_top{k}_{features}_filterAge{exp_params["filter_age"]}.csv'
        results_filepath_4_eval = f'{RESULTS_DIR}/{mode}_{model_filepath.replace("../models/","").replace("sentence-transformers/", "").replace("_"+ str(random_seed),"")}_{random_seed}_top{r}_{features}_filterAge{exp_params["filter_age"]}.csv'

        if not os.path.isfile(results_filepath):

            matches_per_cur = []

            age_to_grade = None
            if exp_params['filter_age'] or 'grade' in features:
                age_to_grade = grade_by_age(curriculums, age_filepath = f'{DATA_DIR}/MASTER Reading levels and age filter settings (pwd 123456).xlsx')

            doc_sums = None
            if 'doc_sum_1sent' in features or 'doc_sum_nsents' in features:
                doc_sums_df = pd.read_csv(f'{DATA_DIR}/doc_sums.csv', sep='\t', dtype= {'queryId': str})
                doc_sums = defaultdict(list)
                for query_id, docs in doc_sums_df.groupby(['queryId']):
                    doc_sums[query_id] = [doc_sum for doc_sum in list(docs['sumText'])]

            for index, cur_group in test.groupby(['TARGET_CURRICULUM']):

                cur_group = cur_group[:1]
                cur_group_dict = cur_group.to_dict(orient='list')
                target_cur = cur_group_dict['TARGET_CURRICULUM'][0]

                start_time = time.perf_counter()

                print(f'\nTARGET CURRICULUM: {target_cur}\n'
                      f'N of SEARCH QUERIES: {len(set(cur_group_dict["TARGET_ID"]))}\n'
                      f'FILTERS: {filters}\n'
                      f'MODEL: {model_filepath.replace("../models/","")}\n'
                      f'FEATURES: {features}\n'
                      f'K: {k}\n'
                      f'RANDOM_SEED: {random_seed}\n')

                target_grade = None
                if exp_params['filter_age']:
                    target_grade = cur_group_dict['TARGET_GRADEID'][0]
                    print(f'TARGET GRADE: {target_grade}')

                target = cur_group.drop_duplicates(['TARGET_ID'])
                target = target.to_dict(orient='list')

                if model_filepath == 'tf-idf':
                    print('Matching queries...')
                    matches = tfidf_match(target_cur, curriculums, target, k)

                else:
                    print('Generating search space...')
                    source = get_search_space(data_dict,filters,target_cur, exp_params['filter_age'], target_grade, age_to_grade)

                    print('Matching queries...')
                    matches = find_best_queries(source,
                                                target,
                                                model_filepath,
                                                features,
                                                r,
                                                mode=method,
                                                age_to_grade=age_to_grade,
                                                doc_sums_dict=doc_sums,
                                                uncased=uncase,
                                                base_model=base_model)

                matches_per_cur.append((target_cur,matches))

                end_time = time.perf_counter()
                print(f"{len(set(cur_group_dict['TARGET_ID']))} search queries matched in {end_time - start_time:0.4f} seconds")

            print('Saving results of ranking...')
            save_results(matches_per_cur,results_filepath,results_filepath_4_eval,k)

        print('Evaluating results...')
        print(f'MODEL: {model_filepath.replace(f"{DATA_DIR}", "")}\n'
              f'RANDOM_SEED: {random_seed}\n')
        predictions = pd.read_csv(results_filepath_4_eval, sep='\t', dtype={'TARGET_ID': str, 'SOURCE_ID': str})
        predictions = add_gold_column(predictions,test,query_copies)
        predictions.to_csv(results_filepath_4_eval,sep='\t',index=False) # save predictions with gold column
        eval_filepath = f'{EVAL_DIR}/{results_filepath.replace("../results/", "").strip(".csv")}.json'
        eval_ranking(predictions, k, eval_filepath)

                # check recall of topn
                # predictions = pd.read_csv(results_filepath, sep='\t',dtype={'TARGET_ID': str, 'SOURCE_ID': str})
                # recall = topn_recall(predictions,test,query_copies,k)
                # print(f'Recall of top {k} predictions: {recall}')
                # n = 20
                # recall = topn_recall(predictions,test,query_copies,n)
                # print(f'Recall of top {n} predictions: {recall}')
                # while recall < 0.9 and n < k:
                #   n += 10
                #   recall = topn_recall(predictions,test,query_copies,n)
                #   print(f'Recall of top {n} predictions: {recall}')

                # method = 'ltr'
                # # layers = "grade,subject,unit,topic"
                # model_save_path = f'../models/ltr_{random_seed}_{layers}.txt'
                # age_to_grade = grade_by_age(source_curriculums)

                # if method == 'ltr' and not os.path.isfile(model_save_path):
                # if method == 'ltr':
                #     train = pd.read_csv(f'../data/train_query_pairs_{random_seed}.csv', sep='\t', dtype= {'TARGET_ID': str,
                #                                                                'TARGET_GRADEID': str,
                #                                                                'SOURCE_ID': str,
                #                                                                'SOURCE_GRADEID': str})
                #     train_pred = pd.read_csv(f'../results/train_{model_filepath.replace("../models/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAge{filterAge}.csv',
                #                              sep='\t', dtype= {'TARGET_ID': str,
                #                                                 'TARGET_GRADEID': str,
                #                                                 'SOURCE_ID': str,
                #                                                 'SOURCE_GRADEID': str})
                #     dev_pred = pd.read_csv(
                #         f'../results/dev_{model_filepath.replace("../models/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAge{filterAge}.csv',
                #         sep='\t', dtype={'TARGET_ID': str,
                #                          'TARGET_GRADEID': str,
                #                          'SOURCE_ID': str,
                #                          'SOURCE_GRADEID': str})
                #     dev = pd.read_csv(f'../data/dev_query_pairs_{random_seed}.csv', sep='\t',
                #                            dtype={'TARGET_ID': str,
                #                                   'TARGET_GRADEID': str,
                #                                   'SOURCE_ID': str,
                #                                   'SOURCE_GRADEID': str})
                #     train_ltr(train,
                #               train_pred,
                #               dev,
                #               dev_pred,
                #               data_dict,
                #               model_filepath,
                #               model_save_path,
                #               random_seed,
                #               age_to_grade,
                #               layers,
                #               query_copies)
                #
                # results_filepath = f'../results/rerank_ltr_{mode}_{random_seed}_{layers}.csv'
                # dev = pd.read_csv(f'../results/dev_{model_filepath.replace("../models/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAge{filterAge}.csv',
                # sep='\t', dtype={'TARGET_ID': str,
                #                  'TARGET_GRADEID': str,
                #                  'SOURCE_ID': str,
                #                  'SOURCE_GRADEID': str})
                # dev_gold = pd.read_csv(f'../data/dev_query_pairs_{random_seed}.csv', sep='\t', dtype={'TARGET_ID': str,
                #                                                                                  'TARGET_GRADEID': str,
                #                                                                                  'SOURCE_ID': str,
                #                                                                                  'SOURCE_GRADEID': str})
                # dev = add_gold_column(dev,dev_gold,query_copies)
                # ltr_infer(dev,
                #           data_dict,
                #           k,
                #           k2,
                #           model_filepath,
                #           model_save_path,
                #           age_to_grade,
                #           layers,
                #           results_filepath)

                # print('Evaluating results...')
                # predictions = pd.read_csv(results_filepath, sep='\t', dtype={'TARGET_ID': str, 'SOURCE_ID': str})
                # eval_ranking(predictions, dev_gold, query_copies, results_filepath)


        print(f'DONE!\n')


if __name__ == '__main__':
    main()