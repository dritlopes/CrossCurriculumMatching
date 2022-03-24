import os.path
import json
import pandas as pd
from utils import dump_to_json, generate_shared_docs_set, find_query_copies, grade_by_age, generate_train_test,generate_doc_sums,generate_test_lebanon
from data_exploration import generate_stats, target_set_stats
from generate_search_space import get_search_space
from match_learning_objectives import find_best_queries
from evaluation import eval_ranking, topn_recall
import time
import os
from tfidf_baseline import tfidf_match
from collections import defaultdict
from finetune_sbert import fine_tune_bert


def read_in_data(source_filepath):

    if os.path.isfile(source_filepath):

        if source_filepath != '../data/data_dict.json':
            dump_to_json(source_filepath)

        with open('../data/data_dict.json') as json_file:
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


def main():


    dump_filepath = '../data/20220215-curriculum-data-export-report-production.csv'
    source_filepath = '../data/data_dict.json'
    # source_filepath = '../data/20220215-curriculum-data-export-report-production.csv'
    # target_filepath = '../data/Lebanon Query Matching.xlsx'
    # k-12 curricula in database: 'ICSE, CBSE, Cambridge, English, Lebanon, CCSS, NGSS, CSTA, Scotland
    source_curriculums = 'ICSE,CBSE,Cambridge,English,CCSS,NGSS,Scotland'
    target_filepath = f'../data/query_pairs_{source_curriculums}.csv'
    source_grades = ''
    source_subjects = ''
    filters = {'curriculums': source_curriculums, 'grades': source_grades, 'subjects': source_subjects}
    # features = ['age','doc_title','subject']
    model_filepath = '../models/paraphrase-sbert-label-title-sumsnsents-rankingloss'
    k = 5 # 20, 50, 100
    filterAge = False
    mode = 'dev'
    fine_tune = False
    fine_tune_args = {'pre_trained_model': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
                      'batch_size': int(12),
                      'epochs': int(3),
                      'train_loss': 'ranking',
                      'evaluator': 'information_retrieval',
                      'source': 'label,doc_titles'}

    print('Reading in data...')
    # read in database
    data_dict = read_in_data(source_filepath)
    # generate_stats(data)

    # find which queries are copies
    query_copies = find_query_copies(data_dict)

    # generate json with each summaryText of each pinned doc for each query
    if not os.path.isfile('../data/doc_sums.csv'):
        generate_doc_sums(data_dict, source_curriculums)

    # generate train and test sets with search queries
    if not os.path.isfile(f'../data/query_pairs_{source_curriculums}.csv'):
        generate_shared_docs_set (dump_filepath, source_curriculums, query_copies)

    if not os.path.isfile(f'../data/{mode}_query_pairs.csv'):
        target_data = read_in_target(target_filepath)
        generate_train_test(target_data)

    if fine_tune:
        fine_tune_bert(fine_tune_args)

    if mode == 'test':
        test = pd.read_csv('../data/test_query_pairs.csv', sep='\t', dtype= {'TARGET_ID': str,
                                                                       'TARGET_GRADEID': str,
                                                                       'SOURCE_ID': str,
                                                                       'SOURCE_GRADEID': str})
    else:
        test = pd.read_csv('../data/dev_query_pairs.csv', sep='\t', dtype= {'TARGET_ID': str,
                                                                       'TARGET_GRADEID': str,
                                                                       'SOURCE_ID': str,
                                                                       'SOURCE_GRADEID': str})

    for features in [['doc_title','doc_sum']]:

        matches_per_cur = []
        results_filepath = f'../results/{mode}_{model_filepath.replace("../models/","").replace("sentence-transformers/", "")}_top{k}_{features}_filterAge{filterAge}.csv'

        age_to_grade = None
        if filterAge or 'age' in features:
            age_to_grade = grade_by_age(source_curriculums)

        doc_sums = None
        if 'doc_sum' in features:
            doc_sums_df = pd.read_csv('../data/doc_sums.csv', sep='\t', dtype= {'queryId': str})
            doc_sums = defaultdict(list)
            for query_id, docs in doc_sums_df.groupby(['queryId']):
                doc_sums[query_id] = [doc_sum for doc_sum in list(docs['sumText'])]

        for index, cur_group in test.groupby(['TARGET_CURRICULUM']):

            cur_group_dict = cur_group.to_dict(orient='list')
            target_cur = cur_group_dict['TARGET_CURRICULUM'][0]

            start_time = time.perf_counter()

            print(f'\nTARGET CURRICULUM: {target_cur}\n'
                  f'N of SEARCH QUERIES: {len(set(cur_group_dict["TARGET_ID"]))}\n'
                  f'FILTERS: {filters}\n'
                  f'MODEL: {model_filepath.replace("../models/","")}\n'
                  f'FEATURES: {features}\n'
                  f'K: {k}\n')

            target_grade = None
            if filterAge:
                target_grade = cur_group_dict['TARGET_GRADEID'][0]
                print(f'TARGET GRADE: {target_grade}')

            target = cur_group.drop_duplicates(['TARGET_ID'])
            target = target.to_dict(orient='list')

            if 'tf-idf' in model_filepath:
                print('Matching queries...')
                matches = tfidf_match(target_cur, source_curriculums, target)

            else:
                print('Generating search space...')
                source = get_search_space(data_dict,filters,target_cur, filterAge, target_grade, age_to_grade)

                print('Matching queries...')
                # target = cur_group.drop_duplicates(['TARGET_ID'])
                # target = target.to_dict(orient='list')
                matches = find_best_queries(source, target, model_filepath, features, k, age_to_grade, doc_sums)

            matches_per_cur.append((target_cur,matches))

            end_time = time.perf_counter()
            print(f"{len(set(cur_group_dict['TARGET_ID']))} search queries matched in {end_time - start_time:0.4f} seconds")

        print('Saving results...')
        with open(results_filepath, 'w') as outfile:
            outfile.write(f'TARGET_CURRICULUM\tTARGET\tTARGET_ID\tTARGET_PATH\tSOURCE\tSOURCE_ID\tSOURCE_PATH\tSCORE\n')
            for target_cur,matches in matches_per_cur:
                for target_id, target_dict in matches.items():
                    for source_label, id, path, score in target_dict["scores"]:
                        outfile.write(f'{target_cur}\t{target_dict["label"]}\t{target_id}\t{target_dict["path"]}'
                                      f'\t{source_label}\t{id}\t{path}\t{score}\n')

        # check recall of topn
        predictions = pd.read_csv(results_filepath, sep='\t',dtype={'TARGET_ID': str, 'SOURCE_ID': str})
        recall = topn_recall(predictions,test,query_copies)
        print(f'Recall of top {k} predictions: {recall}')

        print('Evaluating results...')
        predictions = pd.read_csv(results_filepath,sep='\t',dtype={'TARGET_ID': str, 'SOURCE_ID': str})
        eval = dict()

        for cur_name, cur_predictions in predictions.groupby(['TARGET_CURRICULUM']):
            eval_dict = eval_ranking(cur_predictions,test,query_copies,verbose=True)
            eval[cur_name] = eval_dict

        map_values = [eval_dict['map'] for eval_dict in list(eval.values())]
        rp_values = [eval_dict['rp'] for eval_dict in list(eval.values())]
        mrr_values = [eval_dict['mrr'] for eval_dict in list(eval.values())]
        w_map_values = [eval_dict['map']*eval_dict['support'] for eval_dict in list(eval.values())]
        w_rp_values = [eval_dict['rp']*eval_dict['support'] for eval_dict in list(eval.values())]
        w_mrr_values = [eval_dict['mrr']*eval_dict['support'] for eval_dict in list(eval.values())]

        eval['macro-avg'] = {'map': sum(map_values)/len(map_values),
                             'rp': sum(rp_values)/len(rp_values),
                             'mrr': sum(mrr_values)/len(mrr_values)}
        eval['micro-avg'] = {'map': sum(w_map_values)/len(set(list(predictions['TARGET_ID']))),
                             'rp': sum(w_rp_values)/len(set(list(predictions['TARGET_ID']))),
                             'mrr': sum(w_mrr_values)/len(set(list(predictions['TARGET_ID'])))}

        with open(f'../eval/{results_filepath.replace("../results/", "").strip(".csv")}.json', 'w') as outfile:
            json.dump(eval, outfile)

        print(f'DONE!\n')


if __name__ == '__main__':
    main()