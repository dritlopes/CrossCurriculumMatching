import os.path
import pandas as pd
from utils import read_in_data, generate_shared_docs_set, find_query_copies, grade_by_age, generate_train_test
from utils import target_subset, generate_doc_sums, generate_info_file
from generate_search_space import get_search_space
from match_learning_objectives import find_best_queries
from evaluation import add_gold_column
import time
import os
from tfidf_baseline import tfidf_match
from collections import defaultdict
from ltr import ltr_infer
import argparse


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
    k = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=False, default='cosine', choices=['cosine', 're-rank', 'classification'])
    parser.add_argument("--features", required=False, default="")
    parser.add_argument("--random_seed", required=False, default="42", choices=["42","13","7"])
    parser.add_argument("--mode", required=False, default='test', choices=['dev','test'])
    parser.add_argument("--rerank_model", required=False)
    parser.add_argument("--uncase", required=False, default="False", choices=['False', 'True'])
    parser.add_argument("--features_rerank", required=False, default="topic")
    args = parser.parse_args()

    if args.model not in ['tf-idf','sentence-transformers/paraphrase-MiniLM-L6-v2'] and not os.path.isdir(args.model) and not os.path.isfile(args.model):
        raise Exception(f'{args.model} not found. Please add the specified model to the /models folder and try again.')

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

    if args.method == 're-rank': k = 30

    # run experiment

    if not os.path.isfile(f'../data/{args.mode}_query_pairs_{args.random_seed}.csv'):
        target_data = read_in_target(target_filepath)
        generate_train_test(target_data, int(args.random_seed))

    test = pd.read_csv(f'../data/{args.mode}_query_pairs_{args.random_seed}.csv', sep='\t', dtype= {'TARGET_ID': str,
                                                                                            'TARGET_GRADEID': str,
                                                                                            'SOURCE_ID': str,
                                                                                            'SOURCE_GRADEID': str})

    results_filepath = f'../results/{args.mode}_{args.model.replace(f"../models/","").replace("sentence-transformers/", "").replace("_"+ str(args.random_seed),"")}_{args.random_seed}_top{k}_{args.features}.csv'

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
        # target = target.iloc[:1]
        target = target.to_dict(orient='list')

        if args.method == 'classification':
            target = target_subset(target,50)

        print(f'\nTARGET CURRICULUM: {target_cur}\n'
              f'N of TARGET LO: {len(target["TARGET"])}\n'
              f'FILTERS: {filters}\n'
              f'MODEL: {args.model.replace(f"../models/", "")}\n'
              f'FEATURES: {args.features}\n'
              f'K: {k}\n'
              f'RANDOM_SEED: {args.random_seed}\n')

        if args.model == 'tf-idf':
            print('Matching queries...')
            matches = tfidf_match(target_cur, curriculums, target, k)

        else:
            print('Generating search space...')
            source = get_search_space(data_dict, filters, target_cur)

            print('Matching queries...')
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
        if not os.path.isfile(args.rerank_model):
            raise Exception(
                f'{args.model} not found. Please add the specified model to the /models folder and try again.')
        results_filepath = f'../results/rerank_ltr_{args.mode}_{args.random_seed}_{args.features_rerank}.csv'
        test = pd.read_csv(f'../results/{args.mode}_{args.model.replace("../models/", "").replace("sentence-transformers/", "")}_top{k}_{args.features}.csv',
                            sep='\t', dtype={'TARGET_ID': str,
                            'TARGET_GRADEID': str,
                            'SOURCE_ID': str,
                            'SOURCE_GRADEID': str})

        test_gold = pd.read_csv(f'../data/{args.mode}_query_pairs_{args.random_seed}.csv', sep='\t',
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
                  args.rerank_model,
                  age_to_grade,
                  args.features_rerank,
                  results_filepath,
                  int(args.random_seed))


    print(f'DONE!\n')

if __name__ == '__main__':
    main()
