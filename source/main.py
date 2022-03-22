import os.path
import json
import pandas as pd
from utils import dump_to_json, generate_shared_docs_set, find_query_copies, grade_by_age, generate_train_test,generate_doc_sums,generate_test_lebanon
from data_exploration import generate_stats, target_set_stats
from generate_search_space import get_search_space
from match_learning_objectives import find_best_queries
from evaluation import eval_ranking
import time
import os
from tfidf_baseline import tfidf_match
from collections import defaultdict


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
    model_name = 'query2query-sbert'
    k = 5
    filterAge = False

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
    # generate_shared_docs_set(dump_filepath, source_curriculums, query_copies)
    target_data = read_in_target(target_filepath)
    train, dev, test = generate_train_test(target_data)


    for features in [['doc_sum']]:

        matches_per_cur = []
        results_filepath = f'../results/dev_{model_name}_top{k}_{features}_filterAge{filterAge}.csv'

        age_to_grade = None
        if filterAge or 'age' in features:
            age_to_grade = grade_by_age(source_curriculums)

        doc_sums = None
        if 'doc_sum' in features:
            doc_sums_df = pd.read_csv('../data/doc_sums.csv', sep='\t', dtype= {'queryId': str})
            doc_sums = defaultdict(list)
            for query_id, docs in doc_sums_df.groupby(['queryId']):
                doc_sums[query_id] = [doc_sum for doc_sum in list(docs['sumText'])]

        for index, cur_group in dev.groupby(['TARGET_CURRICULUM']):

            cur_group_dict = cur_group.to_dict(orient='list')
            target_cur = cur_group_dict['TARGET_CURRICULUM'][0]

            start_time = time.perf_counter()

            print(f'\nTARGET CURRICULUM: {target_cur}\n'
                  f'N of SEARCH QUERIES: {len(set(cur_group_dict["TARGET_ID"]))}\n'
                  f'FILTERS: {filters}\n'
                  f'MODEL: {model_name}\n'
                  f'FEATURES: {features}\n'
                  f'K: {k}\n')

            target_grade = None
            if filterAge:
                target_grade = cur_group_dict['TARGET_GRADEID'][0]
                print(f'TARGET GRADE: {target_grade}')

            target = cur_group.drop_duplicates(['TARGET_ID'])
            target = target.to_dict(orient='list')

            if model_name == 'tf-idf':
                print('Matching queries...')
                matches = tfidf_match(target_cur, source_curriculums, target)

            else:
                print('Generating search space...')
                source = get_search_space(data_dict,filters,target_cur, filterAge, target_grade, age_to_grade)

                print('Matching queries...')
                # target = cur_group.drop_duplicates(['TARGET_ID'])
                # target = target.to_dict(orient='list')
                matches = find_best_queries(source, target, model_name, features, k, age_to_grade, doc_sums)

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


        print('Evaluating results...')
        predictions = pd.read_csv(results_filepath,sep='\t')
        eval = dict()

        for cur_name, cur_predictions in predictions.groupby(['TARGET_CURRICULUM']):
            eval_dict = eval_ranking(cur_predictions,dev,query_copies,verbose=True)
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


    # print('Evaluating results...')
    # target_cur = 'Cambridge'
    # features = ['age', 'doc_title', 'subject']
    # results_filepath = f'../results/shared_docs_{target_cur}_{model_name}_top{k}_{features}.csv'
    # predictions = pd.read_csv(results_filepath,sep='\t')
    # map, mrr, r_p = eval_ranking(predictions,train,query_copies)
    # with open(f'../eval/shared_docs_{model_name}_top{k}_{features}.txt', 'w') as outfile:
    #     outfile.write(f'Target curricula: {set(predictions["TARGET_CURRICULUM"])}\n'
    #                   f'MAP: {map}\n'
    #                   f'R-precision: {r_p}\n'
    #                   f'MRR: {mrr}\n')
    #
    # print(f'DONE!\n')


    # for target_grade, filterAge in [('Elementary - Year 6', False),
    #                                 ('Elementary - Year 6',True),
    #                                 ('Intermediate - Year 7',False),
    #                                 ('Intermediate - Year 7', True),
    #                                 ('Intermediate - Year 8', False),
    #                                 ('Intermediate - Year 8', True),
    #                                 ('Secondary - Year 10', False),
    #                                 ('Secondary - Year 10', True)]:

    # for target_grade in ['Secondary - Year 10']:
    #
    #     filters = {'curriculums': source_curriculums, 'grades': (filterAge, source_grades), 'subjects': source_subjects}
    #     parameters_basename = f'{target_cur}_{target_grade}_{model_name}_top{k}_filterAge{filters["grades"][0]}_AgeFeature{age_as_feature}'
    #     results_filepath = f'../results/{parameters_basename}.csv'
    #
    #
    #     start_time = time.perf_counter()
    #
    #     print(f'\nTARGET CURRICULUM: {target_cur}\n'
    #           f'TARGET GRADE: {target_grade}\n'
    #           f'FILTER AGE: {filterAge}\n'
    #           f'MODEL: {model_name}\n'
    #           f'AGE AS FEATURE: {age_as_feature}\n'
    #           f'K: {k}\n')
    #
    #     # read in target data
    #     target = read_in_target (target_filepath, target_grade)
    #
    #     # get general descriptive stats
    #     # generate_stats(data)
    #     # target_set_stats(target)
    #
    #
    #     # generate search space (candidate queries/topics generation)
    #     print('Generating search space...')
    #     source = get_search_space(data, filters, target_cur, target_grade, query_copies, age_to_grade)
    #
    #     # match
    #     print('Matching learning objectives...')
    #     matches = match_lo(source, target, model_name, k, target_grade, target_cur, age_to_grade)
    #
    #     end_time = time.perf_counter()
    #     print(f"{len(list(dict.fromkeys(target['learning objective'].tolist())))} learning objectives matched in {end_time - start_time:0.4f} seconds")
    #
    #     # write out results
    #     with open(results_filepath, 'w') as outfile:
    #         outfile.write(f'learning objective\ttopic/query\tid\tscore\n')
    #         for trgt, pred in matches.items():
    #             for instance, id, score in pred:
    #                 outfile.write(f'{trgt}\t{instance}\t{id}\t{score}\n')
    #
    #     print(f'Evaluating...')
    #     # evaluate
    #     results = pd.read_csv(results_filepath, sep='\t')
    #     evaluate_dev_pre_study(results,target,parameters_basename,query_copies)
    #     eval_tfidf_baseline(target, target_cur, target_grade)
    #
    #
    #     print(f'DONE!')


if __name__ == '__main__':
    main()