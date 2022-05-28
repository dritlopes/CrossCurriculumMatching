import os.path
import json
import pandas as pd
from utils import dump_to_json, generate_shared_docs_set, find_query_copies, grade_by_age, generate_train_test, generate_doc_sums
from utils import generate_test_lebanon, find_age, target_subset
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
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


def read_in_data(source_filepath, DATA_DIR):

    if os.path.isfile(source_filepath):

        if source_filepath != f'{DATA_DIR}/data_dict.json':
            dump_to_json(source_filepath,DATA_DIR)

        with open(f'{DATA_DIR}/data_dict.json') as json_file:
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


def generate_info_file(data_dict_filepath, info_filepath, doc_sums, age_to_grade, curriculums):

    with open(data_dict_filepath) as json_file:
        data = json.load(json_file)

    rows = []
    for cur_id, cur in data.items():
        if cur['label'] in curriculums:
            for grade_id, grade in cur['grade'].items():
                for subject_id, subject in grade['subject'].items():
                    for unit_id, unit in subject['unit'].items():
                        for topic_id, topic in unit['topic'].items():
                            for query_id, query in topic['query'].items():
                                label = query['label']
                                if label == '': label = topic['label']
                                query_dict = {'id': str(query_id),
                                              'query_term': label.strip(),
                                              'topic': topic['label'].strip(),
                                              'subject': subject['label'].strip(),
                                              'grade': grade['label'].strip(),
                                              'curriculum': cur['label'].strip(),
                                              'doc_titles': ' '.join(
                                                  [doc_info['title'].strip() for doc_info in query['docs'].values() if
                                                   doc_info['pin']]),
                                              'doc_sums_1sent': '',
                                              'doc_sums_nsent': ''}

                                age = find_age(age_to_grade, cur['label'], grade['label'])
                                query_dict['age'] = str(age)

                                # Including organic documents
                                # query_dict = {'id': query_id,
                                #              'label': label,
                                #              'doc_titles': ' '.join(
                                #                  [doc_info['title'] for doc_info in query['docs'].values()]),
                                #               'doc_sums_1sent': '',
                                #               'doc_sums_nsent': ''}

                                if doc_sums[query_id]:
                                    doc_sums_1sent = []
                                    doc_sums_nsent = []
                                    for sum in doc_sums[query_id]:
                                        sents = sent_tokenize(sum)
                                        doc_sums_1sent.append(sents[0])
                                        tags_label = pos_tag(word_tokenize(label))
                                        nouns = [word for word, pos in tags_label if pos.startswith('NN')]
                                        for sent in sents:
                                            tokens = word_tokenize(sent)
                                            for token in tokens:
                                                if token in nouns and sent not in doc_sums_nsent:
                                                    doc_sums_nsent.append(sent)
                                    query_dict['doc_sums_1sent'] = ' '.join(doc_sums_1sent).strip()
                                    query_dict['doc_sums_nsent'] = ' '.join(doc_sums_nsent).strip()

                                for key,v in query_dict.items():
                                    new = v.replace('\n','')
                                    new = new.replace('\t','')
                                    query_dict[key] = new

                                rows.append(query_dict)

    with open(info_filepath, 'w', encoding="utf-8") as out:
        headers = ['query_id', 'query_term', 'doc_titles', 'doc_sums_1sent', 'doc_sums_nsents', 'topic', 'subject', 'grade',
                   'age', 'curriculum']
        headers = '\t'.join(headers) + '\n'
        out.write(headers)
        for query_dict in rows:
            out.write(
                f'{query_dict["id"]}\t{query_dict["query_term"]}\t{query_dict["doc_titles"]}\t{query_dict["doc_sums_1sent"]}'
                f'\t{query_dict["doc_sums_nsent"]}\t{query_dict["topic"]}\t{query_dict["subject"]}\t{query_dict["grade"]}'
                f'\t{query_dict["age"]}\t{query_dict["curriculum"]}\n')


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

    # DATA_DIR = sys.argv[1]
    # MODEL_SAVE = sys.argv[2]
    # EVAL_DIR = sys.argv[3]
    # RESULTS_DIR = sys.argv[4]

    DATA_DIR = '../data'
    EVAL_DIR = '../eval'
    MODEL_SAVE = '../models'
    RESULTS_DIR = '../results'

    dump_filepath = f'{DATA_DIR}/20220215-curriculum-data-export-report-production.csv'
    source_filepath = f'{DATA_DIR}/data_dict.json'
    # k-12 curricula in database: 'ICSE, CBSE, Cambridge, English, Lebanon, CCSS, NGSS, CSTA, Scotland
    curriculums = 'ICSE,CBSE,Cambridge,English,CCSS,NGSS,Scotland'
    target_filepath = f'{DATA_DIR}/query_pairs_{curriculums}.csv'
    source_grades = ''
    source_subjects = ''
    filters = {'curriculums': curriculums, 'grades': source_grades, 'subjects': source_subjects}

    exp_params = {'random_seed': [42,13,7],
                  'model': [f'tf-idf'],
                  'features': [''],
                  'k': 5,
                  # 'r': 100,
                  'filter_age': False,
                  'uncase': False,
                  'method': 'cosine',
                  'mode': ['test'],
                  'base_model': "",
                  're-rank': False,
                  'higher_layers': None,
                  'k2': 5}

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
    data_dict = read_in_data(source_filepath, DATA_DIR)
    # generate_stats(data_dict,curriculums=curriculums+',Lebanon,CSTA')

    # find which queries are copies
    query_copies = find_query_copies(data_dict, DATA_DIR)

    # generate json with each summaryText of each pinned doc for each query
    if not os.path.isfile(f'{DATA_DIR}/doc_sums.csv'):
        generate_doc_sums(data_dict, curriculums, DATA_DIR)

    # generate csv with queries ids and features
    if not os.path.isfile(f'{DATA_DIR}/data.csv'):
        age_to_grade = grade_by_age(curriculums,
                                age_filepath=f'{DATA_DIR}/MASTER Reading levels and age filter settings (pwd 123456).xlsx')
        doc_sums_df = pd.read_csv(f'{DATA_DIR}/doc_sums.csv', sep='\t', dtype={'queryId': str})
        doc_sums = defaultdict(list)
        for query_id, docs in doc_sums_df.groupby(['queryId']):
            doc_sums[query_id] = [doc_sum for doc_sum in list(docs['sumText'])]
        generate_info_file(f'{DATA_DIR}/data_dict.json',f'{DATA_DIR}/data.csv',doc_sums,age_to_grade,curriculums)

    # generate train and test sets with search queries
    if not os.path.isfile(f'{DATA_DIR}/query_pairs_{curriculums}.csv'):
        generate_shared_docs_set(dump_filepath, curriculums, query_copies, DATA_DIR)

    for mode in exp_params['mode']:

        for random_seed in exp_params['random_seed']:

            if not os.path.isfile(f'{DATA_DIR}/{mode}_query_pairs_{random_seed}.csv'):
                target_data = read_in_target(target_filepath)
                generate_train_test(target_data, DATA_DIR, random_seed)

            # if mode == 'train': # train classifier or fine_tune sbert
            #     fine_tune_bert(fine_tune_args)

            if mode == 'test_lebanon':
                if not os.path.isfile(f'{DATA_DIR}/test_Lebanon.csv'):
                    generate_test_lebanon(f'{DATA_DIR}/Lebanon Query Matching.xlsx', f'{DATA_DIR}/test_Lebanon.csv')
                test = pd.read_csv(f'{DATA_DIR}/test_Lebanon.csv', sep='\t', dtype= {'TARGET_ID': str,
                                                                                   'TARGET_GRADEID': str,
                                                                                   'SOURCE_ID': str,
                                                                                   'SOURCE_GRADEID': str})
            else:
                test = pd.read_csv(f'{DATA_DIR}/{mode}_query_pairs_{random_seed}.csv', sep='\t', dtype= {'TARGET_ID': str,
                                                                                   'TARGET_GRADEID': str,
                                                                                   'SOURCE_ID': str,
                                                                       'SOURCE_GRADEID': str})

            for model_filepath in exp_params["model"]:

                feature_combi = exp_params['features']
                # if model_filepath == f'{MODEL_SAVE}/paraphrase-sbert-label-title-rankingloss-nodup_':
                #     feature_combi = ['doc_title']
                # elif model_filepath == f'{MODEL_SAVE}/paraphrase-sbert-label-title-sums-rankingloss-nodup_':
                #     feature_combi = ['doc_title,doc_sum_1sent']
                # elif model_filepath == f'{MODEL_SAVE}/paraphrase-sbert-label-title-sumsnsents-rankingloss-nodup_':
                #     feature_combi = ['doc_title,doc_sum_nsents']
                # elif model_filepath == 'sentence-transformers/paraphrase-MiniLM-L6-v2':
                #     feature_combi = ['doc_title']

                if model_filepath not in ['sentence-transformers/paraphrase-MiniLM-L6-v2',
                                           f'{MODEL_SAVE}/cc.en.300.bin',
                                          'tf-idf',
                                           f'{MODEL_SAVE}/model_weights.pth']:
                    model_filepath = f'{model_filepath}{random_seed}'

                k = exp_params['k']
                # r = exp_params['r']
                uncase = exp_params['uncase']
                method = exp_params['method']
                base_model = exp_params['base_model']

                for features in feature_combi:

                    results_filepath = f'{RESULTS_DIR}/{mode}_{model_filepath.replace(f"{MODEL_SAVE}/","").replace("sentence-transformers/", "").replace("_"+ str(random_seed),"")}_{random_seed}_top{k}_{features}_filterAge{exp_params["filter_age"]}.csv'
                    # results_filepath_4_eval = f'{RESULTS_DIR}/{mode}_{model_filepath.replace("../models/","").replace("sentence-transformers/", "").replace("_"+ str(random_seed),"")}_{random_seed}_top{r}_{features}_filterAge{exp_params["filter_age"]}.csv'

                    if not os.path.isfile(results_filepath):
                        print(results_filepath)

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

                            cur_group_dict = cur_group.to_dict(orient='list')
                            target_cur = cur_group_dict['TARGET_CURRICULUM'][0]

                            start_time = time.perf_counter()

                            target_grade = None
                            if exp_params['filter_age']:
                                target_grade = cur_group_dict['TARGET_GRADEID'][0]
                                print(f'TARGET GRADE: {target_grade}')

                            target = cur_group.drop_duplicates(['TARGET_ID'])
                            target = target.to_dict(orient='list')

                            if exp_params['method'] == 'classification':
                                target = target_subset(target,50)

                            print(f'\nTARGET CURRICULUM: {target_cur}\n'
                                  f'N of TARGET LO: {len(target["TARGET"])}\n'
                                  f'FILTERS: {filters}\n'
                                  f'MODEL: {model_filepath.replace(f"{MODEL_SAVE}/", "")}\n'
                                  f'FEATURES: {features}\n'
                                  f'K: {k}\n'
                                  f'RANDOM_SEED: {random_seed}\n')

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
                                                            k,
                                                            mode=method,
                                                            age_to_grade=age_to_grade,
                                                            doc_sums_dict=doc_sums,
                                                            uncased=uncase,
                                                            base_model=base_model)

                            matches_per_cur.append((target_cur,matches))

                            end_time = time.perf_counter()
                            print(f"{len(set(cur_group_dict['TARGET_ID']))} search queries matched in {end_time - start_time:0.4f} seconds")

                        print('Saving results of ranking...')
                        save_results(matches_per_cur,results_filepath,k)

                    print('Evaluating results...')
                    print(f'MODEL: {model_filepath.replace(f"{DATA_DIR}", "")}\n'
                          f'RANDOM_SEED: {random_seed}\n'
                          f'FEATURES: {features}\n')
                    predictions = pd.read_csv(results_filepath, sep='\t', dtype={'TARGET_ID': str, 'SOURCE_ID': str})
                    if 'GOLD' not in predictions.columns:
                        predictions = add_gold_column(predictions,test,query_copies)
                        predictions.to_csv(results_filepath,sep='\t',index=False) # save predictions with gold column
                    eval_filepath = f'{EVAL_DIR}/{results_filepath.replace(f"{RESULTS_DIR}/", "").strip(".csv")}.json'
                    eval_ranking(predictions, test, query_copies, k, eval_filepath)

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

                if exp_params['re-rank']:

                    for layers in exp_params['higher_layers']:

                        model_save_path = f'{MODEL_SAVE}/ltr_{random_seed}_{layers}.txt'
                        age_to_grade = grade_by_age(curriculums)
                        k2 = exp_params['k2']

                        if not os.path.isfile(model_save_path) and os.path.isfile(f'{RESULTS_DIR}/train_{model_filepath.replace(f"{MODEL_SAVE}/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAgeFalse.csv') and os.path.isfile(f'{RESULTS_DIR}/dev_{model_filepath.replace(f"{MODEL_SAVE}/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAgeFalse.csv'):

                            train_pred = pd.read_csv(f'{RESULTS_DIR}/train_{model_filepath.replace(f"{MODEL_SAVE}/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAgeFalse.csv',
                                                     sep='\t', dtype= {'TARGET_ID': str,
                                                                        'TARGET_GRADEID': str,
                                                                        'SOURCE_ID': str,
                                                                        'SOURCE_GRADEID': str})
                            if 'GOLD' not in train_pred.columns:
                                train_gold = pd.read_csv(f'{DATA_DIR}/train_query_pairs_{random_seed}.csv', sep='\t', dtype= {'TARGET_ID': str,
                                                                               'TARGET_GRADEID': str,
                                                                               'SOURCE_ID': str,
                                                                               'SOURCE_GRADEID': str})
                                train_pred = add_gold_column(train_pred, train_gold, query_copies)
                                train_pred.to_csv(f'{RESULTS_DIR}/train_{model_filepath.replace(f"{MODEL_SAVE}/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAgeFalse.csv', sep='\t',
                                                   index=False)  # save predictions with gold column

                            dev_pred = pd.read_csv(
                                f'{RESULTS_DIR}/dev_{model_filepath.replace(f"{MODEL_SAVE}/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAgeFalse.csv',
                                sep='\t', dtype={'TARGET_ID': str,
                                                 'TARGET_GRADEID': str,
                                                 'SOURCE_ID': str,
                                                 'SOURCE_GRADEID': str})

                            if 'GOLD' not in dev_pred.columns:
                                dev_gold = pd.read_csv(f'{DATA_DIR}/dev_query_pairs_{random_seed}.csv', sep='\t',
                                                            dtype={'TARGET_ID': str,
                                                          'TARGET_GRADEID': str,
                                                          'SOURCE_ID': str,
                                                          'SOURCE_GRADEID': str})
                                dev_pred = add_gold_column(dev_pred, dev_gold, query_copies)
                                dev_pred.to_csv(f'{RESULTS_DIR}/dev_{model_filepath.replace(f"{MODEL_SAVE}/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAgeFalse.csv', sep='\t',
                                                   index=False)  # save predictions with gold column

                            train_ltr(
                                      train_pred,
                                      dev_pred,
                                      data_dict,
                                      model_filepath,
                                      model_save_path,
                                      random_seed,
                                      age_to_grade,
                                      layers,
                                      query_copies,
                                      DATA_DIR)

                        if mode == 'test' and os.path.isfile(model_save_path) and os.path.isfile(f'{RESULTS_DIR}/{mode}_{model_filepath.replace("../models/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAgeFalse.csv'):

                            results_filepath = f'{RESULTS_DIR}/rerank_ltr_{mode}_{random_seed}_{layers}.csv'
                            test = pd.read_csv(f'{RESULTS_DIR}/{mode}_{model_filepath.replace("../models/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAgeFalse.csv',
                                sep='\t', dtype={'TARGET_ID': str,
                                             'TARGET_GRADEID': str,
                                             'SOURCE_ID': str,
                                             'SOURCE_GRADEID': str})

                            test_gold = pd.read_csv(f'{DATA_DIR}/{mode}_query_pairs_{random_seed}.csv', sep='\t',
                                                    dtype={'TARGET_ID': str,
                                                           'TARGET_GRADEID': str,
                                                           'SOURCE_ID': str,
                                                           'SOURCE_GRADEID': str})
                            if 'GOLD' not in test.columns:
                                test = add_gold_column(test,test_gold,query_copies)
                                test.to_csv(f'{RESULTS_DIR}/{mode}_{model_filepath.replace(f"{MODEL_SAVE}/", "").replace("sentence-transformers/", "")}_top{k}_{features}_filterAgeFalse.csv', sep='\t',
                                                       index=False)
                            ltr_infer(test,
                                      data_dict,
                                      k,
                                      k2,
                                      model_filepath,
                                      model_save_path,
                                      age_to_grade,
                                      layers,
                                      results_filepath,
                                      random_seed,
                                      DATA_DIR)

                            print('Evaluating results...')
                            predictions = pd.read_csv(results_filepath, sep='\t', dtype={'TARGET_ID': str, 'SOURCE_ID': str})
                            eval_filepath = f'{EVAL_DIR}/{results_filepath.replace(f"{RESULTS_DIR}/", "").strip(".csv")}.json'
                            eval_ranking(predictions, test_gold, query_copies, k2, eval_filepath)


    print(f'DONE!\n')


if __name__ == '__main__':
    main()