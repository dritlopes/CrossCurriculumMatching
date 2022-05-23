import lightgbm as lgb
from collections import defaultdict
from scipy.spatial import distance
from sklearn.feature_extraction import DictVectorizer
from sentence_transformers import SentenceTransformer
from utils import find_age
import pandas as pd
import json
import random
import os
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import time


def sample_ids (predictions, query_copies):

    instances, groups = [], []
    # random.seed(7)

    for target, pred_group in predictions.groupby(['TARGET_ID']):

        pred_group = pred_group.to_dict(orient='list')

        pos_queries = []
        for i, pred in enumerate(pred_group['SOURCE_ID']):
            if pred_group['GOLD'][i] == 1:
                instances.append({'target': target,
                                  'source': pred,
                                  'gold': 1})
                pos_queries.append(pred)

        n_neg = 0

        preds = pred_group['SOURCE_ID']
        preds.reverse()
        for pred in preds:
            if n_neg < len(pos_queries):
                if pred not in pos_queries and pred not in [id for pos in pos_queries for ids in query_copies.values() if pos in ids for id in ids]:
                    instances.append({'target': target,
                                    'source': pred,
                                    'gold': 0})

                    n_neg += 1

        groups.append(len(pos_queries)+n_neg)

        # for i in range(len(pos_queries)):
        #     source = random.choice(predictions.SOURCE_ID)
        #     while source in pos_queries or source in [id for pos in pos_queries for ids in query_copies.values() if pos in ids for id in ids]:
        #         source = random.choice(predictions.SOURCE_ID)
        #     instances.append({'target': target,
        #                     'source': source,
        #                     'gold': 0})
        #     n_neg += 1

        assert n_neg == len(pos_queries), f'{len(pos_queries)} positives and {n_neg} negatives'

    return instances, groups


def get_info (data_dict, ids, age_to_grade, features):

    info = defaultdict(dict)

    for cur_id, cur in data_dict.items():
        for grade_id, grade in cur['grade'].items():
            for subject_id, subject in grade['subject'].items():
                for unit_id, unit in subject['unit'].items():
                    for topic_id, topic in unit['topic'].items():
                        for query_id, query in topic['query'].items():
                            if query_id in set([id for id_dict in ids for column, id in id_dict.items() if column != 'gold']):
                                query_dict = dict()
                                query_dict['query'] = query['label']
                                query_dict['doc_titles'] = [doc_dict['title'] for url, doc_dict in query['docs'].items() if doc_dict['pin']]
                                if query_dict['query'] == '': query_dict['query'] = topic['label']
                                if 'grade' in features:
                                    query_dict['age'] = find_age(age_to_grade,curriculum=cur['label'],grade=grade['label'])
                                if 'subject' in features:
                                    query_dict['subject'] = subject['label']
                                if 'unit' in features:
                                    query_dict['unit'] = unit['label']
                                if 'topic' in features:
                                    query_dict['topic'] = topic['label']

                                info[query_id] = query_dict

    return info


def get_higher_layers (path, features, age_to_grade):

    layers = path.split('>')
    hl_dict = defaultdict()
    assert len(layers) == 5, f'{path} is not a complete path'
    if 'topic' in features:
        hl_dict['topic'] = layers[-1]
    if 'unit' in features:
        hl_dict['unit'] = layers[-2]
    if 'subject' in features:
        hl_dict['subject'] = layers[-3]
    if age_to_grade and 'grade' in features:
        age = find_age(age_to_grade,curriculum=layers[0],grade=layers[-4])
        hl_dict['age'] = age

    return hl_dict


def compare_higher_layers(target,source,cos_dict,model):

    for k, v in target.items():
        if k == 'age':
            target_vector, source_vector = [], []
            for i in range(0,19):
                if int(v) <= i: target_vector.append(1)
                else: target_vector.append(0)
                if int(source[k]) <= i: source_vector.append(1)
                else: source_vector.append(0)
            target_vector = np.array(target_vector)
            source_vector = np.array(source_vector)
        elif k in ['subject','unit','topic']:
            if v == '':
                target_vector = np.zeros((384,))
            else:
                target_vector = model.encode(v)
            if source[k] == '':
                source_vector = np.zeros((384,))
            else:
                source_vector = model.encode(source[k])
        if k not in ['doc_titles','query']:
            sim_score = 1 - distance.cosine(target_vector,source_vector)
            cos_dict[k] = sim_score

    return cos_dict


def read_in_data(data):

    instances, groups = [],[]

    for name, group in data.groupby(['TARGET_ID']):

        for source, gold in zip(group['SOURCE_ID'].tolist(), group['GOLD'].tolist()):
            instances.append({'target': name,
                                'source': source,
                                'gold': gold})

        groups.append(len(group['SOURCE_ID'].tolist()))

    assert len(instances) == sum(groups), 'Length of instances is not the same as the sum of group sizes'

    return instances, groups


def average_embeddings (model, source_features, query_encoding):

    if 'doc_titles' in source_features.keys():
        query_doc_titles = source_features['doc_titles']
        if query_doc_titles == []: query_doc_titles = ['']
        title_encodings = model.encode(query_doc_titles)
        title_encodings = np.mean(title_encodings, axis=0)
        query_encoding = np.stack((query_encoding, title_encodings))

    if 'doc_sums' in source_features.keys():
        query_doc_sums = source_features['doc_sums']
        if query_doc_sums == []: query_doc_sums = ['']
        sum_encodings = model.encode(query_doc_sums)
        sum_encodings = np.mean(sum_encodings,axis=0)
        if query_encoding.shape[0] == 2:
            query_encoding = np.stack((query_encoding[0],query_encoding[1],sum_encodings))
        else:
            query_encoding = np.stack((query_encoding, sum_encodings))

    query_encoding = np.mean(query_encoding, axis=0)

    return query_encoding


def compare_query(target,source,model):

    cos_dict = dict()

    target_query = model.encode(target['query'])
    source_query = model.encode(source['query'])
    source_query = average_embeddings(model, source, source_query)
    cos_dict['query'] = 1 - distance.cosine(target_query, source_query)

    return cos_dict

def get_groups_from_file(input):

    n = 0
    groups = []
    group_column = input.tolist()
    while n != len(group_column):
        i = group_column[n]
        groups.append(i)
        n += i
    return groups

def prepare_data (predictions, data_dict, model_filepath, DATA_DIR, features, age_to_grade, mode, random_seed, query_copies=None, save_cosine=True):

    cos_data, y, groups = [], [], []

    if os.path.isfile(f'{DATA_DIR}/ltr_input_{mode}_{random_seed}_grade,subject,topic.csv'):
        input = pd.read_csv(f'{DATA_DIR}/ltr_input_{mode}_{random_seed}_grade,subject,topic.csv', sep='\t',dtype={'y':int,'group':int})
        y = input['y'].tolist()
        groups = get_groups_from_file(input['group'])
        assert len(input['y']) == sum(groups), 'Sum of queries in groups is not the same as in data'
        features_to_include = features.split(',')
        features_to_include.append('query')
        if 'grade' in features_to_include:
            features_to_include.remove('grade')
            features_to_include.append('age')
        input = input[features_to_include]
        cos_data = input.to_dict('records')

    else:
        if mode == 'train': instances, groups = sample_ids(predictions, query_copies)
        elif mode == 'test': instances, groups = read_in_data(predictions)
        info = get_info(data_dict, instances, age_to_grade, features)
        # print(info['30311'])
        # print(info['30310'])
        model = SentenceTransformer(model_filepath)
        for pair in instances:
            y.append(pair['gold'])
            target = info[pair['target']]
            source = info[pair['source']]
            if mode == 'train':
                cos_dict = compare_query(target, source, model)
            elif mode == 'test':
                for t, s, score in zip(predictions.TARGET_ID, predictions.SOURCE_ID, predictions.SCORE):
                    if t == pair['target'] and s == pair['source']:
                        cos_dict = {'query': score}
            cos_dict = compare_higher_layers(target, source, cos_dict, model)
            cos_data.append(cos_dict)
            # print(cos_data[1], cos_data[26])

        assert len(cos_data) == len(instances)

        if save_cosine:
            input = pd.DataFrame.from_records(cos_data)
            input['group'] = [g for i in groups for g in [i] * i]
            input['y'] = y
            input.to_csv(f'{DATA_DIR}/ltr_input_{mode}_{random_seed}_{features}.csv', sep='\t',index=False)

    vec = DictVectorizer()
    x = vec.fit_transform(cos_data)
    print(f'Input vector feature names: {vec.feature_names_}')
    # print(x[1])
    # print(x[26],'\n')

    return x, groups, y

# from https://www.kaggle.com/code/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761/notebook
def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3
# reference stops here

def train_ltr (train_predictions, dev_predictions, data_dict, model_filepath, model_save_path, random_seed, age_to_grade, features, query_copies, DATA_DIR):

    print(f'Training LambdaMART with features {features+",query"}')
    start_time = time.perf_counter()
    train, train_groups, train_gold = prepare_data(train_predictions,data_dict, model_filepath, DATA_DIR, features, age_to_grade, 'train', random_seed, query_copies)
    dev, dev_groups, dev_gold = prepare_data(dev_predictions,data_dict, model_filepath, DATA_DIR, features, age_to_grade, 'train', random_seed, query_copies)
    # print(train.shape, len(train_groups), len(train_gold))

    feature_names = features.replace('grade', 'age').split(',')
    feature_names.append('query')
    feature_names.sort()

    # hyper-parameter tuning: not possible because sklearn does not support ranking models for model selection (group parameter cannot be passed to estimator)
    # params = {'num_leaves': randint(10,100),
    #           #'max_depth': randint(1,10),
    #           'boosting': ['gbdt','dart','goss'],
    #           'n_estimators' : randint(20,50)}
    # fit_parameters = {'group': train_groups,
    #                 'eval_set': [(dev, dev_gold)],
    #                 "eval_group": [dev_groups],
    #                 "early_stopping_rounds": 15,
    #                 "eval_metric" : ['mrr','binary_logloss'],
    #                 "feature_name" : feature_names}
    # gbm = lgb.LGBMRanker(random_state=random_seed)
    # rs = RandomizedSearchCV(
    #     estimator=gbm, param_distributions=params,
    #     n_iter=10,
    #     scoring=['average_precision', 'top_k_accuracy'],
    #     refit='average_precision',
    #     random_state=random_seed,
    #     verbose=True)
    # rs.fit(train,train_gold,**fit_parameters)
    # print('Best score reached: {} with params: {} '.format(rs.best_score_, rs.best_params_))
    #
    # # train final model
    # gbm = lgb.LGBMRanker(**rs.best_estimator_.get_params())
    # model = gbm.fit(train,train_gold,**fit_parameters,
    #                 callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])

    gbm = lgb.LGBMRanker(random_state=random_seed,
                         num_leaves=50,
                         max_depth=5,
                         n_estimators=20)

    model = gbm.fit(train, train_gold,
                    group=train_groups,
                    eval_set=[(dev, dev_gold)],
                    eval_group=[dev_groups],
                    early_stopping_rounds=15,
                    eval_metric=['mrr','binary_logloss'],
                    feature_name=feature_names,
                    callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])

    model.booster_.save_model(f'{model_save_path}')

    end_time = time.perf_counter()
    seconds = end_time - start_time
    convert = time.strftime("%H:%M:%S", time.gmtime(seconds))
    print(f"Training DONE! It took {convert} to train LTR")

    # print(model.evals_result_.values()[0])
    # eval = model.evals_result_.values()[0].items()
    # eval_dict = {}
    # for top_k in enumerate(eval):
    #     eval_dict[top_k[0]] = top_k[1]
    # with open("../eval/train_ltr_eval.json", "w") as outfile:
    #     json.dump(eval_dict, outfile)


def ltr_infer (test, data_dict, k, k2, model_filepath, model_save_path, age_to_grade, features, results_filepath, random_seed, DATA_DIR):

    print('Re-ranking with LTR...')
    reranking = pd.DataFrame()
    model = lgb.Booster(model_file=model_save_path)
    print(f'LTR model features: {model.feature_name()}')
    print(f'LTR feature importance: {model.feature_importance()}')
    x, groups, y = prepare_data(test, data_dict, model_filepath, DATA_DIR, features, age_to_grade, 'test', random_seed)
    targets = list(test.groupby(['TARGET_ID']))
    split = range(k, x.shape[0], k)
    target_index = range(0, len(groups))

    # score = model.predict(x[1])
    # scores = model.predict(x[1],pred_contrib=True)
    # print(score)
    # print(scores, '\n')
    # score = model.predict(x[26])
    # scores = model.predict(x[26],pred_contrib=True)
    # print(score)
    # print(scores, '\n')
    # exit()

    for i in target_index:
        if i == 0:
            scores = model.predict(x[:split[i]])
        elif i != 0 and i != (len(groups)-1):
            scores = model.predict(x[split[i-1]:split[i]])
        else: # if i == len(groups)
            scores = model.predict(x[split[i-1]:])

        info = targets[i][1]
        info['re-score'] = list(scores)
        info = info.sort_values(by=['re-score'], ascending=False)
        reranking = pd.concat([reranking,info.head(n=k2)])

    reranking.to_csv(results_filepath, sep='\t', index=False)
