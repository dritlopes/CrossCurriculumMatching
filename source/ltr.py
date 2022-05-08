import lightgbm as lgb
from collections import defaultdict
from match_higher_layers import compare_higher_layers
from match_learning_objectives import average_embeddings
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

def sample_ids (gold_pairs, predictions, query_copies):

    instances, groups = [], []
    random.seed(7)

    for gold, pred in zip(gold_pairs.groupby(['TARGET_ID']), predictions.groupby(['TARGET_ID'])):

        target, gold_group = gold[0], gold[1]
        pred_target, pred_group = pred[0], pred[1]

        assert target == pred_target

        pos_queries = gold_group['SOURCE_ID'].tolist()
        # info[target] = {'label': gold_group['TARGET'].tolist()[0],
        #                 'path': gold_group['TARGET_PATH'].tolist()[0]}

        for source in pos_queries:
            instances.append({'target': target,
                            'source': source,
                            'gold': 1})
        n_neg = 0
        # for i in range(len(pos_queries)):
        #     source = random.choice(predictions.SOURCE_ID)
        #     while source in pos_queries or source in [id for pos in pos_queries for ids in query_copies.values() if pos in ids for id in ids]:
        #         source = random.choice(predictions.SOURCE_ID)
        #     instances.append({'target': target,
        #                     'source': source,
        #                     'gold': 0})
        #     n_neg += 1

        for pred in pred_group['SOURCE_ID'].tolist(): # .reverse()
            if n_neg < len(pos_queries):
                if pred not in pos_queries and pred not in [id for pos in pos_queries for ids in query_copies.values() if pos in ids for id in ids]:
                    instances.append({'target': target,
                                    'source': pred,
                                    'gold': 0})

                    n_neg += 1

        groups.append(len(pos_queries)+n_neg)

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

def compare_query(target,source,model,model_filepath):

    cos_dict = dict()

    target_query = model.encode(target['query'])
    source_query = model.encode(source['query'])
    source_query = average_embeddings(model, model_filepath, [source], [source_query])[0]  # add doc title info
    cos_dict['query'] = 1 - distance.cosine(target_query, source_query)

    return cos_dict

def read_in_data(data):

    instances, groups = [],[]

    for name, group in data.groupby(['TARGET_ID']):
        # if name == '10115':
            for source, gold in zip(group['SOURCE_ID'].tolist(), group['gold'].tolist()):
                instances.append({'target': name,
                                'source': source,
                                'gold': gold})
            groups.append(len(group['SOURCE_ID'].tolist()))

    return instances, groups

def prepare_data (gold, data_dict, model_filepath, features, age_to_grade, mode, random_seed, query_copies=None, predictions=None, save_cosine=True):

    cos_data, y, groups = [], [], []

    if os.path.isfile(f'../data/ltr_input_{mode}_{random_seed}_{features}.json'):
       input = pd.read_csv(f'../data/ltr_input_{mode}_{random_seed}_{features}.json', sep='\t',dtype={'y':int,'group':int})
       y = input['y'].tolist()
       # TODO get group info
       input = input.drop(['y','group'],axis=1)
       cos_data = input.to_dict('records')
       print(cos_data[0])

    else:
        if mode == 'train': instances, groups = sample_ids(gold, predictions, query_copies)
        elif mode == 'test': instances, groups = read_in_data(gold)
        info = get_info(data_dict, instances, age_to_grade, features)
        # print(info['30311'])
        # print(info['30310'])
        model = SentenceTransformer(model_filepath)
        for pair in instances:
            y.append(pair['gold'])
            target = info[pair['target']]
            source = info[pair['source']]
            if mode == 'train':
                cos_dict = compare_query(target, source, model, model_filepath)
            elif mode == 'test':
                for t, s, score in zip(gold.TARGET_ID, gold.SOURCE_ID, gold.SCORE):
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
            input.to_csv(f'../data/ltr_input_{mode}_{random_seed}_{features}.json', sep='\t',index=False)

    vec = DictVectorizer()
    x = vec.fit_transform(cos_data)
    print(vec.feature_names_)
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

def train_ltr (train_data, train_predictions, dev_data, dev_predictions, data_dict, model_filepath, model_save_path, random_seed, age_to_grade, features, query_copies):

    print(f'Training LambdaMART with features {features+",query"}')
    train, train_groups, train_gold = prepare_data(train_data, data_dict, model_filepath, features, age_to_grade, 'train', random_seed, query_copies, train_predictions)
    dev, dev_groups, dev_gold = prepare_data(dev_data, data_dict, model_filepath, features, age_to_grade, 'train', random_seed, query_copies, dev_predictions)
    # print(train.shape, len(train_groups), len(train_gold))

    feature_names = features.replace('grade', 'age').split(',')
    feature_names.append('query')
    feature_names.sort()

    # hyper-parameter tuning
    params = {'num_leaves': randint(10,100),
              #'max_depth': randint(1,10),
              'boosting': ['gbdt','dart','goss'],
              'n_estimators' : randint(20,50)}
    fit_parameters = {'group': train_groups,
                    'eval_set': [(dev, dev_gold)],
                    "eval_group": [dev_groups],
                    "early_stopping_rounds": 15,
                    "eval_metric" : ['map','binary_logloss'],
                    "feature_name" : feature_names}
    gbm = lgb.LGBMRanker(random_state=random_seed)
    rs = RandomizedSearchCV(
        estimator=gbm, param_distributions=params,
        n_iter=10,
        scoring=['average_precision'],
        refit=True,
        random_state=random_seed,
        verbose=True)
    rs.fit(train,train_gold,**fit_parameters)
    print('Best score reached: {} with params: {} '.format(rs.best_score_, rs.best_params_))

    # train final model
    gbm = lgb.LGBMRanker(**rs.best_estimator_.get_params())
    model = gbm.fit(train,train_gold,**fit_parameters,
                    callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])

    model.booster_.save_model(f'{model_save_path}')
    print(model.feature_importances_)

    # print(model.evals_result_.values()[0])
    # eval = model.evals_result_.values()[0].items()
    # eval_dict = {}
    # for top_k in enumerate(eval):
    #     eval_dict[top_k[0]] = top_k[1]
    # with open("../eval/train_ltr_eval.json", "w") as outfile:
    #     json.dump(eval_dict, outfile)


def ltr_infer (test, data_dict, k, k2, model_filepath, model_save_path, age_to_grade, features, results_filepath):

    reranking = pd.DataFrame()
    model = lgb.Booster(model_file=model_save_path)
    print(model.feature_name())
    print(model.feature_importance())
    x, groups, y = prepare_data(test, data_dict, model_filepath, features, age_to_grade, 'test')
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
