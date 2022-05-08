from collections import defaultdict
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction import DictVectorizer
import time
import pandas as pd
import numpy as np
from utils import find_age

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


def get_cosines (predictions, model_filepath, features, age_to_grade):

    model = SentenceTransformer(model_filepath)
    predictions = predictions.sort_values(['TARGET_CURRICULUM', 'TARGET_ID'])
    cos_sims, info = [], []

    start_time = time.perf_counter()
    for pred_name, pred_group in predictions.groupby(['TARGET_CURRICULUM', 'TARGET_ID']):

        pred_dict = pred_group.to_dict(orient='list')
        info.append(pred_dict)
        target_hl = get_higher_layers(pred_dict['TARGET_PATH'][0], features, age_to_grade)
        cos_per_target = []

        for i, candidate in enumerate(pred_dict['SOURCE_ID']):
            cos_dict = defaultdict()
            cos_dict['query'] = pred_dict['SCORE'][i]
            # get higher layer info
            source_hl = get_higher_layers(pred_dict['SOURCE_PATH'][i], features, age_to_grade)
            cos_dict = compare_higher_layers(target_hl, source_hl, cos_dict, model)
            cos_per_target.append(cos_dict)

        cos_sims.append(cos_per_target)

    assert len([cos_dict for cos_list in cos_sims for cos_dict in cos_list]) == len(predictions), \
        f'Length is not the same, {len([cos_dict for cos_list in cos_sims for cos_dict in cos_list])} != {len(predictions)}'

    end_time = time.perf_counter()
    print(f"{end_time - start_time:0.4f} seconds to get cosine similarities")

    return cos_sims, info


def rerank (predictions, model_filepath, results_filepath, features = 'topic', k = 5, method = 'average', age_to_grade = None):

    cos_sims, info = get_cosines(predictions, model_filepath, features, age_to_grade)
    rankings = defaultdict(list)

    if method == 'average':
        for cos_list, target_info in zip(cos_sims,info):

            scores = []
            for i, cos_dict in enumerate(cos_list):
                avg = sum([cos for cos in cos_dict.values()])/len(cos_dict.values())
                scores.append((target_info['SOURCE'][i],
                               target_info['SOURCE_ID'][i],
                               target_info['SOURCE_PATH'][i],
                               avg))
            scores.sort(key=lambda x: x[-1], reverse=True)

            for i,score in enumerate(scores[:k]):
                rankings['TARGET_CURRICULUM'].append(target_info['TARGET_CURRICULUM'][i])
                rankings['TARGET_ID'].append(target_info['TARGET_ID'][i])
                rankings['TARGET'].append(target_info['TARGET'][i])
                rankings['TARGET_PATH'].append(target_info['TARGET_PATH'][i])
                rankings['SOURCE_ID'].append(score[1])
                rankings['SOURCE'].append(score[0])
                rankings['SOURCE_PATH'].append(score[2])
                rankings['SCORE'].append(score[-1])

    elif method == 'LTR':
        data_dicts = [cos_dict for cos_list in cos_sims for cos_dict in cos_list]
        vec = DictVectorizer()
        data_vectors = vec.fit_transform(data_dicts)
        pass

    rankings = pd.DataFrame.from_dict(rankings)
    rankings.to_csv(results_filepath,index=False,sep='\t')
