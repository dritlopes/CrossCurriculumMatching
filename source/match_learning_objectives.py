from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from collections import defaultdict
from utils import find_age, get_info_for_pred
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import requests
import time
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
import fasttext.util
import fasttext

def get_features (source, target, features, age_to_grade = None, doc_sums = None):

    source_features = []
    for cur_id, cur in source.items():
        for grade_id, grade in cur['grade'].items():
            for subject_id, subject in grade['subject'].items():
                for unit_id, unit in subject['unit'].items():
                    for topic_id, topic in unit['topic'].items():
                        for query_id, query in topic['query'].items():

                            source_dict = dict()
                            source_dict['label'] = query['label']

                            if source_dict['label'] == '':
                                source_dict['label'] = topic['label']

                            if 'doc_title' in features:
                                source_dict['doc_titles'] = []
                                for doc_url, doc_info in query['docs'].items():
                                    if doc_info['pin']:
                                        source_dict['doc_titles'].append(doc_info['title'])

                            if 'age' in features and age_to_grade:
                                age = find_age(age_to_grade, cur['label'], grade['label'])
                                source_dict['age'] = age

                            if 'subject' in features:
                                source_dict['subject'] = subject['label'].strip()

                            if 'doc_sum' in features:
                                source_dict['doc_sums'] = []
                                if doc_sums[query_id]:
                                    sentences = []
                                    for doc in doc_sums[query_id]:
                                        # sentences.append(sent_tokenize(doc)[0])
                                        sents = sent_tokenize(doc)
                                        tags_label = pos_tag(word_tokenize(query['label']))
                                        nouns = [word for word, pos in tags_label if pos.startswith('NN')]
                                        for sent in sents:
                                            tokens = word_tokenize(sent)
                                            for token in tokens:
                                                if token in nouns and sent not in sentences:
                                                    sentences.append(sent)
                                    source_dict["doc_sums"] = sentences

                            source_features.append(source_dict)

    target_features = []
    for i in range(0, len(target['TARGET'])):
        target_dict = defaultdict()
        target_dict['label'] = target['TARGET'][i]
        if 'TARGET_PATH' in target.keys():
            if 'age' in features:
                age = find_age(age_to_grade, target['TARGET_CURRICULUM'][i], target['TARGET_PATH'][i].split('>')[1])
                target_dict['age'] = age
            if 'subject' in features:
                subject = target['TARGET_PATH'][i].split('>')[2]
                target_dict['subject'] = subject
        target_features.append(target_dict)

    return source_features, target_features


def average_embeddings (model, model_name, source_features,label_encodings):

    source_encodings = []

    for i in range(0,len(label_encodings)):

        query_encoding = label_encodings[i]

        if 'doc_titles' in source_features[0].keys():
            query_doc_titles = source_features[i]['doc_titles']
            # average the encodings only if doc titles is not an empty list (if at least one result is pinned)
            if query_doc_titles != []:
                if model_name == '../models/cc.en.300.bin':
                    title_encodings = []
                    for title in query_doc_titles:
                        title_encodings.append(model.get_sentence_vector(title.replace('\n','')))
                else:
                    title_encodings = model.encode(query_doc_titles)
                title_encodings = np.mean(np.array(title_encodings), axis=0)
                query_encoding = np.vstack((np.array(query_encoding),title_encodings))
                # print(query_encoding.shape)

        if 'doc_sums' in source_features[0].keys():
            query_doc_sums = source_features[i]['doc_sums']
            if query_doc_sums != []:
                if model_name == '../models/cc.en.300.bin':
                    sum_encodings = []
                    for sum in query_doc_sums:
                        sum_encodings.append(model.get_sentence_vector(sum.replace('\n','')))
                else:
                    sum_encodings = model.encode(query_doc_sums)
                sum_encodings = np.mean(np.array(sum_encodings),axis=0)
                query_encoding = np.vstack((np.array(query_encoding), sum_encodings))

        query_encoding = np.mean(query_encoding, axis=0)

        source_encodings.append(query_encoding)

    return source_encodings


def get_encodings (source_features, target_features, model_filepath):

    start_time = time.perf_counter()

    source_labels = [source_dict.pop('label') for source_dict in source_features]
    target_labels = [target_dict.pop('label') for target_dict in target_features]

    if model_filepath == '../models/cc.en.300.bin':
        # fasttext.util.download_model('en', if_exists='ignore')
        model = fasttext.load_model(model_filepath)
        source_encodings, target_encodings = [], []
        for label in source_labels: source_encodings.append(model.get_sentence_vector(label))
        for label in target_labels: target_encodings.append(model.get_sentence_vector(label))

    else:
        model = SentenceTransformer(model_filepath)
        source_encodings = model.encode(source_labels)
        target_encodings = model.encode(target_labels)

    if 'doc_titles' in source_features[0].keys() or 'doc_sums' in source_features[0].keys():
        source_encodings = average_embeddings(model, model_filepath, source_features, source_encodings)

    end_time = time.perf_counter()
    print(f"It took {end_time - start_time:0.4f} seconds to encode queries")


    # if there are other features than label and doc
    if source_features[0].keys() and target_features[0].keys():
        # TODO test if source and target do not have the same keys (e.g. subject key is not here for target)
        vec = DictVectorizer()
        source_vectors = vec.fit_transform(source_features).toarray()
        target_vectors = vec.transform(target_features).toarray()
        # print(source_vectors.shape)
        # print(vec.get_feature_names_out())
        # print(target_vectors.shape)
        # print(np.array(source_encodings).shape)
        source_encodings = np.concatenate((source_vectors, np.array(source_encodings)), axis=1)
        # print(source_encodings.shape)
        # print(np.array(target_encodings).shape)
        target_encodings = np.concatenate((target_vectors, np.array(target_encodings)),axis=1)
        # print(target_encodings.shape)

    return source_encodings, target_encodings


def rank_queries(source_vectors, source_info, target_vectors, target_info, topn):

    predictions = defaultdict(dict)

    for t_vec,t_dict in zip(target_vectors, target_info):
        scores = []
        for s_vec, s_dict in zip(source_vectors,source_info):
            sim_score = 1 - distance.cosine(t_vec, s_vec)
            scores.append((s_dict['label'],s_dict['id'],s_dict['path'],sim_score))

        scores.sort(key=lambda x:x[-1], reverse=True)
        predictions[t_dict["id"]] = {'label': t_dict['label'],
                                     'path' : t_dict['path'],
                                     'scores': scores[:topn]}

    return predictions


def find_best_queries (source, target, embedding_model, features, topn = 5, age_to_grade = None, doc_sums_dict = None):

    source_features, target_features = get_features(source, target, features, age_to_grade, doc_sums_dict)
    source_encodings, target_encodings = get_encodings(source_features, target_features, embedding_model)
    source_info, target_info = get_info_for_pred(source,target)
    predictions = rank_queries(source_encodings,source_info, target_encodings, target_info, topn)

    return predictions

#
#
# def get_input_text (source, target):
#
#     source_text = [instance['label'] for instance in source]
#     # source_text = list(set([instance['label'] for instance in source]))
#     target_text = list(dict.fromkeys(target['learning objective'].tolist()))
#
#     return source_text, target_text
#
#
# def get_embeddings (model_name,source_texts,target_texts):
#
#     if model_name == 'wize-sbert':
#         model_filepath = '../models/biencoder_model'
#         model = SentenceTransformer(model_filepath)
#         target_embeddings = model.encode(target_texts, convert_to_numpy = True)
#         source_embeddings = model.encode(source_texts, convert_to_numpy = True)
#
#     else: raise Exception(f'Model name {model_name} is unknown. Choose from the options <wize-sbert>...')
#
#     return target_embeddings, source_embeddings
#
#
# def create_feature_vectors (source, target, age_to_grade = None, target_grade = None, target_curriculum = None):
#
#     source_dicts = []
#     target_dicts = []
#     vec = DictVectorizer()
#
#     # source
#     for info in source:
#         info_dict = dict()
#         grade = info['path'].split('>')[1]
#         cur = info['path'].split('>')[0]
#         if age_to_grade:
#             age = find_age(age_to_grade, cur, grade)
#             info_dict['age'] = age
#         source_dicts.append(info_dict)
#
#     # target
#     if age_to_grade and target_grade and target_curriculum:
#         target_age = find_age(age_to_grade, target_curriculum, target_grade)
#         for i in range(0, len(list(dict.fromkeys(target['learning objective'].tolist())))):
#             info_dict = dict()
#             info_dict['age'] = target_age
#             target_dicts.append(info_dict)
#
#     source_vec = vec.fit_transform(source_dicts).toarray()
#     # print(source_vec.shape)
#     # print(vec.get_feature_names_out())
#     target_vec = vec.transform(target_dicts).toarray()
#     # print(target_vec.shape)
#
#
#     return np.array(source_vec), np.array(target_vec)
#
#
# def match_lo (source, target, model_name, topn, target_grade = None, target_curriculum = None, age_to_grade = None):
#
#     source_text, target_text = get_input_text(source, target)
#     source_id = [s['id'] for s in source]
#
#     target_embeddings, source_embeddings = get_embeddings (model_name, source_text, target_text)
#     # print(source_embeddings.shape)
#     # print(target_embeddings.shape)
#
#     # age as one-hot encodings
#     if age_to_grade and target_grade and target_curriculum:
#
#         one_hot_source_vec, one_hot_target_vec = create_feature_vectors(source,target,age_to_grade, target_grade, target_curriculum)
#         source_vectors, target_vectors = [],[]
#
#         for i in range(0,one_hot_source_vec.shape[0]):
#             source_vectors.append(list(one_hot_source_vec[i])+list(source_embeddings[i]))
#
#         for i in range(0,one_hot_target_vec.shape[0]):
#             target_vectors.append(list(one_hot_target_vec[i])+list(target_embeddings[i]))

        # print(np.array(source_vectors).shape)
        # print(np.array(target_vectors).shape)
    #
    #     matches = rank_queries(source_vectors, source_text, source_id, target_vectors, target_text,topn)
    #
    # else: matches = rank_queries(source_embeddings, source_text, source_id, target_embeddings, target_text,topn)
    #
    #
    #
    # return matches

