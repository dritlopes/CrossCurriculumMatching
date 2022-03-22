import copy
from utils import find_age, grade_by_age
import json
import time
import requests
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sentence_transformers import SentenceTransformer

def filter_grade (target_grade, target_curriculum, age_to_grade):

    grades_to_include = []
    target_age = find_age(age_to_grade,target_curriculum,target_grade)

    # when target grade is not in 'Reading levels and age filter settings' file, filter age will have no effect
    if target_age == -1: grades_to_include = None

    else:
        for age in [target_age, target_age - 1, target_age + 1, target_age + 2, target_age - 2]:
            if age in age_to_grade.keys():
                for grade in age_to_grade[age]:
                    if grade['CURRICULUM'] != target_curriculum: grades_to_include.append(grade['GRADEID'])

    return grades_to_include


def get_search_space (data, filters, target_curriculum, filterAge = False, target_grade = None, age_to_grade = None):

    data_filtered = copy.deepcopy(data)

    for cur_id, cur in data.items():
        if filters['curriculums'] and cur['label'] not in filters['curriculums']:
            data_filtered.pop(cur_id)
            continue
        elif cur['label'] == target_curriculum:
            data_filtered.pop(cur_id)
            continue

        for grade_id, grade in cur['grade'].items():
            if filters['grades'] and grade['label'] not in filters['grades']:
                data_filtered[cur_id]['grade'].pop(grade_id)
                continue
            if target_grade and filterAge and age_to_grade:
                grades_to_include = filter_grade(target_grade, target_curriculum, age_to_grade)
                if grade_id not in grades_to_include:
                    data_filtered[cur_id]['grade'].pop(grade_id)

            for subject_id, subject in grade['subject'].items():
                if filters['subjects'] and subject['label'] not in filters['subjects']:
                    data_filtered[cur_id]['grade'][grade_id]['subject'].pop(subject_id)
                    continue

    # with open('../data/data_filtered.json', 'w') as outfile:
    #     json.dump(data_filtered, outfile)

    return data_filtered

# def get_source_features (source, features, age_to_grade = None):
#
#     start_time = time.perf_counter()
#
#     source_features = []
#     for cur_id, cur in source.items():
#         for grade_id, grade in cur['grade'].items():
#             for subject_id, subject in grade['subject'].items():
#                 for unit_id, unit in subject['unit'].items():
#                     for topic_id, topic in unit['topic'].items():
#                         for query_id, query in topic['query'].items():
#
#                             source_dict = dict()
#                             source_dict['label'] = query['label']
#
#                             if source_dict['label'] == '':
#                                 source_dict['label'] = topic['label']
#
#                             if 'doc_title' in features:
#                                 source_dict['doc_titles'] = []
#                                 for doc_url, doc_info in query['docs'].items():
#                                     if doc_info['pin']:
#                                         source_dict['doc_titles'].append(doc_info['title'])
#
#                             if 'age' in features and age_to_grade:
#                                 age = find_age(age_to_grade, cur['label'], grade['label'])
#                                 source_dict['age'] = age
#
#                             if 'subject' in features:
#                                 source_dict['subject'] = subject['label'].strip()
#
#                             if 'doc_sum' in features:
#                                 source_dict['doc_sums'] = []
#                                 url = f'https://api.wizenoze.com/v4/curriculum/node/query/{query_id}' \
#                                       f'/results?userUUID=123456&sessionUUID=123456&userType=teacher&resultSize=3'
#                                 headers = {'Authorization': '0b9cb12f-b960-47cb-b7fe-f47253cc4f1e'}
#                                 response = requests.get(url, headers=headers)
#                                 if response:
#                                   resp = response.json()["results"]
#                                   for doc_resp in resp:
#                                       # get summary of pinned docs only
#                                       if doc_resp["fullUrl"] in query['docs'].keys():
#                                             if query["docs"][doc_resp["fullUrl"]]["pin"]:
#                                                 source_dict["doc_sums"].append(doc_resp["summaryText"])
#                                 else: raise Exception(f'Could not get a response from API call for query {query_id} with url {url}')
#
#                             source_features.append(source_dict)
#
#     end_time = time.perf_counter()
#     print(f"It took {end_time - start_time:0.4f} seconds to get search space feature values")
#
#     return source_features
#
# def average_embeddings (model,source_features,label_encodings):
#
#     source_encodings = []
#     for i in range(0,len(label_encodings)):
#         query_encoding = label_encodings[i]
#
#         if 'doc_titles' in source_features[0].keys():
#             query_doc_titles = source_features[i]['doc_titles']
#             # average the encodings only if doc titles is not an empty list (if at least one result is pinned)
#             if query_doc_titles != []:
#                 title_encodings = model.encode(query_doc_titles)
#                 query_encoding = np.vstack((query_encoding,title_encodings))
#                 print(query_encoding.shape)
#         if 'doc_sums' in source_features[0].keys():
#             query_doc_sums = source_features[i]['doc_sums']
#             if query_doc_sums != []:
#                 sum_encodings = model.encode(query_doc_sums)
#                 query_encoding = np.vstack((query_encoding, sum_encodings))
#                 print(query_encoding.shape)
#         if 'doc_titles' or 'doc_sums' in source_features[0].keys():
#             query_encoding = np.mean(query_encoding, axis=0)
#             print(query_encoding.shape)
#
#         source_encodings.append(query_encoding)
#
#     return source_encodings
#
# def get_source_encodings(source_features, embedding_model):
#
#     # sbert fine-tuned for semantic search with queries and documents from wizenoze
#     if embedding_model == 'wize-sbert':
#         model_filepath = '../models/wize_biencoder_model'
#         model = SentenceTransformer(model_filepath)
#
#     # sbert pre-trained on canonical data: AllNLI, sentence-compression, SimpleWiki, altlex, msmarco-triplets, quora_duplicates, coco_captions,flickr30k_captions, yahoo_answers_title_question, S2ORC_citation_pairs, stackexchange_duplicate_questions, wiki-atomic-edits
#     elif embedding_model == 'sbert':
#         model_filepath = "sentence-transformers/paraphrase-MiniLM-L6-v2"
#         model = SentenceTransformer(model_filepath)
#
#     # source embeddings
#     start_time = time.perf_counter()
#     source_labels = [source_dict.pop('label') for source_dict in source_features]
#     source_encodings = model.encode(source_labels)
#
#     if 'doc_titles' or 'doc_sums' in source_features[0].keys():
#         source_encodings = average_embeddings(model, source_features, source_encodings)
#     end_time = time.perf_counter()
#     print(f"It took {end_time - start_time:0.4f} seconds to encode search space")
#
#     if source_features[0].keys():
#         vec = DictVectorizer()
#         source_vectors = vec.fit_transform(source_features).toarray()
#         source_encodings = np.concatenate((source_vectors, np.array(source_encodings)), axis=1)
#
#     return source_encodings
#
# def encode_search_space(source, embedding_model, features, age_to_grade=None):
#     pass
#     source_features = get_source_features(source, features, age_to_grade)
#     source_encodings = get_source_encodings(source_features, embedding_model)

