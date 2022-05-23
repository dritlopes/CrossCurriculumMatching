import warnings
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from collections import defaultdict
from utils import find_age
import numpy as np
import time
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
import fasttext.util
import fasttext
from classifier import pre_processor, PairDataset, Model
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from train_classifier import format_time
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn as nn


def get_target_features (target, features, age_to_grade, uncased):

    target_features = []

    for i in range(0, len(target['TARGET'])):

        target_dict = dict()

        target_dict['query'] = target['TARGET'][i].strip()
        if uncased: target_dict['query'] = target_dict['query'].lower()

        if 'TARGET_PATH' in target.keys():
            layers = target['TARGET_PATH'][i].split('>')
            if 'grade' in features:
                age = find_age(age_to_grade, target['TARGET_CURRICULUM'][i], layers[1].strip())
                target_dict['age'] = age
            if 'subject' in features:
                target_dict['subject'] = layers[2].strip()
                if uncased: target_dict['subject'] = target_dict['subject'].lower()
            if 'topic' in features:
                target_dict['topic'] = layers[-1].strip()
                if uncased: target_dict['topic'] = target_dict['topic'].lower()

        target_features.append(target_dict)

    print(f'N of target learning objectives: {len(target_features)}')
    # warnings.warn(f'N of target learning objetives: {len(target_features)}')

    return target_features


def get_source_features (source, features, age_to_grade, uncased, doc_sums):

    source_features = []

    for cur_id, cur in source.items():
        for grade_id, grade in cur['grade'].items():
            for subject_id, subject in grade['subject'].items():
                for unit_id, unit in subject['unit'].items():
                    for topic_id, topic in unit['topic'].items():
                        for query_id, query in topic['query'].items():

                            source_dict = dict()
                            source_dict['query'] = query['label'].strip()
                            if uncased: source_dict['query'] = source_dict['query'].lower()

                            if source_dict['query'] == '':
                                source_dict['query'] = topic['label'].strip()
                                if uncased: source_dict['query'] = source_dict['query'].lower()

                            if 'doc_title' in features:
                                source_dict['doc_titles'] = []
                                for doc_url, doc_info in query['docs'].items():
                                    if doc_info['pin']:
                                        doc_title = doc_info['title']
                                        if uncased: doc_title.lower()
                                        source_dict['doc_titles'].append(doc_title)

                            if 'grade' in features and age_to_grade:
                                grade_name = grade['label'].strip()
                                age = find_age(age_to_grade, cur['label'], grade_name)
                                source_dict['age'] = age

                            if 'subject' in features:
                                source_dict['subject'] = subject['label'].strip()
                                if uncased: source_dict['subject'] =  source_dict['subject'].lower()

                            if 'topic' in features:
                                source_dict['topic'] = topic['label'].strip()
                                if uncased: source_dict['topic'] = source_dict['topic'].lower()

                            if 'doc_sum_nsents' in features or 'doc_sum_1sent' in features:
                                source_dict['doc_sums'] = []
                                if doc_sums[query_id]:
                                    sentences = []
                                    for doc in doc_sums[query_id]:
                                        sents = sent_tokenize(doc)
                                        if 'doc_sum_1sent' in features:
                                            sentence = sents[0]
                                            if uncased: sentence = sentence.lower()
                                            sentences.append(sentence)
                                        elif 'doc_sum_nsents' in features:
                                            tags_label = pos_tag(word_tokenize(source_dict['query']))
                                            nouns = [word for word, pos in tags_label if pos.startswith('NN')]
                                            for sent in sents:
                                                tokens = word_tokenize(sent)
                                                for token in tokens:
                                                    if token in nouns and sent not in sentences:
                                                        if uncased: sent = sent.lower()
                                                        sentences.append(sent)
                                    source_dict["doc_sums"] = sentences

                            source_features.append(source_dict)

    print(f'N of source queries: {len(source_features)}')
    # warnings.warn(f'N of source queries: {len(source_features)}')

    return source_features


def average_embeddings (model, source_features, queries):

    source_encodings = []

    for i in range(0,queries.shape[0]):

        query_encoding = queries[i]

        if 'doc_titles' in source_features[0].keys():
            query_doc_titles = source_features[i]['doc_titles']
            if query_doc_titles == []: query_doc_titles = ['']
            # average the encodings only if doc titles is not an empty list (if at least one result is pinned)
            # if model_name.endswith('cc.en.300.bin'):
            #     title_encodings = []
            #     for title in query_doc_titles:
            #         title_encodings.append(model.get_sentence_vector(title.replace('\n','')))
            # else:
            title_encodings = model.encode(query_doc_titles, convert_to_tensor=True)
            title_encodings = torch.mean(title_encodings, dim=0)
            query_encoding = torch.stack((query_encoding, title_encodings))

        if 'doc_sums' in source_features[0].keys():
            query_doc_sums = source_features[i]['doc_sums']
            if query_doc_sums == []: query_doc_sums = ['']
            # if model_name.endswith('cc.en.300.bin'):
            #     sum_encodings = []
            #     for sum in query_doc_sums:
            #         sum_encodings.append(model.get_sentence_vector(sum.replace('\n','')))
            # else:
            sum_encodings = model.encode(query_doc_sums, convert_to_tensor=True)
            sum_encodings = torch.mean(sum_encodings,dim=0)
            if query_encoding.size()[0] == 2:
                query_encoding = torch.stack((query_encoding[0],query_encoding[1],sum_encodings))
            else:
                query_encoding = torch.stack((query_encoding, sum_encodings))

        query_encoding = torch.mean(query_encoding, dim=0)

        source_encodings.append(query_encoding)

    return torch.squeeze(torch.stack([query_encoding for query_encoding in source_encodings]))


def get_encodings (source_features, target_features, model_filepath):

    start_time = time.perf_counter()

    if model_filepath.endswith('cc.en.300.bin'):

        source_queries = [source_dict.pop('query') for source_dict in source_features]
        target_queries = [target_dict.pop('query') for target_dict in target_features]
        # fasttext.util.download_model('en', if_exists='ignore')
        model = fasttext.load_model(model_filepath)
        source_encodings, target_encodings = [], []
        for label in source_queries: source_encodings.append(model.get_sentence_vector(label))
        for label in target_queries: target_encodings.append(model.get_sentence_vector(label))
        # if 'doc_titles' in source_features[0].keys() or 'doc_sums' in source_features[0].keys():
        #     source_encodings = average_embeddings(model, model_filepath, source_features, source_encodings)

    else:
        # queries
        source_queries = [source_dict.pop('query') for source_dict in source_features]
        target_queries = [target_dict.pop('query') for target_dict in target_features]
        model = SentenceTransformer(model_filepath)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        source_encodings = model.encode(source_queries,convert_to_tensor=True)
        target_encodings = model.encode(target_queries,convert_to_tensor=True)
        # doc info
        if 'doc_titles' in source_features[0].keys() or 'doc_sums' in source_features[0].keys():
            source_encodings = average_embeddings(model, source_features, source_encodings)
        # topic
        if 'topic' in source_features[0].keys() and 'topic' in target_features[0].keys():
            source_topics = [source_dict.pop('topic') for source_dict in source_features]
            target_topics = [target_dict.pop('topic') for target_dict in target_features]
            source_topic_encodings, target_topic_encodings = model.encode(source_topics, convert_to_tensor=True), model.encode(target_topics,convert_to_tensor=True)
            source_encodings = torch.concat((source_encodings,source_topic_encodings),dim=-1)
            target_encodings = torch.concat((target_encodings,target_topic_encodings), dim=-1)
        # subject
        if 'subject' in source_features[0].keys() and 'subject' in target_features[0].keys():
            source_subj = [source_dict.pop('subject') for source_dict in source_features]
            target_subj = [target_dict.pop('subject') for target_dict in target_features]
            source_subj_encodings, target_subj_encodings = model.encode(source_subj, convert_to_tensor=True), model.encode(target_subj,convert_to_tensor=True)
            source_encodings = torch.concat((source_encodings, source_subj_encodings), dim=-1)
            target_encodings = torch.concat((target_encodings, target_subj_encodings), dim=-1)
        # age
        if 'age' in source_features[0].keys() and 'age' in target_features[0].keys():
            vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
            dimensions = ['4 5 6 7 8 9 10 11 12 13 14 15 16 17 18']
            vec.fit(dimensions)
            source_age = vec.transform([str(source_dict.pop('age')) for source_dict in source_features])
            target_age = vec.transform([str(target_dict.pop('age')) for target_dict in target_features])
            # convert one-hot to dense vectors
            embedding_layer = nn.Embedding(len(dimensions[0].split(' ')), len(dimensions[0].split(' ')))
            source_age = torch.squeeze(torch.stack([torch.from_numpy(age.toarray()) for age in source_age]))
            target_age = torch.squeeze(torch.stack([torch.from_numpy(age.toarray()) for age in target_age]),dim=1)
            source_age_embeddings = embedding_layer(source_age)
            target_age_embeddings = embedding_layer(target_age)
            source_age_encodings = torch.mean(source_age_embeddings, -1)
            target_age_encodings = torch.mean(target_age_embeddings, -1)
            source_encodings = torch.concat((source_encodings,source_age_encodings),dim=-1)
            target_encodings = torch.concat((target_encodings,target_age_encodings),dim=-1)

        source_encodings = source_encodings.detach().cpu().numpy()
        target_encodings = target_encodings.detach().cpu().numpy()


    end_time = time.perf_counter()
    print(f"It took {end_time - start_time:0.4f} seconds to encode queries")

    return source_encodings, target_encodings


def get_info_for_pred(source,target):

    """
    Given target queries and source queries, generate lists of dictionaries, where each dict contains info on the target or source query
    :param source: dict with included curriculum trees, except target curriculum
    :param target: pandas dataframe with target queries
    :return: lists of dicts
    """

    target_info = []
    for i in range(0, len(target['TARGET'])):
        target_dict = dict()
        target_dict['label'] = target['TARGET'][i]
        target_dict['id'] = target['TARGET_ID'][i]
        target_dict['path'] = target['TARGET_PATH'][i]
        target_info.append(target_dict)

    source_info = []
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
                            source_dict['id'] = query_id
                            source_dict['path'] = f'{cur["label"]}>{grade["label"]}>{subject["label"]}>{unit["label"]}>{topic["label"]}'
                            source_info.append(source_dict)

    return source_info, target_info


def rank_cosine(source_vectors, source_info, target_vectors, target_info, r):

    predictions = defaultdict(dict)

    for t_vec,t_dict in zip(target_vectors, target_info):
        scores = []
        for s_vec, s_dict in zip(source_vectors,source_info):
            sim_score = 1 - distance.cosine(t_vec, s_vec)
            scores.append((s_dict['label'],s_dict['id'],s_dict['path'],sim_score))

        scores.sort(key=lambda x:x[-1], reverse=True)
        predictions[t_dict["id"]] = {'label': t_dict['label'],
                                     'path' : t_dict['path'],
                                     'scores': scores[:r]}

    return predictions


def input_pairing (target_features,source_features):

    source, target, age, subject= [], [], [], []

    for target_dict in target_features:
        for source_dict in source_features:
            target.append(f'{target_dict["query"]} {target_dict["topic"]}')
            source.append(f'{source_dict["query"]} {source_dict["topic"]} {" ".join(source_dict["doc_titles"])}')
            age.append((str(target_dict["age"]),str(source_dict["age"])))
            subject.append((target_dict["subject"],source_dict["subject"]))

    return {'target': target,
            'source': source,
            'age': age,
            'subject': subject}


def classifier_predict(model_filepath, loader):

    t0 = time.time()

    predictions = []
    model = Model(age=True,subject=True)
    model.load_state_dict(torch.load(model_filepath))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    for step, batch in enumerate(loader):

        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            # warnings.warn('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(loader), elapsed))
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(loader), elapsed))

        b_input_ids = batch["input_ids"].to(device)
        b_input_mask = batch["att_masks"].to(device)
        b_age = batch["age"].to(device)
        b_subj_ids = batch["sbj_input_ids"].to(device)
        b_subj_att = batch["sbj_att_masks"].to(device)

        with torch.no_grad():
            logits = model.forward(b_input_ids,
                                   b_input_mask,
                                   b_age,
                                   b_subj_ids,
                                   b_subj_att)

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    predictions = [pred for b in predictions for pred in b]

    return predictions


def rank_logits(predictions, source_info, target_info, r):

    rankings = dict()

    for t_pred, t_dict in zip(predictions, target_info):
        scores = []
        for pred, s_dict in zip(t_pred, source_info):
            scores.append((s_dict['label'], s_dict['id'], s_dict['path'], pred[1]))
        scores.sort(key=lambda x: x[-1], reverse=True)
        rankings[t_dict["id"]] = {'label': t_dict['label'],
                                   'path': t_dict['path'],
                                   'scores': scores}

    return rankings


def classify(target_features, source_features, base_model, model_filepath, target_info, source_info, r):

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    input_pairs = input_pairing(target_features, source_features)
    print(f"N of inferences: {len(input_pairs['source'])}")
    # warnings.warn(f"N of inferences: {len(input_pairs['source'])}")
    input_encoded = PairDataset(input_pairs,pre_processor,tokenizer)
    loader = DataLoader(input_encoded, batch_size=64, shuffle=False)
    preds = classifier_predict(model_filepath, loader)
    preds = [preds[x:x + len(source_features)] for x in range(0, len(preds), len(source_features))]
    rankings = rank_logits(preds, source_info, target_info, r)

    return rankings


def find_best_queries (source, target, model, features, r, mode = None, age_to_grade = None, doc_sums_dict = None, uncased=False, base_model = None):

    source_features = get_source_features(source, features, age_to_grade, uncased, doc_sums_dict)
    target_features = get_target_features(target, features, age_to_grade, uncased)

    if mode == 'classification':
        source_info, target_info = get_info_for_pred(source, target)
        predictions = classify(target_features,source_features,base_model,model,target_info,source_info,r)

    else:
        source_encodings, target_encodings = get_encodings(source_features, target_features, model)
        source_info, target_info = get_info_for_pred(source,target)
        predictions = rank_cosine(source_encodings,source_info, target_encodings, target_info, r)

    return predictions

# source_features = [{'query': 'Standard Deviation',
#                 'topic': 'Statistics',
#                 'subject': 'Mathematics',
#                 'age': '16',
#                 'doc_titles': ['How To Calculate The Standard Deviation',
#                                'Standard Deviation (formulas, examples, solutions, videos)']},
#                 {'query': 'Solar System',
#                  'topic': 'Universe',
#                  'subject': 'Physics',
#                  'age': '7',
#                  'doc_titles': ['']}]
#
# target_features = [{'query': 'Comparing standard deviation in datasets',
#                 'topic': 'Summarizing data to a single value',
#                 'subject': 'Statistics and Probability',
#                 'age': '16'}]
#
# source_info = [{'id': '3432',
#                 'label': 'Standard Deviation',
#                 'path': 'Class 11>Mathematics>15 Statistics>Statistics'},
#                {'id': '6768',
#                 'label': 'Solar System',
#                 'path': 'Class 7>Physics>The Universe>Universe>Solar System'}]
#
# target_info = [{'id': '5886',
#                 'label': 'Comparing standard deviation in datasets',
#                 'path': 'CCSS High School>Statistics and Probability>Interpreting Categorical and Quantitative Data>Summarizing data to a single value'}]
#
# source_encodings, target_encodings = get_encodings(source_features,target_features,'../models/cc.en.300.bin')
# rankings = rank_cosine(source_encodings,source_info,target_encodings,target_info,2)

# base_model = 'distilbert-base-uncased'
# model_filepath = '../models/distilbert_doctitle,topic,subject,age_13.pth'
#
# predictions = classify(target_features,source_features,base_model,model_filepath,target_info,source_info,100)
# print(predictions)