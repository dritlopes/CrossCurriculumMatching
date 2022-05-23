import pandas as pd
from collections import defaultdict
import random
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from evaluation import recall_at_k
import matplotlib.pyplot as plt
import numpy as np
import csv

def read_in_rq1 (random_seed):

    tf_idf = pd.read_csv(f'../results/test_tf-idf_{random_seed}_top5__filterAgeFalse.csv', sep='\t',
                         dtype={'TARGET_ID': str, 'SOURCE_ID': str})
    tf_idf['MODEL'] = 'tfidf'

    fasttext = pd.read_csv(f'../results/test_cc.en.300.bin_{random_seed}_top5__filterAgeFalse.csv', sep='\t',
                           dtype={'TARGET_ID': str, 'SOURCE_ID': str})
    fasttext['MODEL'] = 'fasttext'

    sbert = pd.read_csv(f'../results/test_paraphrase-MiniLM-L6-v2_{random_seed}_top5__filterAgeFalse.csv', sep='\t',
                        dtype={'TARGET_ID': str, 'SOURCE_ID': str})
    sbert['MODEL'] = 'sbert'

    ft_sbert = pd.read_csv(
        f'../results/test_paraphrase-sbert-label-rankingloss-nodup_{random_seed}_top5__filterAgeFalse.csv',
        sep='\t', dtype={'TARGET_ID': str, 'SOURCE_ID': str})
    ft_sbert['MODEL'] = 'ft_sbert'

    all = pd.concat((tf_idf, fasttext, sbert, ft_sbert), axis=0, ignore_index=True)
    gold = pd.read_csv(f'../data/test_query_pairs_{random_seed}.csv', sep='\t',
                                                    dtype={'TARGET_ID': str,
                                                           'TARGET_GRADEID': str,
                                                           'SOURCE_ID': str,
                                                           'SOURCE_GRADEID': str})

    return all, gold

def categorize_rankings (all, gold, data):

    rows = []
    gold_queries = dict()
    map = defaultdict(dict)

    # age mapping
    for target, age, topic, subject, cur in zip(data.query_id, data.age, data.topic, data.subject, data.curriculum):
        map[target] = {'age': age,
                       'topic': topic,
                       'subject': subject,
                       'curriculum': cur}

    # gold queries
    for target, group in gold.groupby(['TARGET_ID']):
        gold_queries[target] = group.SOURCE_ID

    missing = set()
    for target, group in all.groupby(['TARGET_ID','MODEL']):

        # path = group['TARGET_PATH'].tolist()[0]
        # layers = path.split('>')
        # assert len(layers) == 5, f'{path} is not a complete path'

        ranking = group.GOLD
        fn = len(gold_queries[target[0]]) - sum(ranking)
        recall = recall_at_k(ranking,fn,5)

        if target[0] in map.keys():
            row = {'TARGET_ID': target[0],
                   'TOPIC': map[target[0]]['topic'],
                   'SUBJECT': map[target[0]]['subject'],
                   'AGE': map[target[0]]['age'],
                   'CURRICULUM': map[target[0]]['curriculum'],
                   'MODEL': target[1],
                   'RECALL': recall}

            rows.append(row)
        else:
            missing.add(target[0])

    # print(f'{len(missing)} missing target ids: {missing}')

    df = pd.DataFrame(rows)
    df.to_csv(f'../results/results_per_target_rq1.csv', sep='\t', index=False)

    return df

def analyse_rankings (df,layers):

    mean_recall_dict = defaultdict()

    for layer in layers:
        bin_dict = defaultdict(tuple)
        for model, model_group in df.groupby(['MODEL']):
            mean_recall, bins, sds = [], [], []
            if layer == 'AGE':
                model_group[layer] = pd.to_numeric(model_group[layer],downcast='integer')
                model_group[layer] = model_group[layer].sort_values()
            for bin, bin_group in model_group.groupby([layer]):
                if len(bin_group) > 15:
                    if bin != '-1':
                        recall_scores = bin_group.RECALL
                        mean = np.mean(recall_scores)
                        sd = np.std(recall_scores, ddof=1) / np.sqrt(np.size(recall_scores))
                        mean_recall.append(mean)
                        bins.append(bin)
                        sds.append(sd)
                        bin_dict[model] = (bins, mean_recall, sds)
        mean_recall_dict[layer] = bin_dict

    return mean_recall_dict


def barplot_scores_bins (mean_recall_dict, models=None,features=None, rq='1'):

    if not features:
        features = mean_recall_dict.keys()

    for feature in features:

        plt.figure(num=feature, clear=True, figsize=(20,20))

        if not models:
            models = list(mean_recall_dict[feature].keys())

        for i, model in enumerate(models):

            names = mean_recall_dict[feature][model][0]
            values = mean_recall_dict[feature][model][1]
            sds = mean_recall_dict[feature][model][2]

            plt.subplot(2,2,i+1)
            plt.bar(names, values, yerr=sds)
            plt.xlabel(model)
            plt.xticks()
            if feature != 'AGE': plt.xticks(rotation=90)
            plt.ylabel('recall@5')
            plt.yticks()
            plt.ylim(0, 1)
            #params = {'xtick.labelsize': 25, 'ytick.labelsize': 25}
            #plt.rcParams.update(params)

        plt.title(f'Mean recall at top 5 per {feature.lower()} per encoder on learning objective')
        plt.savefig(f'../eval/fig_{feature}_rq{rq}.png')

def heatmap_compare_models (mean_recall_dict):

    pass


def generate_rankings_rq1 (n_targets, all):

    combinations = defaultdict(list)

    for target, group in all.groupby(['TARGET_ID']):

        pred = {'ft_sbert_pos': [],
                'ft_sbert_neg': [],
                'sbert_pos': [],
                'sbert_neg': [],
                'fasttext_pos': [],
                'fasttext_neg': [],
                'tfidf_pos': [],
                'tfidf_neg': []}

        for model, ranking in group.groupby(['MODEL']):
            if 1 in ranking['GOLD'].tolist():
                pred[f'{model}_pos'].append(ranking['TARGET_ID'].tolist()[0])
            if 1 not in ranking['GOLD'].tolist():
               pred[f'{model}_neg'].append(ranking['TARGET_ID'].tolist()[0])

        for i in pred['ft_sbert_pos']:
            if i in pred['sbert_neg'] and i in pred['fasttext_neg'] and i in pred['tfidf_neg']:
                combinations['1 0 0 0'].append(i)
            if i in pred['sbert_pos'] and i in pred['fasttext_neg'] and i in pred['tfidf_neg']:
                combinations['1 1 0 0'].append(i)
        for i in pred['ft_sbert_neg']:
            if i in pred['sbert_neg'] and i in pred['fasttext_pos'] and i in pred['tfidf_pos']:
                combinations['0 0 1 1'].append(i)
            if i in pred['sbert_neg'] and i in pred['fasttext_neg'] and i in pred['tfidf_neg']:
                combinations['0 0 0 0'].append(i)

    subset = pd.DataFrame()
    target_col, target_path_col, target_id_col, combi = [],[],[], []

    for key, ids in combinations.items():

        subset_ids = []
        if len(ids) < n_targets:
            subset_ids = ids
        else:
            for n in range(n_targets):
                subset_ids.append(random.choice(ids))

        for id in subset_ids:
            for target, group in all.groupby(['TARGET_ID']):

                if id == target:

                    subset_col = pd.DataFrame()
                    subset_rankings = []

                    for n in group['TARGET'].tolist()[:5]:
                        target_col.append(n)
                    for n in group['TARGET_ID'].tolist()[:5]:
                        target_id_col.append(n)
                    for n in group['TARGET_PATH'].tolist()[:5]:
                        target_path_col.append(n)
                    for n in range(5):
                        combi.append(key)

                    for model, ranking in group.groupby(['MODEL']):
                        model_ranking = ranking[['SOURCE','SOURCE_ID','SOURCE_PATH','GOLD']].copy().reset_index()
                        model_ranking = model_ranking.rename(columns={'SOURCE': f'{model.upper()}',
                                                                        'SOURCE_ID': f'{model.upper()}_ID',
                                                                        'SOURCE_PATH': f'{model.upper()}_PATH',
                                                                        'GOLD': f'{model.upper()}_GOLD'})

                        model_ranking.drop(columns='index')
                        subset_rankings.append(model_ranking)

                    for r in subset_rankings:
                        subset_col = pd.concat((subset_col,r),axis=1)

                    subset = pd.concat((subset,subset_col),axis=0)

    assert len(target_col) == len(subset), f'Length of target: {len(target_col)}, length of subset {len(subset)}'

    subset['TARGET'] = target_col
    subset['TARGET_ID'] = target_id_col
    subset['TARGET_PATH'] = target_path_col
    subset['COMBI'] = combi
    subset = subset[['TARGET', 'COMBI', 'TARGET_ID', 'TARGET_PATH',
                     'FT_SBERT_ID', 'FT_SBERT', 'FT_SBERT_GOLD', 'FT_SBERT_PATH',
                     'SBERT_ID', 'SBERT', 'SBERT_GOLD', 'SBERT_PATH',
                     'FASTTEXT_ID', 'FASTTEXT', 'FASTTEXT_GOLD', 'FASTTEXT_PATH',
                     'TFIDF_ID', 'TFIDF', 'TFIDF_GOLD', 'TFIDF_PATH']]
    subset.to_csv('../eval/results_subset_rankings_rq1.csv', sep='\t', index=False)


def generate_pairs_rq1 (n_targets, all):

    all['INDEX'] = range(len(all))
    combinations = defaultdict(list)

    for target, group in all.groupby(['TARGET_ID']):
        pred = {'ft_sbert_pos': [],
                'ft_sbert_neg': [],
                'sbert_pos': [],
                'sbert_neg': [],
                'fasttext_pos': [],
                'fasttext_neg': [],
                'tfidf_pos': [],
                'tfidf_neg': []}
        group = group.to_dict(orient='list')
        for i,model in enumerate(group['MODEL']):
            if group['GOLD'][i] == 1:
                pred[f'{model}_pos'].append(group['INDEX'][i])
            if group['GOLD'][i] == 0:
               pred[f'{model}_neg'].append(group['INDEX'][i])

        for i in range(len(pred['ft_sbert_pos'])):
            if i < len(pred['sbert_neg']) and i < len(pred['fasttext_neg']) and i < len(pred['tfidf_neg']):
                combinations['1 0 0 0'].append((pred['ft_sbert_pos'][i],pred['sbert_neg'][i],pred['fasttext_neg'][i],pred['tfidf_neg'][i]))
            if i < len(pred['sbert_pos']) and i < len(pred['fasttext_neg']) and i < len(pred['tfidf_neg']):
                combinations['1 1 0 0'].append((pred['ft_sbert_pos'][i], pred['sbert_pos'][i], pred['fasttext_neg'][i], pred['tfidf_neg'][i]))
        for i in range(len(pred['ft_sbert_neg'])):
            if i < len(pred['sbert_neg']) and i < len(pred['fasttext_pos']) and i < len(pred['tfidf_pos']):
                combinations['0 0 1 1'].append((pred['ft_sbert_neg'][i],pred['sbert_neg'][i],pred['fasttext_pos'][i],pred['tfidf_pos'][i]))
            if i < len(pred['sbert_neg']) and i < len(pred['fasttext_neg']) and i < len(pred['tfidf_neg']):
                combinations['0 0 0 0'].append((pred['ft_sbert_neg'][i],pred['sbert_neg'][i],pred['fasttext_neg'][i],pred['tfidf_neg'][i]))

    subset = []
    for key, indeces in combinations.items():
        if len(indeces) < n_targets:
            for rows in indeces:
                subset.append({'TARGET': all['TARGET'][rows[0]],
                               'COMBI': key,
                               'TARGET_ID': all['TARGET_ID'][rows[0]],
                               'TARGET_PATH': all['TARGET_PATH'][rows[0]],
                               'FTSBERT_ID': all['SOURCE_ID'][rows[0]],
                               'FTSBERT': all['SOURCE'][rows[0]],
                               'FTSBERT_PATH': all['SOURCE_PATH'][rows[0]],
                               'SBERT_ID': all['SOURCE_ID'][rows[1]],
                               'SBERT': all['SOURCE'][rows[1]],
                               'SBERT_PATH': all['SOURCE_PATH'][rows[1]],
                               'FASTTEXT_ID': all['SOURCE_ID'][rows[2]],
                               'FASTTEXT': all['SOURCE'][rows[2]],
                               'FASTTEXT_PATH': all['SOURCE_PATH'][rows[2]],
                               'TFIDF_ID': all['SOURCE_ID'][rows[3]],
                               'TFIDF': all['SOURCE'][rows[3]],
                               'TFIDF_PATH': all['SOURCE_PATH'][rows[3]],
                               })
        else:
            for n in range(n_targets):
                rows = random.choice(indeces)
                subset.append({'TARGET': all['TARGET'][rows[0]],
                               'COMBI': key,
                               'TARGET_ID': all['TARGET_ID'][rows[0]],
                               'TARGET_PATH': all['TARGET_PATH'][rows[0]],
                               'FTSBERT': all['SOURCE'][rows[0]],
                               'FTSBERT_ID': all['SOURCE_ID'][rows[0]],
                               'FTSBERT_PATH': all['SOURCE_PATH'][rows[0]],
                               'SBERT': all['SOURCE'][rows[1]],
                               'SBERT_ID': all['SOURCE_ID'][rows[1]],
                               'SBERT_PATH': all['SOURCE_PATH'][rows[1]],
                               'FASTTEXT': all['SOURCE'][rows[2]],
                               'FASTTEXT_ID': all['SOURCE_ID'][rows[2]],
                               'FASTTEXT_PATH': all['SOURCE_PATH'][rows[2]],
                               'TFIDF': all['SOURCE'][rows[3]],
                               'TFIDF_ID': all['SOURCE_ID'][rows[3]],
                               'TFIDF_PATH': all['SOURCE_PATH'][rows[3]],
                               })
    subset_df = pd.DataFrame(subset)
    subset_df.to_csv('../eval/results_subset_pairs_rq1.csv', sep='\t', index=False)

###############

random_seed = 42
n_examples = 5
data_info = pd.read_csv(f'../data/data.csv',sep='\t', dtype= {'query_id': str,'age': str})
all, gold = read_in_rq1(random_seed)
# generate_pairs_rq1(n_examples, all)
generate_rankings_rq1(n_examples, all)
df = categorize_rankings(all,gold,data_info)
mean_recall_dict = analyse_rankings(df,layers=['SUBJECT','AGE','CURRICULUM'])
barplot_scores_bins(mean_recall_dict)

# check if models give reasonable vectors similarity for subject
# sentences = ['Mathematics', 'Physics', 'History']
# model = SentenceTransformer("../models/paraphrase-sbert-label-rankingloss-nodup_42")
# vectors = model.encode(sentences)
# for pair in [(0,1),(0,2)]:
#     sim_score = 1 - distance.cosine(vectors[pair[0]],vectors[pair[1]])
#     print(f'Sentence {pair[0]} and Sentence {pair[1]}: {sim_score}')