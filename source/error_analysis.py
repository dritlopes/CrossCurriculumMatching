import pandas as pd
from collections import defaultdict
import random
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from evaluation import recall_at_k
import matplotlib.pyplot as plt
import numpy as np
import csv
import rbo
from utils import grade_by_age,find_age
from data_exploration import tokenize_instances, check_n_gram_overlap
import json
import seaborn as sb
from collections import Counter

def read_in_sheet (filepath, work_sheet):

    df = pd.read_excel(filepath,work_sheet)
    return df


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

def read_in_rq2 (random_seed):

    ft_sbert = pd.read_csv(
        f'../results/test_paraphrase-sbert-label-rankingloss-nodup_{random_seed}_top5__filterAgeFalse.csv',
        sep='\t', dtype={'TARGET_ID': str, 'SOURCE_ID': str})
    ft_sbert['MODEL'] = '-title'

    ft_sbert_doc = pd.read_csv(
        f'../results/test_paraphrase-sbert-label-title-rankingloss-nodup_{random_seed}_top5_doc_title_filterAgeFalse.csv',
        sep='\t', dtype={'TARGET_ID': str, 'SOURCE_ID': str})
    ft_sbert_doc['MODEL'] = '+title'

    all = pd.concat((ft_sbert,ft_sbert_doc), axis=0, ignore_index=True)
    gold = pd.read_csv(f'../data/test_query_pairs_{random_seed}.csv', sep='\t',
                       dtype={'TARGET_ID': str,
                              'TARGET_GRADEID': str,
                              'SOURCE_ID': str,
                              'SOURCE_GRADEID': str})
    return all, gold

def generate_rankings_rq2 (n_targets, all):

    combinations = defaultdict(list)

    for target, group in all.groupby(['TARGET_ID']):

        pred = {'-title_pos': [],
                '-title_neg': [],
                '+title_pos': [],
                '+title_neg': []}

        for model, ranking in group.groupby(['MODEL']):
            if 1 in ranking['GOLD'].tolist():
                pred[f'{model}_pos'].append(ranking['TARGET_ID'].tolist()[0])
            if 1 not in ranking['GOLD'].tolist():
                pred[f'{model}_neg'].append(ranking['TARGET_ID'].tolist()[0])

        for i in pred['+title_pos']:
            if i in pred['-title_neg']:
                combinations['1 0'].append(i)
            if i in pred['+title_pos']:
                combinations['1 1'].append(i)
        for i in pred['+title_neg']:
            if i in pred['-title_pos']:
                combinations['0 1'].append(i)
            if i in pred['-title_neg']:
                combinations['0 0'].append(i)

    subset = pd.DataFrame()
    target_col, target_path_col, target_id_col, combi = [], [], [], []

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
                        model_ranking = ranking[['SOURCE', 'SOURCE_ID', 'SOURCE_PATH', 'GOLD']].copy().reset_index()
                        model_ranking = model_ranking.rename(columns={'SOURCE': f'{model.upper()}',
                                                                      'SOURCE_ID': f'{model.upper()}_ID',
                                                                      'SOURCE_PATH': f'{model.upper()}_PATH',
                                                                      'GOLD': f'{model.upper()}_GOLD'})

                        model_ranking.drop(columns='index')
                        subset_rankings.append(model_ranking)

                    for r in subset_rankings:
                        subset_col = pd.concat((subset_col, r), axis=1)

                    subset = pd.concat((subset, subset_col), axis=0)

    assert len(target_col) == len(subset), f'Length of target: {len(target_col)}, length of subset {len(subset)}'

    subset['TARGET'] = target_col
    subset['TARGET_ID'] = target_id_col
    subset['TARGET_PATH'] = target_path_col
    subset['COMBI'] = combi
    subset = subset[['TARGET', 'COMBI', 'TARGET_ID', 'TARGET_PATH',
                     'DOC_ID', 'DOC', 'DOC_GOLD', 'DOC_PATH',
                     'NO_DOC_ID', 'NO_DOC', 'NO_DOC_GOLD', 'NO_DOC_PATH']]
    subset.to_csv('../eval/results_subset_rankings_rq2.csv', sep='\t', index=False)


def generate_rankings_rq1(n_targets, all):

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
    target_col, target_path_col, target_id_col, combi = [], [], [], []

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
                        model_ranking = ranking[['SOURCE', 'SOURCE_ID', 'SOURCE_PATH', 'GOLD']].copy().reset_index()
                        model_ranking = model_ranking.rename(columns={'SOURCE': f'{model.upper()}',
                                                                      'SOURCE_ID': f'{model.upper()}_ID',
                                                                      'SOURCE_PATH': f'{model.upper()}_PATH',
                                                                      'GOLD': f'{model.upper()}_GOLD'})

                        model_ranking.drop(columns='index')
                        subset_rankings.append(model_ranking)

                    for r in subset_rankings:
                        subset_col = pd.concat((subset_col, r), axis=1)

                    subset = pd.concat((subset, subset_col), axis=0)

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


def rbo_rankings(results, n_examples, models):

    rankings = defaultdict()

    for model, model_group in results.groupby(['MODEL']):

        if model in models:
            target_dict = defaultdict()
            for target, r in model_group.groupby(['TARGET_ID']):
                target_dict[target] = {'ranking': list(r.SOURCE_ID),
                                            'ranking_text': list(r.SOURCE),
                                            'target': target,
                                            'target_text': list(r.TARGET)[0],
                                            'gold': list(r.GOLD)}
                rankings[model] = target_dict

    # assert len(rankings[models[0]]['rankings']) == len(rankings[models[1]]['rankings']), f'Not the same number of rankings between models. {len(rankings[models[0]]["rankings"])} for model {models[0]} and {len(rankings[models[1]]["rankings"])} for model {models[1]}'

    rbo_scores = []

    for t in rankings[models[0]].keys():
        s = rbo.RankingSimilarity(rankings[models[0]][t]['ranking'],rankings[models[1]][t]['ranking']).rbo()
        rbo_scores.append((t,rankings[models[0]][t]['target_text'],rankings[models[0]][t]['ranking'],
                           rankings[models[0]][t]['ranking_text'],rankings[models[0]][t]['gold'],
                           rankings[models[1]][t]['ranking'],rankings[models[1]][t]['ranking_text'],
                           rankings[models[1]][t]['gold'],s))
    rbo_scores.sort(key=lambda x: x[-1], reverse=True)

    rbo_scores_topk = rbo_scores[:n_examples]
    rbo_scores_lask = rbo_scores[len(rbo_scores) - n_examples:]
    rbo_scores = rbo_scores_topk + rbo_scores_lask

    df = pd.DataFrame(rbo_scores,
                      columns=['TARGET_ID','TARGET',f'{models[0]}_SOURCE_ID',f'{models[0]}_SOURCE', f'{models[0]}_GOLD',
                                f'{models[1]}_SOURCE_ID',f'{models[1]}_SOURCE',f'{models[1]}_GOLD','RBO_SCORE'])

    df.to_csv(f'../eval/rbo_{models[0]},{models[1]}.csv', sep='\t', index=False)


def categorize_rankings (all, gold, data, models):

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

    print(f'{len(missing)} missing target ids: {missing}')

    df = pd.DataFrame(rows)
    df.to_csv(f'../results/results_per_target_{models}.csv', sep='\t', index=False)

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


def barplot_scores_bins (mean_recall_dict, models=None,features=None):

    if not features:
        features = mean_recall_dict.keys()

    for feature in features:

        plt.figure(figsize=(25,20))
        plt.rcParams.update({'font.size': 25})

        if not models:
            models = []
            if 'ft_sbert' in list(mean_recall_dict[feature].keys()):
                models.append('ft_sbert')
            if 'sbert' in list(mean_recall_dict[feature].keys()):
                models.append('sbert')
            if 'fasttext' in list(mean_recall_dict[feature].keys()):
                models.append('fasttext')
            if 'tfidf' in list(mean_recall_dict[feature].keys()):
                models.append('tfidf')
            if '-title' in list(mean_recall_dict[feature].keys()):
                models.append('-title')
            if '+title' in list(mean_recall_dict[feature].keys()):
                models.append('+title')

        bins = mean_recall_dict[feature][models[0]][0]
        n_bins = len(bins)
        bar_positions = np.arange(n_bins)
        width = 0.20

        for model in models:

            values = mean_recall_dict[feature][model][1]
            sds = mean_recall_dict[feature][model][2]
            if model == '+title':
                plt.barh(bar_positions, values, width, label=model, xerr=sds, color='navy')
            elif model == '-title':
                plt.barh(bar_positions, values, width, label=model, xerr=sds, color='tab:blue')
            else:
                plt.barh(bar_positions, values, width, label=model, xerr=sds)
            bar_positions = bar_positions + width

        plt.xlabel('recall@5')
        plt.xlim(0, 1)
        plt.yticks(np.arange(n_bins) + width, bins)
        plt.legend()
        plt.savefig(f'../eval/fig_{feature}_{models}.png')


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


def sample (predictions, age_to_grade, models='', proportion=0.05):

    random_seed = 42
    random.seed(random_seed)
    df_matches = []
    df_mismatches = []

    for model, model_pred in predictions.groupby(['MODEL']):

        for cur, cur_group in model_pred.groupby(['TARGET_CURRICULUM']):

            all_matches, all_mismatches = [], []

            for target, ranking in cur_group.groupby(['TARGET_ID']):

                ranking = ranking.to_dict(orient='list')

                add_match, add_mismatch = True, True
                for i, source in enumerate(ranking['SOURCE_ID']):
                    if ranking['GOLD'][i] == 1 and add_match:
                        all_matches.append((target,ranking['TARGET'][i],
                                         ranking['TARGET_PATH'][i],source,
                                         ranking['SOURCE'][i],ranking['SOURCE_PATH'][i]))
                        add_match = False
                    elif ranking['GOLD'][i] == 0 and add_mismatch:
                        all_mismatches.append((target, ranking['TARGET'][i],
                                         ranking['TARGET_PATH'][i], source,
                                         ranking['SOURCE'][i], ranking['SOURCE_PATH'][i]))
                        add_mismatch = False

            sample_matches, sample_mismatches = [],[]

            n_examples = int(proportion * len(all_matches))
            if n_examples == 0: n_examples = 1
            while len(sample_matches) < n_examples:
                sample_matches.append(random.choice(all_matches))

            n_examples = int(proportion * len(all_mismatches))
            if n_examples == 0: n_examples = 1
            while len(sample_mismatches) < n_examples:
                sample_mismatches.append(random.choice(all_mismatches))

            for pair in sample_mismatches:

                target_path = pair[2].split('>')
                source_path = pair[-1].split('>')

                # define subject
                sbj_map = [{'Mathematics','Algebra','Algebra Concepts'},
                           {'Mathematics','Geometry Concepts', 'Geometry'},
                           {'Mathematics','Statistics and Probability', 'Statistics Concepts'},
                           {'Mathematics', 'Numbers and Quantities', 'Numeracy and Mathematics', 'Numbers Concepts'},
                           {'Mathematics', 'Functions'},
                           {'Mathematics', 'Measurement Concepts'},
                           {'Sciences','Science','Biology','Environmental Sciences'},
                           {'Sciences','Science','Chemistry'},
                           {'Sciences','Science','Physics'},
                           {'Computing','Computer Science'},
                           {'Physical Science','Physics', 'Physical Sciences'},
                           {'Physical Science', 'Physical Sciences', 'Chemistry'},
                           {'Earth and Space Sciences', 'Physics'},
                           {'Earth and Space Sciences', 'Biology'},
                           {'Life Sciences', 'Biology'}]
                target_subject = target_path[2]
                source_subject = source_path[2]
                subject = 0
                if target_subject.strip() == source_subject.strip():
                    subject = 1
                else:
                    for subj_set in sbj_map:
                        if target_subject in subj_set and source_subject in subj_set:
                            subject = 1
                            break
                # define age
                target_age = find_age(age_to_grade, target_path[0].strip(), target_path[1].strip())
                source_age = find_age(age_to_grade, source_path[0].strip(), source_path[1].strip())
                age = 0
                if int(target_age) in [int(source_age), int(source_age) + 1, int(source_age) - 1]:
                    age = 1

                # define terminological type of mismatch
                # target_text = pair[1].strip()
                # source_text = pair[4].strip()
                # overlap = 'no'
                # if target_text.lower() == source_text.lower():
                #     overlap = 'full'
                # elif bool(set(target_text.split(' ')) & set(source_text.split(' '))):
                #     overlap = 'partial'

                df_mismatches.append({'TARGET': pair[1],
                                      'TARGET_PATH': pair[2],
                                      'SOURCE': pair[4],
                                      'SOURCE_PATH': pair[-1],
                                      'GOLD': 0,
                                      'MODEL': model,
                                      'SUBJECT': subject,
                                      'AGE': age})

            for pair in sample_matches:

                df_matches.append({'TARGET': pair[1],
                                    'TARGET_PATH': pair[2],
                                    'SOURCE': pair[4],
                                    'SOURCE_PATH': pair[-1],
                                    'GOLD': 1,
                                    'MODEL': model})

    df_mismatches = pd.DataFrame(df_mismatches)
    df_matches = pd.DataFrame(df_matches)
    df_mismatches.to_csv(f'../eval/incorrect_matches_{models}.csv',sep='\t',index=False)
    df_matches.to_csv(f'../eval/correct_matches_{models}.csv',sep='\t',index=False)

def count_error_types (df):

    all_models = dict()
    for model, group in df.groupby(['MODEL']):

        subj = list(group['SUBJECT'])
        age = list(group['AGE'])
        topic = list(group['UNIT/TOPIC'])
        topic = [cell.strip() for cell in topic]
        query = list(group['QUERY'])

        all = [combi for combi in zip(subj,age,topic,query)]
        counts = Counter(all)
        # print(counts)

        subj_age = [combi for combi in zip(subj,age)]
        subj_age_counts = Counter(subj_age)

        topic_query = [combi for combi in zip(topic,query)]
        topic_query_counts = Counter(topic_query)

        all_models[model] = [counts, subj_age_counts, topic_query_counts]

    return all_models

def error_types_barplot (all_models, output_filepath, layers, freq = 0.1):

    # model in x axis, percentage in y axis, error combinations in colors
    if layers == 'TOPIC,QUERY': index = 2
    else: index = 1

    if index == 2:
        freq_dict = defaultdict(dict)
        for model, list_ in all_models.items():
            for combi, count in list_[index].items():
                frequency = round(count/sum(list(list_[index].values())),2)
                freq_dict[model][combi] = frequency
        add = []
        for model, dict_ in freq_dict.items():
            for combi, frequency in dict_.items():
                if frequency > freq:
                    add.append(combi)

        df_dict = defaultdict(list)
        for combi in set(add):
            for model, dict_ in freq_dict.items():
                key = f'{combi[0]}/{combi[1]}'
                if combi[0] == combi[1]: key = combi[0]
                frequency = 0
                if combi in dict_.keys(): frequency = dict_[combi]
                df_dict[key].append(frequency)
        df_dict['model'] = list(freq_dict.keys())

    # df_dict = defaultdict(list)
    #
    # if layers == 'TOPIC,QUERY': index = 2
    # else: index = 1
    #
    # for model, list_ in all_models.items():
    #     df_dict['model'].append(model)
    #     for combi, count in list_[index].items():
    #         frequency = round(count/sum(list(list_[index].values())),2)
    #         df_dict[f'{combi[0]}-{combi[1]}'].append(frequency)
    #

    df = pd.DataFrame(df_dict)
    plt.rcParams.update({'font.size': 13})
    df.set_index('model').plot(kind='bar', stacked=True)
    if index == 2: plt.gcf().set_size_inches(12, 10)
    plt.xticks(rotation=45)
    if index == 1: plt.legend(labels=['sbj+age+','sbj+age-','sbj-age-','sbj-age+'])
    plt.savefig(output_filepath)
    plt.show()

##############

# random_seed = 42
# curriculums = 'ICSE,CBSE,Cambridge,English,CCSS,NGSS,Scotland'
# n_examples = 5
# data_info = pd.read_csv(f'../data/data.csv',sep='\t', dtype= {'query_id': str,'age': str})
# all, gold = read_in_rq1(random_seed)
# generate_pairs_rq1(n_examples, all)
# generate_rankings_rq1(n_examples, all)
# df = categorize_rankings(all,gold,data_info,models='tfidf,fasttext,sbert,ft_sbert')
# mean_recall_dict = analyse_rankings(df,layers=['SUBJECT','AGE','CURRICULUM'])
# barplot_scores_bins(mean_recall_dict)
# rbo_rankings(all,n_examples=20,models=('tfidf','ft_sbert'))
# rbo_rankings(all,n_examples=20,models=('sbert','ft_sbert'))
# age_to_grade = grade_by_age('ICSE,CBSE,Cambridge,English,CCSS,NGSS,Scotland',
#                             age_filepath=f'../data/MASTER Reading levels and age filter settings (pwd 123456).xlsx')
# sample(all, age_to_grade, models='tf_idf,fasttext,sbert,ft-sbert', proportion=0.10)

# random_seed = 42
# n_examples = 5
# curriculums = 'ICSE,CBSE,Cambridge,English,CCSS,NGSS,Scotland'
# data_info = pd.read_csv(f'../data/data.csv',sep='\t', dtype= {'query_id': str,'age': str})
# all, gold = read_in_rq2(random_seed)
# generate_rankings_rq2(n_examples, all)
# df = categorize_rankings(all,gold,data_info,models='+title,-title')
# mean_recall_dict = analyse_rankings(df,layers=['SUBJECT','AGE','CURRICULUM'])
# barplot_scores_bins(mean_recall_dict)
# rbo_rankings(all,n_examples=20,models=('doc','no_doc'))
# age_to_grade = grade_by_age(curriculums,
#                             age_filepath=f'../data/MASTER Reading levels and age filter settings (pwd 123456).xlsx')
# sample(all, age_to_grade, models='doc,no_doc', proportion=0.10)

# check if models give reasonable vectors similarity for subject
# sentences = ['Mathematics', 'Physics', 'History']
# model = SentenceTransformer("../models/paraphrase-sbert-label-rankingloss-nodup_42")
# vectors = model.encode(sentences)
# for pair in [(0,1),(0,2)]:
#     sim_score = 1 - distance.cosine(vectors[pair[0]],vectors[pair[1]])
#     print(f'Sentence {pair[0]} and Sentence {pair[1]}: {sim_score}')

# N-gram overlap
# outputfile= f"../data/1_gram_overlap_Cambridge', 'CBSE', 'CCSS', 'English', 'ICSE', 'NGSS', 'Scotland.json"
# with open(f'../data/data_dict.json') as json_file:
#     data = json.load(json_file)
# tokenized = tokenize_instances(data,curriculums)
# check_n_gram_overlap(tokenized, outputfile)
# with open(outputfile) as json_file:
#     lexical_overlap = json.load(json_file)
# combi = list(lexical_overlap.keys())
# curs = [key.split('-') for key in combi]
# curs_reverse = [[key[1],key[0]] for key in curs]
# curs = curs + curs_reverse
# curA = [key[0] for key in curs]
# curB = [key[1] for key in curs]
# ngram_overlap = [dict['query-query'] for dict in lexical_overlap.values()]
# ngram_overlap = ngram_overlap + ngram_overlap
# df = pd.DataFrame({'curA': curA, 'curB': curB, 'overlap': ngram_overlap})
# df = df.pivot(index='curA',columns='curB',values='overlap')
# mask = np.triu(np.ones_like(df, dtype=bool))
# fig, ax = plt.subplots(figsize=(11, 9))
# hm = sb.heatmap(df, mask=mask, cmap='rocket_r')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# hm.set(xlabel=None, ylabel=None)
# plt.savefig(f'../data/1_gram_overlap.png')
# plt.show()

# with open(f'../data/data_dict.json') as json_file:
#     data = json.load(json_file)
# for subject in ['Science','Mathematics','English']:
#     tokenized = tokenize_instances(data,curriculums,subjects=subject)
#     check_n_gram_overlap(tokenized, f'../data/1_gram_overlap_{subject}.json')
# for age in ['11','12','13','14','15','16','17']:
#     tokenized = tokenize_instances(data, curriculums, ages=age)
#     check_n_gram_overlap(tokenized, f'../data/1_gram_overlap_{age}.json')
# mean_ages = dict()
# for age in ['11','12','13','14','15','16','17']:
#     with open(f'../data/1_gram_overlap_{age}.json') as json_file:
#         age_data = json.load(json_file)
#     scores = [dict['query-query'] for dict in age_data.values()]
#     mean_ages[age] = sum(scores)/len(scores)
# for age in ['11','12','13','14','15','16','17']:
#     tokenized = tokenize_instances(data, curriculums='ICSE,CBSE', subjects='Mathematics', ages=age)
#     check_n_gram_overlap(tokenized, f'../data/1_gram_overlap_ICSE,CBSE_Math_{age}.json')

annotations = '../eval/analysis_hits_errors_Somya.xlsx'
work_sheet = 'incorrect_matches_rq1_1%'
df = read_in_sheet(annotations, work_sheet)
all_counts = count_error_types(df)
error_types_barplot(all_counts, f'../eval/fig_error_types_rq1_topic_query.png',layers='TOPIC,QUERY')
annotations = '../eval/analysis_hits_errors_Somya.xlsx'
work_sheet = 'incorrect_matches_rq2_1%'
df = read_in_sheet(annotations, work_sheet)
all_counts = count_error_types(df)
error_types_barplot(all_counts, f'../eval/fig_error_types_rq2_topic_query.png',layers='TOPIC,QUERY')