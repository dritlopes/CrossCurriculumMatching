import numpy as np
import json
import argparse
import pandas as pd
from utils import find_query_copies, read_in_data

def recall_at_k(r, fn, k):

    assert k >= 1
    topk = np.asarray(r)[:k] != 0
    # rest = np.asarray(r)[k:]
    tp = np.sum(topk)
    # fn = np.sum(rest)
    if fn + tp > 5: fn = 5 - tp # some queries in the automatic annotated data have too many matches, which is very unlikely
    # if topk.size != k:
    #     raise ValueError('Ranking length < k')
    if tp == 0: return 0.0
    return tp/(tp+fn)

# taken from https://gist.github.com/bwhite/3726239
def precision_at_k(r, k):

    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):

    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):

    return np.mean([average_precision(r) for r in rs])

def mean_reciprocal_rank(rs):

    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def r_precision(r):

    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])
# finish taken from https://gist.github.com/bwhite/3726239

def topn_recall(predictions, n):

    tp, fn = 0, 0
    for pred_name, pred_group in predictions.groupby(['TARGET_CURRICULUM', 'TARGET_ID']):
        gold = pred_group['GOLD'].tolist()
        tp += np.sum(gold[:n])
        fn += np.sum(gold[n:])
    recall = tp / (tp + fn)

    return recall

def mean_recall_at_k(rs,rest,k):

   return np.mean([recall_at_k(r,fn,k) for r,fn in zip(rs,rest)])

def mean_precision_at_k(rs,k):

    return np.mean([precision_at_k(r,k) for r in rs])


def add_gold_column (predictions, gold, query_copies):

    correct = []

    for pred_name, pred_group in predictions.groupby(['TARGET_CURRICULUM', 'TARGET_ID']):

        pred_dict = pred_group.to_dict(orient='list')
        # if pred_group.isnull().values.any():
        #     print(pred_dict)

        for gold_name, gold_group in gold.groupby(['TARGET_CURRICULUM', 'TARGET_ID']):

            if gold_name == pred_name:

                gold_dict = gold_group.to_dict(orient='list')
                gold_queries = gold_dict['SOURCE_ID']

                for i in range(0,len(pred_dict['SOURCE_ID'])):

                    if pred_dict['SOURCE_ID'][i] in gold_queries:
                        correct.append(1)
                    elif query_copies[str(pred_dict['SOURCE'][i]).lower()]:
                        if set(query_copies[str(pred_dict['SOURCE'][i]).lower()]).intersection(set(gold_queries)):
                            correct.append(1)
                        else:
                            correct.append(0)
                    else: correct.append(0)

    predictions['GOLD'] = correct

    return predictions


def eval_ranking_per_cur (predictions, gold, query_copies, k, verbose = False):

    ranks = []
    rest = []

    for pred_name, pred_group in predictions.groupby(['TARGET_CURRICULUM', 'TARGET_ID']):

        correct = []
        pred_dict = pred_group.to_dict(orient='list')

        for gold_name, gold_group in gold.groupby(['TARGET_CURRICULUM', 'TARGET_ID']):

            if gold_name == pred_name:

                gold_dict = gold_group.to_dict(orient='list')
                gold_queries = gold_dict['SOURCE_ID']

                for i in range(0, len(pred_dict['SOURCE_ID'])):

                    if pred_dict['SOURCE_ID'][i] in gold_queries:
                        correct.append(1)

                    elif query_copies[str(pred_dict['SOURCE'][i]).lower()]:
                        if set(query_copies[str(pred_dict['SOURCE'][i]).lower()]).intersection(set(gold_queries)):
                            correct.append(1)
                        else:
                            correct.append(0)

                    else:
                        correct.append(0)

                rest.append(len(gold_queries) - sum(correct))
                ranks.append(correct)

    # map = mean_average_precision(ranks)
    recall = mean_recall_at_k(ranks,rest,k)
    # precision = mean_precision_at_k(ranks,k)
    # r_p = np.mean([r_precision(r) for r in ranks])
    mrr = mean_reciprocal_rank(ranks)

    if verbose: print(f'Target curriculum: {list(predictions["TARGET_CURRICULUM"])[0]}\n'
                      #f'Mean Average Precision (MAP): {round(map,3)}\n'
                      #f'R-precision: {round(r_p,3)}\n'
                      # f'Precision@{k}: {precision}\n'
                      f'Recall@{k}: {recall}\n'
                      f'Mean Reciprocal Rank (MRR): {round(mrr,3)}\n'
                      f'Support: {len(ranks)}\n\n')

    return {#'map': map,
            #'rp': r_p,
            'r': recall,
            #'p': precision,
            'mrr': mrr,
            'support': len(ranks)}


def eval_ranking (predictions, gold, query_copies, k, eval_filepath):

    eval = dict()

    for cur_name, cur_predictions in predictions.groupby(['TARGET_CURRICULUM']):
        eval_dict = eval_ranking_per_cur(cur_predictions,gold,query_copies,k,verbose=True)
        eval[cur_name] = eval_dict

    #map_values = [eval_dict['map'] for eval_dict in list(eval.values())]
    #rp_values = [eval_dict['rp'] for eval_dict in list(eval.values())]
    #p_values = [eval_dict['p'] for eval_dict in list(eval.values())]
    r_values = [eval_dict['r'] for eval_dict in list(eval.values())]
    mrr_values = [eval_dict['mrr'] for eval_dict in list(eval.values())]

    #w_map_values = [eval_dict['map'] * eval_dict['support'] for eval_dict in list(eval.values())]
    #w_rp_values = [eval_dict['rp'] * eval_dict['support'] for eval_dict in list(eval.values())]
    #w_p_values = [eval_dict['p'] * eval_dict['support'] for eval_dict in list(eval.values())]
    w_r_values = [eval_dict['r'] * eval_dict['support'] for eval_dict in list(eval.values())]
    w_mrr_values = [eval_dict['mrr'] * eval_dict['support'] for eval_dict in list(eval.values())]

    eval['macro-avg'] = {#'map': sum(map_values) / len(map_values),
                         #'rp': sum(rp_values) / len(rp_values),
                         #'p@k': sum(p_values) / len(p_values),
                         'r@k': sum(r_values) / len(r_values),
                         'mrr': sum(mrr_values) / len(mrr_values)}
    eval['micro-avg'] = {#'map': sum(w_map_values) / len(set(list(predictions['TARGET_ID']))),
                         #'rp': sum(w_rp_values) / len(set(list(predictions['TARGET_ID']))),
                         #'p@k': sum(w_p_values) / len(set(list(predictions['TARGET_ID']))),
                         'r@k': sum(w_r_values)/ len(set(list(predictions['TARGET_ID']))),
                         'mrr': sum(w_mrr_values) / len(set(list(predictions['TARGET_ID'])))}

    with open(eval_filepath, 'w') as outfile:
        json.dump(eval, outfile)

def eval_previous_study (ann, target_cur, target_grade, verbose = True):

    ranks = []
    by_lo = ann.groupby(['learning objective','grade'])
    for name, group in by_lo:
        # print(group)
        annotations = group['truth'].tolist()
        # print(annotations)
        annotations = [1 if i.lower() == 'yes' else 0 for i in annotations]
        ranks.append(annotations)

    # mean average precision
    map = mean_average_precision(ranks)
    # mean reciprocal rank
    mrr = mean_reciprocal_rank(ranks)

    with open(f'../eval/tfidf_baseline_{target_cur}_{target_grade}.txt', 'w') as outfile:
        outfile.write(f'TF-IDF baseline\n'
                      f'Curriculum: {target_cur}\n'
                      f'Grade: {target_grade}\n'
                      f'MAP: {map},'
                      f'MRR: {mrr}')

    if verbose:
        print(f'Mean Average Precision (MAP) of tf_idf baseline: {round(map,3)}')
        print(f'Mean Reciprocal Rank (MRR) of tf_idf baseline: {round(mrr, 3)}')


def evaluate_dev_pre_study (pred, ann, parameters_basepath, query_copies, verbose = True):

    correct = []
    true_positives = 0
    false_positives = 0

    for lo, match, match_txt in zip(pred['learning objective'], pred['id'], pred['topic/query']):
        found_pair = False
        for l_o, pre_match, pre_match_txt, truth in zip(ann['learning objective'], ann['nodeId'], ann['topic/query'], ann['truth']):
            if lo == l_o and found_pair == False:
                if match == pre_match:
                    found_pair = True
                    if truth.lower() == 'yes': correct.append('yes')
                    elif truth.lower() == 'no': correct.append('no')
                elif query_copies[str(match_txt).lower()]:
                    if pre_match_txt.lower() == match_txt.lower() and pre_match in query_copies[match_txt.lower()]:
                        found_pair = True
                        if truth.lower() == 'yes': correct.append('yes')
                        elif truth.lower() == 'no': correct.append('no')
        if found_pair == False: correct.append('not in dev')

    pred['truth'] = correct

    if 'yes' in pred['truth'].unique():
        true_positives = pred['truth'].value_counts().yes
    if 'no' in pred['truth'].unique():
        false_positives = pred['truth'].value_counts().no
    if true_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    with open(f'../eval/pre_study_{parameters_basepath}.txt', 'w') as outfile:
        outfile.write(f'{len(ann["learning objective"].unique())} learning objectives\n'
                      f'{len(ann["learning objective"])} predictions in pre study\n'
                      f'{true_positives + false_positives} predicted by system and also in pre_study\n'
                      f'{precision} correctly predicted')

    if verbose:
        print(
        f'Number of matches predicted in preliminarity study which are also predicted by this system: {true_positives + false_positives}\n'
        f'Out of these, proportion of correctly predicted matches: {precision}\n')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--gold", required=True)
    args = parser.parse_args()

    k = 5
    source_filepath = f'../data/data_dict.json'
    data_dict = read_in_data(source_filepath)
    query_copies = find_query_copies(data_dict)

    print('Evaluating results...')
    print(f'MODEL: {args.results.replace(f"../results/", "")}\n')
    test = pd.read_csv(args.gold, sep='\t', dtype={'TARGET_ID': str,'TARGET_GRADEID': str,'SOURCE_ID': str,'SOURCE_GRADEID': str})
    predictions = pd.read_csv(args.results, sep='\t', dtype={'TARGET_ID': str, 'SOURCE_ID': str})
    if 'GOLD' not in predictions.columns:
        predictions = add_gold_column(predictions,test,query_copies)
        predictions.to_csv(args.results,sep='\t',index=False) # save predictions with gold column
    eval_filepath = f'../eval/{args.results.replace(f"../results/", "").strip(".csv")}.json'
    eval_ranking(predictions, test, query_copies, k, eval_filepath)

if __name__ == '__main__':
    main()