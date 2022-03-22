import numpy as np


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


def eval_ranking (predictions, gold, query_copies, verbose = False):

    ranks = []
    r_ranks= []
    # gold[["TARGET_ID", "SOURCE_ID"]] = gold[["TARGET_ID", "SOURCE_ID"]].astype(str)
    for column in ['TARGET_ID', 'SOURCE_ID']:
        predictions[column] = predictions[column].astype(str).map(lambda x: x.strip('.0'))
    # print(predictions.isnull().any())

    for pred_name, pred_group in predictions.groupby(['TARGET_CURRICULUM', 'TARGET_ID']):

        pred_dict = pred_group.to_dict(orient='list')
        # if pred_group.isnull().values.any():
        #     print(pred_dict)

        for gold_name, gold_group in gold.groupby(['TARGET_CURRICULUM', 'TARGET_ID']):

            if gold_name == pred_name:

                gold_dict = gold_group.to_dict(orient='list')
                gold_queries = gold_dict['SOURCE_ID']

                correct = []

                for i in range(0,len(pred_dict['SOURCE_ID'])):

                    if pred_dict['SOURCE_ID'][i] in gold_queries:
                        correct.append(1)
                    elif query_copies[str(pred_dict['SOURCE'][i]).lower()]:
                        if set(query_copies[str(pred_dict['SOURCE'][i]).lower()]).intersection(set(gold_queries)):
                            correct.append(1)
                    else: correct.append(0)

                ranks.append(correct)
                r_ranks.append(correct[:len(gold_queries)]) # k is conditioned to R
                break

    map = mean_average_precision(ranks)
    r_p = np.mean([r_precision(r) for r in r_ranks])
    mrr = mean_reciprocal_rank(ranks)

    if verbose: print(f'Target curriculum: {list(predictions["TARGET_CURRICULUM"])[0]}\n'
                      f'Mean Average Precision (MAP): {round(map,3)}\n'
                      f'R-precision: {round(r_p,3)}\n'
                      f'Mean Reciprocal Rank (MRR): {round(mrr,3)}\n'
                      f'Support: {len(ranks)}\n\n')

    return {'map': map,
            'rp': r_p,
            'mrr': mrr,
            'support': len(ranks)}


def eval_previous_study (ann, target_cur,target_grade, verbose = True):

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



