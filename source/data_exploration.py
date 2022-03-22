from utils import dump_to_json
from collections import defaultdict
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
import math
import string

from itertools import combinations
import ssl

# In case nltk lemmatizer throws "Resource omw-1.4 not found", uncomment block below.
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download('omw-1.4')

def tokenize_instances (data_dict):

    # create dict with tokenized topics and queries per curriculum
    tokenized_cur = defaultdict(lambda: defaultdict(list))

    for cur_id, cur in data_dict.items():
        for grade_id, grade in cur['grade'].items():
            for subj_id, subj in grade['subject'].items():
                for unit_id, unit in subj['unit'].items():
                    for topic_id, topic in unit['topic'].items():
                        tokens_topic = word_tokenize(str(topic['label']))
                        tokenized_cur[cur['label']]['topics'].append(tokens_topic)
                        for query_id, query_label in topic['query'].items():
                            if query_label != '':
                                tokens_query = word_tokenize(str(query_label))
                                tokenized_cur[cur['label']]['queries'].append(tokens_query)

    return tokenized_cur


def descriptive_stats (tokenized_cur):

    n_tokens_topic = [len(topic) for cur in tokenized_cur.keys() for topic in tokenized_cur[cur]['topics']]
    n_tokens_query = [len(topic) for cur in tokenized_cur.keys() for topic in tokenized_cur[cur]['queries']]
    n_topics_per_cur = [len(tokenized_cur[cur]['topics']) for cur in tokenized_cur.keys()]
    n_queries_per_cur = [len(tokenized_cur[cur]['queries']) for cur in tokenized_cur.keys()]
    total_n_topics, total_n_queries = sum(n_topics_per_cur), sum(n_queries_per_cur)

    # Mean, max, min number of topics and queries in a curriculum
    avg_topics_total = round(total_n_topics / len(tokenized_cur.keys()))
    avg_queries_total = round(total_n_queries / len(tokenized_cur.keys()))
    avg_queries_per_topic = round(total_n_queries / total_n_topics, 1)
    min_n_topics, max_n_topics = min(n_topics_per_cur), max(n_topics_per_cur)
    min_n_queries, max_n_queries = min(n_queries_per_cur), max(n_queries_per_cur)

    # Mean, min and max n of tokens per topic and per query
    avg_tokens_topic, avg_tokens_query = round(sum(n_tokens_topic)/total_n_topics,1), round(sum(n_tokens_query)/total_n_queries,1)
    max_tokens_topic, min_tokens_topic = max(n_tokens_topic), min(n_tokens_topic)
    max_tokens_query, min_tokens_query = max(n_tokens_query), min(n_tokens_query)

    print('\n%%%%%%% GENERAL STATS %%%%%%%%%')
    print(f'N of CURRICULA: {len(tokenized_cur.keys())}\n'
          f'N of TOPICS: {total_n_topics}\n'
          f'N of QUERIES: {total_n_queries}\n')
    for cur in list(tokenized_cur.keys()):
        print(f'CURRICULUM {cur}\n'
              f'N of TOPICS: {len(tokenized_cur[cur]["topics"])}\n'
              f'N of QUERIES: {len(tokenized_cur[cur]["queries"])}\n')
    print(f'Mean n of TOPICS per CURRICULUM: {avg_topics_total} ({min_n_topics}-{max_n_topics})\n'
          f'Mean n of QUERIES per CURRICULUM: {avg_queries_total} ({min_n_queries}-{max_n_queries})\n'
          f'Mean n of QUERIES per TOPIC: {avg_queries_per_topic}\n'
          f'Mean n of tokens per TOPIC: {avg_tokens_topic} ({min_tokens_topic}-{max_tokens_topic})\n'
          f'Mean n of tokens per QUERY: {avg_tokens_query} ({min_tokens_query}-{max_tokens_query})\n')

def clean_text (text, remove_punct = True, remove_stop_words = True, lemmatize = True):

    punct = string.punctuation
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    clean_text = text.copy()
    if lemmatize:
        clean_text = [lemmatizer.lemmatize(token) for token in clean_text]
    if remove_punct:
        clean_text = [token for token in clean_text if token not in punct]
    if remove_stop_words:
        clean_text = [token for token in clean_text if token not in stop_words]

    return clean_text

def cosine_similarity_ngrams(a, b):

    # taken from https://gist.github.com/gaulinmp/da5825de975ed0ea6a24186434c24fe4

    vec1 = Counter(a)
    vec2 = Counter(b)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0

    return float(numerator) / denominator

def jaccard_distance(a, b):

    # taken from https://gist.github.com/gaulinmp/da5825de975ed0ea6a24186434c24fe4

    a = set(a)
    b = set(b)

    return 1.0 * len(a&b)/len(a|b)

def check_n_gram_overlap (tokenized_cur, target_cur = 'Lebanon', source_cur = [], NGRAM = 1):

    n_gram_dict = defaultdict(dict)

    if source_cur == []:
        source_cur = list(tokenized_cur.keys())
        source_cur.remove(target_cur)

    for combi in [(target_cur,source) for source in source_cur]:
        combi_dict = defaultdict(list)
        for index, target in enumerate([tokenized_cur[combi[0]]['topics'], tokenized_cur[combi[0]]['queries']]):
            # print(index)
            for instance in target:
                # print(instance)
                clean_target = clean_text(instance)
                # print(clean_target)
                ng_target = list(ngrams(clean_target, NGRAM))
                # print(ng_target)
                for i, source in enumerate([tokenized_cur[combi[1]]['topics'], tokenized_cur[combi[1]]['queries']]):
                    overlap_target = []
                    # print(i)
                    for item in source:
                        clean_source = clean_text(item)
                        # print(clean_source)
                        ng_source = list(ngrams(clean_source,NGRAM))
                        # print(ng_source)
                        overlap_ngram = cosine_similarity_ngrams(ng_target,ng_source)
                        # overlap_ngram = jaccard_distance(ng_target,ng_source)
                        # print(cosine_ngram)
                        overlap_target.append(overlap_ngram)

                    if overlap_target == []: max_overlap = 0.0
                    else: max_overlap = max(overlap_target)

                    if index == 0 and i == 0 : combi_dict['topic-topic'].append(max_overlap)
                    elif index == 0 and i == 1: combi_dict['topic-query'].append(max_overlap)
                    elif index == 1 and i == 0: combi_dict['query-topic'].append(max_overlap)
                    else: combi_dict['query-query'].append(max_overlap)

        for combi_level in combi_dict.keys():
            avg_cosine = sum(combi_dict[combi_level])/len(combi_dict[combi_level])
            n_gram_dict[f'{combi[0]}-{combi[1]}'][combi_level] = round(avg_cosine,3)

    print()
    print(f'%%%%%%%%%%% {NGRAM}-GRAM OVERLAP %%%%%%%%%%%%%')
    for cur_combi, cur_combi_dict in n_gram_dict.items():
        print(cur_combi)
        print(cur_combi_dict)
    char_remove = "[]'"
    with open(f"../data/{NGRAM}_gram_overlap_{target_cur}_{str(source_cur).strip(char_remove)}.json",'w') as outfile: json.dump(n_gram_dict, outfile)

def target_set_stats (target_df):

    print(f'%%%%%%% TARGET SET %%%%%%%\n'
          f'Total of {len(target_df["learning objective"].unique())} learning objectives to be matched from grades {target_df["grade"].unique()}.\n'
          f'Total of {len(target_df["learning objective"])} matches of learning objective and query/topic.\n'
          f'{target_df["truth"].value_counts().Yes} are positive matches.')


def generate_stats (data):

    tokenized_cur = tokenize_instances(data)
    descriptive_stats(tokenized_cur)
    # check_n_gram_overlap(tokenized_cur)