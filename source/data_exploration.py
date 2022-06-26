import itertools
from collections import defaultdict
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
import math
import string
from utils import grade_by_age, find_age
from main import read_in_data

# In case nltk lemmatizer throws "Resource omw-1.4 not found", uncomment block below.
# import nltk
# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download('omw-1.4')

def tokenize_instances (data_dict, curriculums, subjects=None, ages=None):

    """
    Create dict with tokenized topics and queries
    :param data_dict: dict containing curriculum trees of wizenoze database
    :param curriculums: the subset of curriculums wanted
    :return: a dict of dicts = {curriculum_name : {grades : [grade1, grade2], topics: [[standard, deviation], [solar, system]]}}
    """

    tokenized_cur = defaultdict(lambda: defaultdict(list))
    if ages: age_to_grade = grade_by_age([cur for cur in curriculums.split(',')])

    for cur_id, cur in data_dict.items():
        if cur['label'] in curriculums:
            for grade_id, grade in cur['grade'].items():
                add = True
                if ages:
                    age = find_age(age_to_grade,cur['label'],grade['label'])
                    if str(age) not in ages: add = False
                if add:
                    for subj_id, subj in grade['subject'].items():
                        add = True
                        if subjects:
                            if subj['label'] not in subjects: add = False
                        if add:
                            tokenized_cur[cur['label']]['grades'].append(str(grade['label']))
                            tokenized_cur[cur['label']]['subjects'].append(str(subj['label']))
                            for unit_id, unit in subj['unit'].items():
                                tokenized_cur[cur['label']]['units'].append(str(unit['label']))
                                for topic_id, topic in unit['topic'].items():
                                    tokenized_cur[cur['label']]['topics'].append(word_tokenize(str(topic['label'])))
                                    for query_id, query in topic['query'].items():
                                        if query['label'] != '':
                                            tokenized_cur[cur['label']]['queries'].append(word_tokenize(str(query["label"])))

    return tokenized_cur


def descriptive_stats (tokenized_cur):

    """
    Print out descriptive stats of subset of curricula
    :param tokenized_cur:
    """

    n_tokens_topic = [len(topic) for cur in tokenized_cur.keys() for topic in tokenized_cur[cur]['topics']]
    n_tokens_query = [len(topic) for cur in tokenized_cur.keys() for topic in tokenized_cur[cur]['queries']]

    n_grades_per_cur = [len(tokenized_cur[cur]['grades']) for cur in tokenized_cur.keys()]
    n_subjects_per_cur = [len(tokenized_cur[cur]['subjects']) for cur in tokenized_cur.keys()]
    n_units_per_cur = [len(tokenized_cur[cur]['units']) for cur in tokenized_cur.keys()]
    n_topics_per_cur = [len(tokenized_cur[cur]['topics']) for cur in tokenized_cur.keys()]
    n_queries_per_cur = [len(tokenized_cur[cur]['queries']) for cur in tokenized_cur.keys()]

    total_n_grades = sum(n_grades_per_cur)
    total_n_subj = sum(n_subjects_per_cur)
    total_n_units = sum(n_units_per_cur)
    total_n_topics = sum(n_topics_per_cur)
    total_n_queries = sum(n_queries_per_cur)

    avg_tokens_topic = round(sum(n_tokens_topic)/total_n_topics,1)
    avg_tokens_query = round(sum(n_tokens_query)/total_n_queries,1)
    max_tokens_topic, min_tokens_topic = max(n_tokens_topic), min(n_tokens_topic)
    max_tokens_query, min_tokens_query = max(n_tokens_query), min(n_tokens_query)

    age_to_grade = grade_by_age([cur for cur in tokenized_cur.keys()])

    print('\n%%%%%%% GENERAL STATS %%%%%%%%%')
    print(f'N of CURRICULA: {len(tokenized_cur.keys())}\n'
          f'N of QUERIES: {total_n_queries}\n'
          f'Mean n of QUERIES per TOPIC: {round(total_n_queries / total_n_topics, 1)}\n'
          f'Mean n of TOPICS per UNIT: {round(total_n_topics / total_n_units, 1)}\n'
          f'Mean n of UNITS per SUBJECT: {round(total_n_units / total_n_subj, 1)}\n'
          f'Mean n of SUBJECTS per GRADE: {round(total_n_subj / total_n_grades, 1)}\n')

    print(f'Mean n of tokens per QUERY: {avg_tokens_query} ({min_tokens_query}-{max_tokens_query})\n'
          f'Mean n of tokens per TOPIC: {avg_tokens_topic} ({min_tokens_topic}-{max_tokens_topic})\n')

    for cur_name, cur in list(tokenized_cur.items()):
        print(f'CURRICULUM {cur_name}\n'
              f'N of QUERIES: {len(cur["queries"])}\n'
              f'Mean n of QUERIES per TOPIC: {len(cur["queries"]) / len(cur["topics"])}\n'
              f'Mean n of TOPICS per UNIT:{len(cur["topics"]) / len(cur["units"])}\n'
              f'Mean n of UNITS per SUBJECT:{len(cur["units"]) / len(cur["subjects"])}\n'
              f'Mean n of SUBJECTS per GRADE:{len(cur["subjects"]) / len(cur["grades"])}\n'
              f'N of GRADES:{len(cur["grades"])}\n'
              f'Ages:{[find_age(age_to_grade,cur_name,grade) for grade in cur["grades"]]}\n'
              f'Subjects:{set(cur["subjects"])}\n')

def clean_text (text, remove_punct, remove_stop_words, lemmatize):

    """
    Pre-process tokenized text
    :param text: a list of strings (tokenized text)
    :param remove_punct: if True, remove punctuation
    :param remove_stop_words: if True, remove stop words
    :param lemmatize: if True, lemmatize tokens
    :return: a list of tokens processed according to parameters
    """

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

def check_n_gram_overlap (tokenized_cur, outputfile, curriculums = None, NGRAM = 1, combi_types = ['query-query'], clean = {'remove_punct': True, 'remove_stop': True, 'lemma': True}, verbose=True):

    """
    Check n-gram overlap between topics and queries of a curriculum pair. Write out results.
    :param tokenized_cur: dict with tokenized queries and topics
    :param curriculums: curriculums to compare
    :param NGRAM: the size of the token gram, default = 1
    :param verbose: if True, print out results
    """

    n_gram_dict = defaultdict(dict)
    if not curriculums:
        curriculums = list(tokenized_cur.keys())

    for combi in itertools.combinations(curriculums,2):

        combi_dict = defaultdict(list)

        for index, target in enumerate([tokenized_cur[combi[0]]['topics'], tokenized_cur[combi[0]]['queries']]):

            add = False
            if index == 0 and ('topic-topic' in combi_types or 'topic-query' in combi_types): add = True
            elif index == 1 and ('query-topic' in combi_types or 'query-query' in combi_types): add = True

            if add:
                for instance in target:

                    clean_target = clean_text(instance, clean['remove_punct'], clean['remove_stop'], clean['lemma'])
                    ng_target = list(ngrams(clean_target, NGRAM))

                    for i, source in enumerate([tokenized_cur[combi[1]]['topics'], tokenized_cur[combi[1]]['queries']]):

                        overlap_target = []

                        add = False
                        if i == 0 and index == 0 and 'topic-topic' in combi_types: add = True
                        elif i == 0 and index == 1 and 'topic-query' in combi_types: add = True
                        elif i == 1 and index == 0 and 'query-topic' in combi_types: add = True
                        elif i == 1 and index == 1 and 'query-query' in combi_types: add = True

                        if add:
                            for item in source:
                                clean_source = clean_text(item, clean['remove_punct'], clean['remove_stop'], clean['lemma'])
                                ng_source = list(ngrams(clean_source,NGRAM))
                                overlap_ngram = cosine_similarity_ngrams(ng_target,ng_source)
                                # overlap_ngram = jaccard_distance(ng_target,ng_source)
                                overlap_target.append(overlap_ngram)

                            if overlap_target == []: max_overlap = 0.0
                            else: max_overlap = max(overlap_target)

                            if index == 0 and i == 0: combi_dict['topic-topic'].append(max_overlap)
                            elif index == 0 and i == 1: combi_dict['topic-query'].append(max_overlap)
                            elif index == 1 and i == 0: combi_dict['query-topic'].append(max_overlap)
                            else: combi_dict['query-query'].append(max_overlap)

        for combi_level in combi_dict.keys():

            avg_cosine = sum(combi_dict[combi_level])/len(combi_dict[combi_level])
            n_gram_dict[f'{combi[0]}-{combi[1]}'][combi_level] = round(avg_cosine,3)

    if verbose:
        print()
        print(f'%%%%%%%%%%% {NGRAM}-GRAM OVERLAP %%%%%%%%%%%%%')
        for cur_combi, cur_combi_dict in n_gram_dict.items():
            print(cur_combi)
            print(cur_combi_dict)

    with open(outputfile,'w') as outfile: json.dump(n_gram_dict, outfile)

def target_set_stats (target_df):

    """
    Print out info on new curriculum to be matched. The function assumes the dataframe has a learning objective column.
    :param target_df: new curriculum in pandas dataframe
    """

    print(f'%%%%%%% TARGET SET %%%%%%%\n'
          f'Total of {len(target_df["learning objective"].unique())} learning objectives to be matched from grades {target_df["grade"].unique()}.\n'
          f'Total of {len(target_df["learning objective"])} matches of learning objective and query/topic.\n'
          f'{target_df["truth"].value_counts().Yes} are positive matches.')

def generate_stats (data, curriculums, check_ngram = False):

    """
    Generate descriptive stats on curriculums
    :param data: json dict of curriculum trees
    :param curriculums: subset of curriculums from database
    """

    tokenized_cur = tokenize_instances(data, curriculums)
    descriptive_stats(tokenized_cur)
    if check_ngram:
        check_n_gram_overlap(tokenized_cur,curriculums)

def data_distribution (data):

    """
    Print out curriculum distribution of query pairs data
    :param data: pandas dataframe with query pairs
    """

    print(f'Total number of LO: {len(data["TARGET_ID"].unique())}')
    for cur, group in data.groupby(['TARGET_CURRICULUM']):
        print(f'Curriculum: {cur}\n'
              f'N of target LO: {len(group["TARGET_ID"].unique())}\n'
              f'Proportion: {len(group["TARGET_ID"].unique())/len(data["TARGET_ID"].unique())}')

def main():

    source_filepath = f'../data/data_dict.json'
    curriculums = 'ICSE,CBSE,Cambridge,English,CCSS,NGSS,Scotland'
    data_dict = read_in_data(source_filepath)
    generate_stats(data_dict,curriculums=curriculums+',Lebanon,CSTA')