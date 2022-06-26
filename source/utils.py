import pandas as pd
from collections import defaultdict
import json
from itertools import permutations
import re
import warnings
from sklearn.model_selection import train_test_split
import time
import requests
import random
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import os


def read_in_dump (dump_filepath):

    """
    Read in dump filepath as a pandas dataframe
    :param dump_filepath: filepath as string
    :return: pandas dataframe
    """

    levels_with_ids = ['CURRICULUM', 'CURRICULUMID',
                       'GRADE', 'GRADEID',
                       'SUBJECT', 'SUBJECTID',
                       'UNIT', 'UNITID',
                       'TOPIC', 'TOPICID',
                       'QUERY', 'QUERYID']

    webpages = ['RESULTURL', 'RESULTTITLE', 'RESULTPIN']

    df = pd.read_csv(dump_filepath, sep='\t', skiprows=1, usecols=levels_with_ids + webpages, dtype={'CURRICULUMID': str,
                                                                                                'GRADEID': str,
                                                                                                'TOPICID': str,
                                                                                                'QUERYID': str})
    # reorder columns
    df = df[levels_with_ids + webpages]
    # check nan values and replace with empty string
    df = df.fillna('')
    # correct \n in query labels
    df['QUERY'] = df['QUERY'].replace('\n', ' ', regex=True)

    return df

def dump_to_json (filepath):

    """
    Create dictionary of all curriculum trees from the dumpfile for quicker lookup
    :param filepath: dump filepath
    :return: write out json file
    """

    df = read_in_dump(filepath)

    tree = lambda: defaultdict(tree)
    data_dict = tree()

    for index, row in df.iterrows():

        # cur level {curriculumID : {'label' : curriculum, 'grade' :
        data_dict[row['CURRICULUMID']]['label'] = row['CURRICULUM']
        # grade level {gradeID : {'label' : grade, 'subject':
        data_dict[row['CURRICULUMID']]['grade'][row['GRADEID']]['label'] = row['GRADE']
        # subject level {subjectID : {'label' : subject, 'unit':
        data_dict[row['CURRICULUMID']]['grade'][row['GRADEID']]['subject'][row['SUBJECTID']]['label'] = row['SUBJECT']
        # unit level {unitID : {'label' : unit, 'topic':
        data_dict[row['CURRICULUMID']]['grade'][row['GRADEID']]['subject'][row['SUBJECTID']]['unit'][row['UNITID']][
            'label'] = row['UNIT']
        # topic level {topicID: {'label' : topic, 'query':
        data_dict[row['CURRICULUMID']]['grade'][row['GRADEID']]['subject'][row['SUBJECTID']]['unit'][row['UNITID']][
            'topic'][row['TOPICID']]['label'] = row['TOPIC']
        # query level {queryID: {'label' : query, 'pinned_pages':
        data_dict[row['CURRICULUMID']]['grade'][row['GRADEID']]['subject'][row['SUBJECTID']]['unit'][row['UNITID']][
            'topic'][row['TOPICID']]['query'][row['QUERYID']]['label'] = row['QUERY']
        # web pages {resulturl: { 'pin' : resultpin,
        data_dict[row['CURRICULUMID']]['grade'][row['GRADEID']]['subject'][row['SUBJECTID']]['unit'][row['UNITID']][
            'topic'][row['TOPICID']]['query'][row['QUERYID']]['docs'][row['RESULTURL']]['pin'] = row['RESULTPIN']
        # web pages 'pin' : resultpin
        data_dict[row['CURRICULUMID']]['grade'][row['GRADEID']]['subject'][row['SUBJECTID']]['unit'][row['UNITID']][
            'topic'][row['TOPICID']]['query'][row['QUERYID']]['docs'][row['RESULTURL']]['title'] = row['RESULTTITLE']
        # web pages 'title' : resulttitle}}}}}}

    with open(f'../data/data_dict.json','w') as outfile:
        json.dump(data_dict, outfile)


def find_query_copies (data_dict):

    """
    Find which query ID's are copies of each other. Query ID's are considered copies when they have the same uncased text
    and the same pinned documents. This is needed for a more accurate evaluation of predictions.
    :param data_dict: dict with curriculum trees
    :return: dict with uncased query text as key and a list of query ids as value.
    """

    query_docs = list()
    query_copies = defaultdict(list)

    for cur_id, cur in data_dict.items():
        for grade_id, grade in cur['grade'].items():
            for subj_id, subj in grade['subject'].items():
                for unit_id, unit in subj['unit'].items():
                    for topic_id, topic in unit['topic'].items():
                        for query_id, query in topic['query'].items():

                            docs_set = set()
                            query_label = query['label'].lower()

                            if query_label == '':

                                query_label = topic['label']
                                query_id = topic_id

                            for url, doc in query['docs'].items():

                                if doc['pin']:
                                    docs_set.add(url)

                            query_docs.append((query_id, query_label, docs_set))

    for i in range(0,len(query_docs)):
        for query in query_docs[i+1:]:
            if query_docs[i][1] == query[1] and query_docs[i][2] == query[2]:
                if query[0] not in query_copies[query[1]]:
                    query_copies[query[1]].append(query[0])
                if query_docs[i][0] not in query_copies[query[1]]:
                    query_copies[query[1]].append(query_docs[i][0])

    with open(f'../data/query_copies.json','w') as outfile:
        json.dump(query_copies, outfile)

    return query_copies


def generate_shared_docs_set (dump_filepath, included_cur, query_copies, verbose = False):

    """
    Generate data of automatically annotated query pairs. If two queries from different curricula share at least one pinned document,
    they are stored as a match. Write out query pairs data.
    :param dump_filepath: string filepath of the dump file
    :param included_cur: sub-set of curricula included
    :param query_copies: the dict of query copies
    :param verbose: if True, print info on generated data of query pairs.
    """

    dump = read_in_dump(dump_filepath)

    eval_dict = defaultdict(list)

    for url, group in dump.groupby(['RESULTURL']):

        group = group.to_dict(orient='list')
        queries = group['QUERYID']

        if len(queries) > 1:

            idx = range(0,len(queries))
            # ONLY PINNED RESULTS
            idx = [i for i in idx if group['RESULTPIN'][i] == True]

            if len(idx) > 1:

                # COMBINATIONS [(0,1),(0,2),(1,2)] != PERMUTATIONS [(0,1),(1,0)...]
                combis = permutations(idx, 2)

                for combi in combis:

                    add = True

                    # TARGET AND SOURCE ARE NOT FROM THE SAME CURRICULUM
                    if group['CURRICULUM'][combi[0]] != group['CURRICULUM'][combi[1]]:

                        # SOME CURRICULA ARE NOT INCLUDED
                        if group['CURRICULUM'][combi[0]] in included_cur and group['CURRICULUM'][combi[1]] in included_cur:

                            # EMPTY QUERY IS REPLACED BY RESPECTIVE TOPIC
                            for i in [0,1]:
                                if group['QUERY'][combi[i]] == '':
                                    group['QUERY'][combi[i]] = group['TOPIC'][combi[0]]

                            # EXCLUDE DUPLICATES
                            if str(group['QUERY'][combi[0]]).lower() == str(group['QUERY'][combi[1]]).lower():
                                if query_copies[str(group['QUERY'][combi[0]]).lower()]:
                                    if group['QUERYID'][combi[0]] in query_copies[str(group['QUERY'][combi[0]]).lower()]:
                                        if group['QUERYID'][combi[1]] in query_copies[str(group['QUERY'][combi[0]]).lower()]:
                                            add = False

                            if add:
                                eval_dict['TARGET_CURRICULUM'].append(group['CURRICULUM'][combi[0]])
                                eval_dict['TARGET'].append(group['QUERY'][combi[0]])
                                eval_dict['TARGET_ID'].append(group['QUERYID'][combi[0]])
                                eval_dict['TARGET_PATH'].append(
                                    f'{group["CURRICULUM"][combi[0]]}>{group["GRADE"][combi[0]]}>{group["SUBJECT"][combi[0]]}>{group["UNIT"][combi[0]]}>{group["TOPIC"][combi[0]]}')
                                eval_dict['TARGET_GRADEID'].append(f'{group["GRADEID"][combi[0]]}')
                                eval_dict['SOURCE'].append(group['QUERY'][combi[1]])
                                eval_dict['SOURCE_ID'].append(group['QUERYID'][combi[1]])
                                eval_dict['SOURCE_PATH'].append(
                                    f'{group["CURRICULUM"][combi[1]]}>{group["GRADE"][combi[1]]}>{group["SUBJECT"][combi[1]]}>{group["UNIT"][combi[1]]}>{group["TOPIC"][combi[1]]}')
                                eval_dict['SOURCE_GRADEID'].append(f'{group["GRADEID"][combi[1]]}')
                                eval_dict['SOURCE_CURRICULUM'].append(group['CURRICULUM'][combi[1]])

    # order rows by target curriculum and query
    df = pd.DataFrame(eval_dict)
    df = df.drop_duplicates(subset=['TARGET_ID', 'SOURCE_ID'])  # WHEN TARGET AND SOURCE SHARE MORE THAN ONE WEB PAGE
    df = df.sort_values(by=['TARGET_CURRICULUM','TARGET_ID'])
    df.to_csv(f"../data/query_pairs_{included_cur}.csv",sep="\t",index=False)

    if verbose:
        print('\n####### Set of queries that share webpages #######')
        print(f'{len(df["TARGET_ID"].tolist())} unique query pairs that share at least one web result\n'
              f'{sum([1 for t, s in zip(df["TARGET"], df["SOURCE"]) if t == s])} query pairs have the exact same label\n'
              f'{len(set(list(df["TARGET_CURRICULUM"].unique()) + list(df["SOURCE_CURRICULUM"].unique())))} curricula covered'
              )
        dist = [(cur,len(group['TARGET'].tolist())/len(df['TARGET'].tolist())) for cur, group in df.groupby(['TARGET_CURRICULUM'])]
        dist.sort(key=lambda x:x[1],reverse=True)
        for cur, p in dist:
            print(f"{cur}: {round(p,4)}")

def grade_by_age (included_cur, age_filepath = '../data/MASTER Reading levels and age filter settings (pwd 123456).xlsx'):

    """
    Create a dictionary with age as key and grades as value.
    :param included_cur: sub-set of curricula included
    :return: dict
    """

    if not os.path.isfile(age_filepath):
        print(f'{age_filepath} not found. Please add this file to get target age per grade')
        exit()
    else:
        age_df = pd.read_excel(age_filepath)
        age_df = age_df.fillna('')
        age_to_grade = defaultdict(list)
        for age, group in age_df.groupby(['Column1.nodes.defaultAge']):
            group = group.to_dict(orient='list')
            for i in range(0,len(group['Column1.nodes.description'])):
                cur = group['Column1.description'][i]
                curid = group['Column1.curriculumId'][i]
                gradeid = group['Column1.nodes.nodeId'][i]
                grade = group['Column1.nodes.description'][i]
                if group['Column1.description'][i] == 'CSTA':
                    grade = 'Level ' + grade
                if cur in included_cur:
                    age_key = age
                    if 'age' in group['NEW VALUE'][i].lower():
                        age_key = re.sub(r'^[A-Z]*[a-z]* *', '', group['NEW VALUE'][i])
                        age_key = re.sub(r', *[A-Z]* *[1-5]*', '', age_key)

                    age_to_grade[int(age_key)].append({'CURRICULUM': cur,
                                                       'CURRICULUMID': str(curid),
                                                        'GRADEID': str(gradeid).strip('.0'),
                                                        'GRADE': grade})

    return age_to_grade

def find_age (age_to_grade,curriculum,grade):

    """
    Given a curriculum and a grade, find the corresponding age
    :param age_to_grade: dict with age as key and grades as value
    :param curriculum: curriculum to which grade belongs
    :param grade: grade to convert into age information
    :return: age as int
    """

    target_age = -1
    if grade.startswith(curriculum): grade = grade.replace(curriculum+' ', '')
    for age, grades in age_to_grade.items():
        for dict in grades:
            if (curriculum == dict['CURRICULUM'] or curriculum == dict['CURRICULUMID']) and (grade == dict['GRADE'] or grade == dict['GRADEID']):
                target_age = age
                break

    if target_age == -1:

        warnings.warn(
        f"Grade {grade} from curriculum {curriculum} not found in age_to_grade document. "
        f"Grade is either a nan value or you need to make sure grade is in 'Reading levels and age filter settings' file.")

    return target_age

def generate_doc_sums (data_dict, source_curriculums):

    """
    Write out file with summary of documents per query.
    :param data_dict: dict with curriculum trees
    :param source_curriculums: sub-set of curriculums included
    """

    start_time = time.perf_counter()
    with open(f'../data/doc_sums.csv','w') as out:
        out.write(f'queryId\turl\tsumText\n')

        for cur_id, cur in data_dict.items():
            if cur['label'] in source_curriculums:
                for grade_id, grade in cur['grade'].items():
                    for subject_id, subject in grade['subject'].items():
                        for unit_id, unit in subject['unit'].items():
                            for topic_id, topic in unit['topic'].items():
                                for query_id, query in topic['query'].items():

                                    url = f'https://api.wizenoze.com/v4/curriculum/node/query/{query_id}/results?userUUID=123456&sessionUUID=123456&userType=teacher&resultSize=5'
                                    headers = {'Authorization': '0b9cb12f-b960-47cb-b7fe-f47253cc4f1e'}
                                    response = requests.get(url, headers=headers)
                                    if response.status_code == 200:
                                        resp = response.json()["results"]
                                        for doc_resp in resp:
                                            # get summary of pinned docs only
                                            if doc_resp["fullUrl"] in query['docs'].keys():
                                                if query["docs"][doc_resp["fullUrl"]]["pin"]:
                                                    out.write(f'{query_id}\t{doc_resp["fullUrl"]}\t{doc_resp["summaryText"]}\n')
                                    else:
                                        print(f'{response.status_code} response status for query {query_id}')
                                        continue

    end_time = time.perf_counter()
    print(f'It took {end_time - start_time:0.4f} seconds to get all summary texts wtih API calls')

def target_subset(target_data, n):

    target_sub = defaultdict(list)
    percent = {'ICSE': 39, 'CBSE': 37, 'Cambridge': 11, 'English': 5, 'NGSS': 2, 'CCSS': 2, 'Scotland': 1}
    cur = target_data['TARGET_CURRICULUM'][0]
    cur_n = int((n * percent[cur])/100)

    for n in range(cur_n):
        i = random.randint(0,len(target_data['TARGET']))
        for key in target_data.keys():
            target_sub[key].append(target_data[key][i])

    return target_sub

def generate_train_test(target_data, random_seed = 42, verbose=False):

    """
    Given dataset of query pairs, split into train, dev and test and write them out.
    :param target_data: pandas dataframe with query pairs.
    :param random_seed: an int.
    :param verbose: if True, print info on train, dev and test sets of query pairs.
    """

    data_train = pd.DataFrame()
    data_dev = pd.DataFrame()
    data_test = pd.DataFrame()

    filepaths = [f'../data/train_query_pairs_{random_seed}.csv',
                 f'../data/dev_query_pairs_{random_seed}.csv',
                 f'../data/test_query_pairs_{random_seed}.csv']

    train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
    for cur_name, cur_group in target_data.groupby(['TARGET_CURRICULUM']):
        train, test = train_test_split(list(cur_group.groupby(['TARGET_ID'])), test_size=1-train_ratio, random_state=random_seed)
        dev, test = train_test_split(test, test_size=test_ratio/(test_ratio+val_ratio), random_state=random_seed)

        for split in [train, dev, test]:
            to_concat = pd.concat([target_series for target_name, target_series in split])
            if split == train: data_train = pd.concat([data_train, to_concat])
            elif split == dev: data_dev = pd.concat([data_dev, to_concat])
            else: data_test = pd.concat([data_test, to_concat])

    for data, filepath in [(data_train,filepaths[0]),(data_dev, filepaths[1]),(data_test,filepaths[2])]:
        data = data.sort_values(['TARGET_CURRICULUM','TARGET_ID'])
        data.to_csv(filepath, sep='\t', index=False)

        if verbose:

            if filepath == filepaths[0]: set_name = 'Train'
            elif filepath == filepaths[1]: set_name = 'Dev'
            else: set_name = 'Test'

            print(f'\n####### {set_name} set of queries that share documents #######')
            print(f'{len(data)} query pairs that share at least one document\n'
                  f'{sum([1 for t, s in zip(data["TARGET"], data["SOURCE"]) if str(t).lower() == str(s).lower()])} query pairs have the same label\n'
                  f'{len(set(list(data["TARGET_CURRICULUM"].unique())))} curricula covered\n')

            dist = [(cur, len(group['TARGET'].tolist()) / len(data['TARGET'].tolist())) for cur, group in
                        data.groupby(['TARGET_CURRICULUM'])]

            dist.sort(key=lambda x: x[1], reverse=True)

            for cur, p in dist:
                print(f"{cur}: {round(p, 4)}\n")

def generate_test_lebanon (filepath_to_excel, output_file):

    """
    Unify grade excel sheets into one pandas dataframe with Lebanon learning objectives
    :param filepath_to_excel: excel with pre-study on Lebanon
    """

    target_data = pd.DataFrame()
    for grade in [('lebanon_grade6 txt','Elementary - Year 6'),
                  ('lebanon_grade7 txt (1)','Intermediate - Year 7'),
                  #('lebanon_grade8 txt (1)','Intermediate - Year 8'),
                  ('lebanon_grade10 txt (1)','Secondary - Year 10')]:
        target_grade = pd.read_excel(filepath_to_excel, grade[0])
        target_grade["TARGET_GRADE"] = grade[1]
        target_grade = target_grade.rename(columns={'Topic/Queries from CT': 'SOURCE',
                                                  'Topics/Queries from CT': 'SOURCE',
                                                  'topic/query': 'SOURCE',
                                                  'learning objective': 'TARGET',
                                                  'path': 'SOURCE_PATH',
                                                  'queryId': 'SOURCE_ID',
                                                  'Lebanon Learning Objective': 'TARGET',
                                                  'Matching to the learning Objective (Yes/No)': 'GOLD',
                                                  'curriculumId': 'SOURCE_CURRICULUM'})
        target_data = pd.concat([target_data,target_grade],ignore_index=True)

    target_data['TARGET_CURRICULUM'] = 'Lebanon'
    target_data['TARGET_PATH'] = target_data['TARGET_GRADE'].apply(lambda x : f'Lebanon>{x}> > > ')
    # only save rows where gold is 'yes' and qualifier is 'query'
    target_data = target_data[target_data['qualifier'] == 'QUERY']
    target_data = target_data[target_data['GOLD'].isin(['Yes','YEs','yes'])]
    target_data = target_data.drop(['nodeId','Explanation','Comments on why not matching'],axis=1)
    target_data['SOURCE_ID'] = target_data['SOURCE_ID'].apply(lambda x:int(x))

    target_ids = []
    id = 0
    for target, group in target_data.groupby(['TARGET','TARGET_GRADE'], sort=False):
        for i in range(len(group)): target_ids.append(id)
        id += 1
    assert len(target_ids) == len(target_data)
    target_data['TARGET_ID'] = target_ids

    target_data.to_csv(output_file, index=False, sep='\t')

def generate_triplets_file (FILEPATH, TRIPLETS_FILE):

    """
    For training SBERT on triplet architecture, generate file where each row is a instance containing the ids of the
    anchor text, the positive example and the negative example.
    :param FILEPATH: filepath to train or dev set
    :param TRIPLETS_FILE: filepath to save triplets file
    """

    positive_pairs = pd.read_csv(FILEPATH, sep='\t', dtype={'TARGET_ID': str,
                                                            'SOURCE_ID': str,
                                                            'TARGET_GRADEID': str,
                                                            'SOURCE_GRADEID': str})

    positive_pairs = positive_pairs.drop_duplicates(['TARGET_ID'])
    target_queries = positive_pairs['TARGET_ID'].tolist()
    positive_queries = positive_pairs['SOURCE_ID'].tolist()

    # find a negative example for each target query from the remaining queries (not in target curriculum, not in positive set and not the search query)
    negative_queries = []
    queries = [(cur, query) for cur, query in
               zip(positive_pairs['TARGET_CURRICULUM'].tolist(), positive_pairs['TARGET_ID'].tolist())]
    random.seed(42)

    for group_name, group in positive_pairs.groupby(['TARGET_CURRICULUM', 'TARGET_ID']):
        for i, row in group.iterrows():
            neg = random.choice(queries)
            while neg[0] == group_name[0] or neg[1] in group['SOURCE_ID'].tolist() or neg[1] in group[
                'TARGET_ID'].tolist():
                neg = random.choice(queries)
            negative_queries.append(neg[1])

    assert len(negative_queries) == len(
        positive_pairs), f'Not the same number of negative and positive examples. {len(negative_queries)} negative examples, {len(positive_pairs)} positive examples'

    with open(f'{TRIPLETS_FILE}', 'w') as out:
        out.write('qid' + '\t' + 'pos_id' + '\t' + 'neg_id' + '\n')
        for i in range(len(target_queries)):
            out.write(f'{target_queries[i]}\t{positive_queries[i]}\t{negative_queries[i]}\n')

def generate_query_file(DATA_DICT_FILE, QUERY_FILE, queries, doc_sums):

    """
    Given query ids that form the triples for training, store corresponding texts.
    :param DATA_DICT_FILE: dict with curriculum trees
    :param QUERY_FILE: filepath to write out file with texts
    :param queries: query ids that appear in training instances
    :param doc_sums: in case summary of documents is used, a dict with doc summaries per query id
    """

    with open(DATA_DICT_FILE) as json_file:
        data = json.load(json_file)

    rows = []
    for cur_id, cur in data.items():
        for grade_id, grade in cur['grade'].items():
            for subject_id, subject in grade['subject'].items():
                for unit_id, unit in subject['unit'].items():
                    for topic_id, topic in unit['topic'].items():
                        for query_id, query in topic['query'].items():

                            if query_id in queries:
                                label = query['label']
                                if label == '': label = topic['label']
                                query_dict = {'id': query_id,
                                              'label': label,
                                              'doc_titles': ' '.join(
                                                  [doc_info['title'] for doc_info in query['docs'].values() if
                                                   doc_info['pin']]),
                                              'doc_sums_1sent': '',
                                              'doc_sums_nsent': ''}

                                # Including organic documents
                                # query_dict = {'id': query_id,
                                #              'label': label,
                                #              'doc_titles': ' '.join(
                                #                  [doc_info['title'] for doc_info in query['docs'].values()]),
                                #               'doc_sums_1sent': '',
                                #               'doc_sums_nsent': ''}

                                if doc_sums[query_id]:
                                    doc_sums_1sent = []
                                    doc_sums_nsent = []
                                    for sum in doc_sums[query_id]:
                                        sents = sent_tokenize(sum)
                                        doc_sums_1sent.append(sents[0])
                                        tags_label = pos_tag(word_tokenize(label))
                                        nouns = [word for word, pos in tags_label if pos.startswith('NN')]
                                        for sent in sents:
                                            tokens = word_tokenize(sent)
                                            for token in tokens:
                                                if token in nouns and sent not in doc_sums_nsent:
                                                    doc_sums_nsent.append(sent)
                                    query_dict['doc_sums_1sent'] = ' '.join(doc_sums_1sent)
                                    query_dict['doc_sums_nsent'] = ' '.join(doc_sums_nsent)

                                rows.append(query_dict)

    with open(QUERY_FILE, 'w') as out:
        out.write('id' + '\t' + 'label' + '\t' + 'docTitles' + '\t' + 'docSums1sent' + '\t' + 'docSumsNsent' + '\n')
        for query_dict in rows:
            out.write(
                f'{query_dict["id"]}\t{query_dict["label"]}\t{query_dict["doc_titles"]}\t{query_dict["doc_sums_1sent"]}'
                f'\t{query_dict["doc_sums_nsent"]}\n')

def generate_info_file(data_dict_filepath, info_filepath, doc_sums, age_to_grade, curriculums):

    with open(data_dict_filepath) as json_file:
        data = json.load(json_file)

    rows = []
    for cur_id, cur in data.items():
        if cur['label'] in curriculums:
            for grade_id, grade in cur['grade'].items():
                for subject_id, subject in grade['subject'].items():
                    for unit_id, unit in subject['unit'].items():
                        for topic_id, topic in unit['topic'].items():
                            for query_id, query in topic['query'].items():
                                label = query['label']
                                if label == '': label = topic['label']
                                query_dict = {'id': str(query_id),
                                              'query_term': label.strip(),
                                              'topic': topic['label'].strip(),
                                              'subject': subject['label'].strip(),
                                              'grade': grade['label'].strip(),
                                              'curriculum': cur['label'].strip(),
                                              'doc_titles': ' '.join(
                                                  [doc_info['title'].strip() for doc_info in query['docs'].values() if
                                                   doc_info['pin']]),
                                              'doc_sums_1sent': '',
                                              'doc_sums_nsent': ''}

                                age = find_age(age_to_grade, cur['label'], grade['label'])
                                query_dict['age'] = str(age)

                                # Including organic documents
                                # query_dict = {'id': query_id,
                                #              'label': label,
                                #              'doc_titles': ' '.join(
                                #                  [doc_info['title'] for doc_info in query['docs'].values()]),
                                #               'doc_sums_1sent': '',
                                #               'doc_sums_nsent': ''}

                                if doc_sums[query_id]:
                                    doc_sums_1sent = []
                                    doc_sums_nsent = []
                                    for sum in doc_sums[query_id]:
                                        sents = sent_tokenize(sum)
                                        doc_sums_1sent.append(sents[0])
                                        tags_label = pos_tag(word_tokenize(label))
                                        nouns = [word for word, pos in tags_label if pos.startswith('NN')]
                                        for sent in sents:
                                            tokens = word_tokenize(sent)
                                            for token in tokens:
                                                if token in nouns and sent not in doc_sums_nsent:
                                                    doc_sums_nsent.append(sent)
                                    query_dict['doc_sums_1sent'] = ' '.join(doc_sums_1sent).strip()
                                    query_dict['doc_sums_nsent'] = ' '.join(doc_sums_nsent).strip()

                                for key,v in query_dict.items():
                                    new = v.replace('\n','')
                                    new = new.replace('\t','')
                                    query_dict[key] = new

                                rows.append(query_dict)

    with open(info_filepath, 'w', encoding="utf-8") as out:
        headers = ['query_id', 'query_term', 'doc_titles', 'doc_sums_1sent', 'doc_sums_nsents', 'topic', 'subject', 'grade',
                   'age', 'curriculum']
        headers = '\t'.join(headers) + '\n'
        out.write(headers)
        for query_dict in rows:
            out.write(
                f'{query_dict["id"]}\t{query_dict["query_term"]}\t{query_dict["doc_titles"]}\t{query_dict["doc_sums_1sent"]}'
                f'\t{query_dict["doc_sums_nsent"]}\t{query_dict["topic"]}\t{query_dict["subject"]}\t{query_dict["grade"]}'
                f'\t{query_dict["age"]}\t{query_dict["curriculum"]}\n')