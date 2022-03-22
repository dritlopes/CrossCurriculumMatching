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

def dump_to_json (filepath):

    levels = ['CURRICULUM', 'GRADE', 'SUBJECT', 'UNIT', 'TOPIC', 'QUERY']
    levels_with_ids = ['CURRICULUM','CURRICULUMID',
                       'GRADE','GRADEID',
                       'SUBJECT','SUBJECTID',
                       'UNIT','UNITID',
                       'TOPIC','TOPICID',
                       'QUERY','QUERYID']
    webpages = ['RESULTURL', 'RESULTTITLE', 'RESULTPIN']

    df = pd.read_csv(filepath, sep = '\t', skiprows = 1, usecols = levels_with_ids + webpages, dtype= {'CURRICULUMID': str,
                                                                                     'GRADEID': str,
                                                                                     'TOPICID': str,
                                                                                     'QUERYID': str})
    # reorder columns
    df = df[levels_with_ids + webpages]
    # print(df['CURRICULUM'].unique())

    # check nan values and replace with empty string
    # df_no_id = df[levels]
    # print(df_no_id.isnull().any())
    # print(df_no_id.isnull().groupby(df['CURRICULUM']).all())
    df = df.fillna('')
    # convert ids to strings
    # for column_id in ['CURRICULUMID', 'GRADEID', 'SUBJECTID', 'UNITID', 'TOPICID', 'QUERYID']:
    #     df[column_id] = df[column_id].astype(str)
    #     if column_id in {'GRADEID', 'UNITID', 'SUBJECTID'}:
    #         df[column_id] = df[column_id].map(lambda x: x.strip('.0'))

    # correct \n in query labels
    df['QUERY'] = df['QUERY'].replace('\n', ' ',regex=True)
    # for query in df['QUERY'].tolist():
    #     if '\n' in query: print('New line is still there')

    tree = lambda: defaultdict(tree)
    data_dict = tree()

    for index, row in df.iterrows():
        # if row['CURRICULUM'] in list(included_cur.split(',')):

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

    with open('../data/data_dict.json','w') as outfile:
        json.dump(data_dict, outfile)


def find_query_copies (data_dict):

    query_docs = list()
    query_copies = defaultdict(set)

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
                query_copies[query[1]].add(query[0])
                query_copies[query[1]].add(query_docs[i][0])

    return query_copies


def generate_shared_docs_set (dump_filepath, included_cur, query_copies, verbose = False):

    cols = ['CURRICULUM', 'CURRICULUMID', 'QUERY', 'QUERYID', 'RESULTURL', 'RESULTTITLE', 'RESULTPIN','GRADE','GRADEID','SUBJECT','UNIT','TOPIC','TOPICID']
    dump = pd.read_csv(dump_filepath,sep='\t', skiprows = 1, usecols = cols, dtype= {'CURRICULUMID': str,
                                                                                     'GRADEID': str,
                                                                                     'TOPICID': str,
                                                                                     'QUERYID': str})
    dump = dump.fillna('')
    eval_dict = defaultdict(list)

    # for column_id in ['CURRICULUMID', 'GRADEID', 'TOPICID', 'QUERYID']:
    #     dump[column_id] = dump[column_id].astype(str)
    #     if column_id in {'GRADEID'}:
    #         dump[column_id] = dump[column_id].map(lambda x: x.strip('.0'))

    dump['QUERY'] = dump['QUERY'].replace('\n', ' ', regex=True)
    # for query in df['QUERY'].tolist():
    #     if '\n' in query: print('New line is still there')

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
                                    # group['QUERYID'][combi[i]] = group['TOPICID'][combi[0]]
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


def grade_by_age (included_cur):

    age_filepath = '../data/MASTER Reading levels and age filter settings (pwd 123456).xlsx'
    age_df = pd.read_excel(age_filepath)
    age_df = age_df.fillna('')
    age_to_grade = defaultdict(list)
    for age, group in age_df.groupby(['Column1.nodes.defaultAge']):
        group = group.to_dict(orient='list')
        for i in range(0,len(group['Column1.nodes.description'])):
            cur = group['Column1.description'][i]
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
                                                'GRADEID': str(gradeid).strip('.0'),
                                                'GRADE': grade})

    return age_to_grade

def find_age (age_to_grade,curriculum,grade):

    target_age = -1
    if grade.startswith(curriculum): grade = grade.replace(curriculum+' ', '')
    for age, grades in age_to_grade.items():
        for dict in grades:
            if curriculum == dict['CURRICULUM'] and (grade == dict['GRADE'] or grade == dict['GRADEID']):
                target_age = age
                break

    if target_age == -1:

        warnings.warn(
        f"Grade {grade} from curriculum {curriculum} not found in age_to_grade document. "
        f"Grade is either a nan value or you need to make sure grade is in 'Reading levels and age filter settings' file.")

    return target_age

def get_info_for_pred(source,target):

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

def generate_doc_sums (data_dict, source_curriculums):

    start_time = time.perf_counter()
    with open('../data/doc_sums.csv','w') as out:
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


def generate_train_test(target_data, verbose=False):

    data_train = pd.DataFrame()
    data_dev = pd.DataFrame()
    data_test = pd.DataFrame()
    filepaths = ['../data/train_query_pairs.csv', '../data/dev_query_pairs.csv', '../data/test_query_pairs.csv']

    for cur_name, cur_group in target_data.groupby(['TARGET_CURRICULUM']):
        train, test = train_test_split(list(cur_group.groupby(['TARGET_ID'])), test_size=0.2, random_state=42)
        train, dev = train_test_split(train, test_size=0.1, random_state=42)
        for split in [train, dev, test]:
            to_concat = pd.concat([target_series for target_name, target_series in split])
            if split == train: data_train = pd.concat([data_train, to_concat])
            elif split == dev: data_dev = pd.concat([data_dev, to_concat])
            else: data_test = pd.concat([data_test, to_concat])

    for data, filepath in [(data_train,filepaths[0]),(data_dev, filepaths[1]),(data_test,filepaths[2])]:
        data = data.sort_values(['TARGET_CURRICULUM','TARGET_ID'])
        data.to_csv(filepath, sep='\t', index=False)

    # train, test = train_test_split(list(target_data['TARGET_CURRICULUM'].unique()), test_size=0.2, random_state=42)
    # filepaths = ['../data/train_query_pairs.csv', '../data/test_query_pairs.csv']
    # data_train = pd.DataFrame()
    # data_test = pd.DataFrame()
    #
    # for cur_name, group in target_data.groupby(['TARGET_CURRICULUM']):
    #     if cur_name in train:
    #         data_train = pd.concat([data_train, group])
    #     elif cur_name in test:
    #         data_test = pd.concat([data_test, group])
    #
    # for data, filepath in [(data_train,filepaths[0]),(data_test,filepaths[1])]:
    #     data.to_csv(filepath, sep='\t', index=False)

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

    return data_train, data_dev, data_test


def generate_test_lebanon (filepath_to_excel):

        # TODO test this code after changes
        target_data = pd.DataFrame()
        for grade in ['Elementary - Year 6', 'Intermediate - Year 7', 'Intermediate - Year 8', 'Secondary - Year 10']:
            target_grade = pd.read_excel(filepath_to_excel, grade)
            target_grade["grade"] = grade
            target_data = pd.concat([target_data,target_grade],ignore_index=True)
        target_data = target_data.rename(columns={'Topic/Queries from CT': 'topic/query',
                                                   'Topics/Queries from CT': 'topic/query',
                                                   'Lebanon Learning Objective': 'learning objective',
                                                   'Matching to the learning Objective (Yes/No)': 'truth'})
        target_data.to_csv(f'../data/test_Lebanon.csv', index=False)

def generate_triplets_file(FILEPATH, TRIPLETS_FILE):

    positive_pairs = pd.read_csv(FILEPATH, sep='\t')

    # cast ids from floats to strings
    for column_id in ['TARGET_ID', 'SOURCE_ID', 'TARGET_GRADEID', 'SOURCE_GRADEID']:
        positive_pairs[column_id] = positive_pairs[column_id].astype(str)
        if column_id in {'TARGET_GRADEID', 'SOURCE_GRADEID'}:
            positive_pairs[column_id] = positive_pairs[column_id].map(lambda x: x.strip('.0'))

    # find a negative example for each target query from the remaining queries (not in target curriculum, not in positive set and not the search query)
    negative_queries = []
    random.seed(0)
    queries = [(cur, query) for cur, query in
               zip(positive_pairs['TARGET_CURRICULUM'].tolist(), positive_pairs['TARGET_ID'].tolist())]
    for group_name, group in positive_pairs.groupby(['TARGET_CURRICULUM', 'TARGET_ID']):
        for i, row in group.iterrows():
            neg = random.choice(queries)
            while neg[0] != group_name[0] and neg[1] not in group['SOURCE_ID'].tolist() and neg[1] != \
                    group['TARGET_ID'].tolist()[0]:
                neg = random.choice(queries)
            negative_queries.append(neg[1])
    assert len(negative_queries) == len(
        positive_pairs), f'Not the same number of negative and positive examples. {len(negative_queries)} negative examples, {len(positive_pairs)} positive examples'

    target_queries = positive_pairs['TARGET_ID'].tolist()
    positive_queries = positive_pairs['SOURCE_ID'].tolist()
    with open(TRIPLETS_FILE, 'w') as out:
        out.write('qid' + '\t' + 'pos_id' + '\t' + 'neg_id' + '\n')
        for i in range(len(target_queries)):
            out.write(f'{target_queries[i]}\t{positive_queries[i]}\t{negative_queries[i]}\n')
    queries = target_queries + positive_queries

    return set(queries)

def generate_query_file(DATA_DICT_FILE, QUERY_FILE, target_queries):

  with open(DATA_DICT_FILE) as json_file:
      data = json.load(json_file)
  rows = []
  for cur_id, cur in data.items():
    for grade_id, grade in cur['grade'].items():
      for subject_id, subject in grade['subject'].items():
        for unit_id, unit in subject['unit'].items():
          for topic_id, topic in unit['topic'].items():
            for query_id, query in topic['query'].items():
              if query_id in target_queries:
                label = query['label']
                if label == '': label = topic['label']
                rows.append({'id': query_id,
                            'label': label,
                            'doc_titles': ' '.join([doc_info['title'] for doc_info in query['docs'].values() if doc_info['pin']])})
  with open(QUERY_FILE, 'w') as out:
    out.write('id'+'\t'+'label'+'\t'+'docTitles'+'\n')
    for query_dict in rows:
      out.write(f'{query_dict["id"]}\t{query_dict["label"]}\t{query_dict["doc_titles"]}\n')