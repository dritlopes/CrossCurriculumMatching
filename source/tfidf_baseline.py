import requests
from collections import defaultdict

def match_query (q, curriculums, qualifiers):

    q = q.replace(' ', '%20')
    url = f'https://api.wizenoze.com/v4/curriculum/curator/curriculum/search?q={q}&qualifiers={qualifiers}&curriculums={curriculums}'
    headers = {'Authorization': 'cb880a2e-e227-4e80-82e3-c510136cd0cd',
               'Content-Type': 'application/json',
               'charset': 'utf8'}
    response = requests.get(url, headers=headers)

    return response

def tfidf_match (target_cur, source_curriculums, target_queries, k):

    matches = defaultdict(dict)
    curriculums = source_curriculums.replace(target_cur+',', '')
    qualifiers = 'QUERY'
    target = [(target_id, target_label, target_path) for target_id, target_label, target_path in zip(target_queries['TARGET_ID'], target_queries['TARGET'], target_queries['TARGET_PATH'])]
    for tq in target:
        matches[tq[0]] = {'label': tq[1].strip(),
                          'path': tq[2].strip()}
        response = match_query(tq[1],curriculums,qualifiers)
        if response.status_code == 200:
            resp = response.json()
            scores = []
            for match in resp:
                if match['queryId'] != tq[0] and match['curriculumMetadata']['CURRICULUM']['name'] != target_cur and match['curriculumMetadata']['CURRICULUM']['name'] in curriculums and len(scores) < k:
                    scores.append((match['name'].strip(),
                                   match['queryId'],
                                   match['curriculumMetadata']['CURRICULUM']['name'] + '>' + match['path'].strip(),
                                   1))
            matches[tq[0]]['scores'] = scores
        else:
            print(f'{response.status_code} response status for query {tq[0]}')
            matches[tq[0]]['scores'] = [('','','','')]

    return matches