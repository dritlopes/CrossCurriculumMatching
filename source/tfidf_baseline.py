import requests
from collections import defaultdict

def match_query (q, curriculums, qualifiers):

    # q = q.replace(' ', '%20')
    url = f'https://api.wizenoze.com/v2/curriculum/curator/curriculum/search?q=${q}&qualifiers=${qualifiers}' \
          f'&curriculumFilters=${curriculums}'
    headers = {'Authorization': 'cb880a2e-e227-4e80-82e3-c510136cd0cd'}
               #'Content-Type': 'application/json'}
    response = requests.get(url, headers=headers)

    return response

def tfidf_match (target_cur, source_curriculums, target_queries):

    matches = defaultdict(dict)
    curriculums = source_curriculums.replace(target_cur, '')
    qualifiers = 'QUERY'
    target = [(target_id, target_label) for target_id, target_label in zip(target_queries['TARGET_ID'], target_queries['TARGET'])]
    for tq in target:
        response = match_query(tq[1],curriculums,qualifiers)
        if response.status_code == 200:
            resp = response.json()
            print(resp)
        else:
            print(f'{response.status_code} response status for query {tq[0]}')
            continue
        # TODO save only the first 5 results ordered according to sim score
    exit()

    return matches