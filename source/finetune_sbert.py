import csv
import os
from typing import Iterable, Dict, List, Set
from typing import Tuple
from sentence_transformers import InputExample
from torch.utils.data import IterableDataset
from collections import defaultdict
from sentence_transformers import models, SentenceTransformer, evaluation, losses
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import argparse


class QueryLinks:

    def __init__(self, input: Iterable[Dict]):
        self.queries: Dict[str, str] = dict()
        self.docs: Dict[str, str] = dict()

        for line in input:
            query_id = str(line["id"])
            query_term = str(line["label"])
            doc_titles = str(line["docTitles"])
            doc_sums_1sent = str(line["docSums1sent"])
            doc_sums_nsent = str(line["docSumsNsent"])

            self.queries[query_id] = query_term
            self.docs[query_id] = query_term
            # self.docs[query_id] = query_term + ' ' + doc_titles
            # self.docs[query_id] = query_term + ' ' + doc_titles + ' ' + doc_sums_1sent
            # self.docs[query_id] = query_term + ' ' + doc_titles + ' ' + doc_sums_nsent
            # self.docs[query_id] = query_term + doc_sums_nsent
            # self.docs[query_id] = query_term + doc_sums_1sent

    def get_document(self, doc_id: str) -> str:
        return self.docs[doc_id]

    def get_query(self, query_id: str) -> str:
        return self.queries[query_id]

    @classmethod
    def from_file(cls, input_file: str):
        with open(input_file) as f:
            return cls(input=csv.DictReader(f, delimiter='\t'))


class TripletsDataset(IterableDataset):

    def __init__(
            self, model, query_links: QueryLinks, triplets: List[Dict[str, str]]
    ):
        self.model = model
        self._query_links = query_links
        self._triplets = triplets

    def __iter__(self):
        yield from (
            InputExample(texts=self.triplet_to_text(triplet))
            for triplet in self._triplets
        )

    def __len__(self):
        return len(self._triplets)

    def triplet_to_text(self, triplet: Dict[str, str]) -> Tuple[str, str, str]:
        return (
            self._query_links.get_query(triplet["qid"]),
            self._query_links.get_document(triplet["pos_id"]),
            self._query_links.get_document(triplet["neg_id"])
        )


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


def main ():

    TRAIN_FILEPATH = '../data/train_query_pairs_7.csv'
    DEV_FILEPATH = '../data/dev_query_pairs_7.csv'
    TRAIN_TRIPLETS_FILE = '../data/train_triplets.csv'
    DEV_TRIPLETS_FILE = '../data/dev_triplets.csv'
    DATA_DICT_FILE = '../data/data_dict.json'
    QUERY_FILE = '../data/query_info.csv'
    MODEL_OUTPUT_PATH = '../models/paraphrase-sbert-label-rankingloss-nodup_7'
    DOC_SUMS = '../data/doc_sums.csv'

    # generate triplets
    if not os.path.isfile(TRAIN_TRIPLETS_FILE):
        generate_triplets_file(TRAIN_FILEPATH, TRAIN_TRIPLETS_FILE)
    if not os.path.isfile(DEV_TRIPLETS_FILE):
        generate_triplets_file(DEV_FILEPATH, DEV_TRIPLETS_FILE)

    with open(TRAIN_TRIPLETS_FILE) as f:
        train_triplets: List[Dict[str, str]] = [line for line in csv.DictReader(f, delimiter='\t')]
    with open(DEV_TRIPLETS_FILE) as f:
        test_triplets: List[Dict[str, str]] = [line for line in csv.DictReader(f, delimiter='\t')]

    # generate query info file
    if not os.path.isfile(QUERY_FILE):

        all_queries = set()

        for triplet_list in [train_triplets, test_triplets]:
            for triplet_dict in triplet_list:
                all_queries.add(triplet_dict['qid'])
                all_queries.add(triplet_dict['pos_id'])
                all_queries.add(triplet_dict['neg_id'])

        doc_sums_df = pd.read_csv(DOC_SUMS, sep='\t', dtype={'queryId': str})
        doc_sums = defaultdict(list)
        for query_id, docs in doc_sums_df.groupby(['queryId']):
            doc_sums[query_id] = [doc_sum for doc_sum in list(docs['sumText'])]

        generate_query_file(DATA_DICT_FILE, QUERY_FILE, all_queries, doc_sums)

    # generate query links
    query_links = QueryLinks.from_file(QUERY_FILE)

    # generate triplets for val set
    test_corpus: Dict[str, str] = {}
    test_queries: Dict[str, str] = {}
    test_rel_docs: Dict[str, Set[str]] = defaultdict(set)

    for triplet in test_triplets:
        qid = triplet["qid"]
        pos_id = triplet["pos_id"]
        neg_id = triplet["neg_id"]

        query_term = query_links.get_query(triplet["qid"])
        pos_doc = query_links.get_document(triplet["pos_id"])
        neg_doc = query_links.get_document(triplet["neg_id"])

        test_queries[qid] = query_term
        test_corpus[pos_id] = pos_doc
        test_corpus[neg_id] = neg_doc

        test_rel_docs[qid].add(pos_id)

    _query_ids = []
    for qid in test_queries:
        if qid in test_rel_docs and len(test_rel_docs[qid]) > 0:
            _query_ids.append(qid)

    # model training
    BASE_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    # word_embedding_model = models.Transformer(BASE_MODEL, max_seq_length=350)
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model = SentenceTransformer(BASE_MODEL)
    # model.max_seq_length = 200

    # check how long the source queries are with doc titles and sums
    # tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # source_queries = []
    # for triplet in train_triplets:
    #     pos_doc = query_links.get_document(triplet["pos_id"])
    #     neg_doc = query_links.get_document(triplet["neg_id"])
    #     source_queries.append(pos_doc)
    #     source_queries.append(neg_doc)
    # encoded_input = tokenizer(source_queries, padding=False, truncation=False)
    # longer = [len(encoded) for encoded in encoded_input['input_ids'] if len(encoded) > 128]
    # print(len(source_queries), len(longer), sum(longer)/len(longer))
    # lengths = [len(encoded) for encoded in encoded_input['input_ids']]
    # print(sum(lengths)/len(lengths), np.std(np.array(lengths)))
    # exit()

    evaluator = evaluation.InformationRetrievalEvaluator(
        test_queries, test_corpus, test_rel_docs)

    TRAIN_BATCH_SIZE: int = 12

    train_dataset = TripletsDataset(
        model=model, query_links=query_links, triplets=train_triplets
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, batch_size=TRAIN_BATCH_SIZE
    )

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=3,
        warmup_steps=100,
        output_path=MODEL_OUTPUT_PATH,
        evaluation_steps=500,
        use_amp=True
    )

if __name__ == '__main__':
    main()
