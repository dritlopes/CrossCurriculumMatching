from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from classifier import pre_processor, PairDataset, Model, input_features
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import random
import time, datetime
import numpy as np
import pandas as pd
import json
import csv
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from utils import find_age, grade_by_age
from typing import Dict, List
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import sys
import nltk

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


def generate_pairs_file(data_filepath, pairs_filepath, random_seed):

    positive_pairs = pd.read_csv(data_filepath, sep='\t', dtype={'TARGET_ID': str,
                                                          'SOURCE_ID': str,
                                                          'TARGET_GRADEID': str,
                                                          'SOURCE_GRADEID': str})

    positive_pairs = positive_pairs.drop_duplicates(['TARGET_ID'])
    target_queries = positive_pairs['TARGET_ID'].tolist()
    positive_queries = positive_pairs['SOURCE_ID'].tolist()

    # find a negative example for each target query from the remaining queries (not in target curriculum, not in positive set and not the search query)

    negative_queries = []
    queries = [(cur, query) for cur, query in zip(positive_pairs['TARGET_CURRICULUM'].tolist(),positive_pairs['TARGET_ID'].tolist())]
    random.seed(random_seed)

    for group_name, group in positive_pairs.groupby(['TARGET_CURRICULUM','TARGET_ID']):
        for i, row in group.iterrows():
            neg = random.choice(queries)
            while neg[0] == group_name[0] or neg[1] in group['SOURCE_ID'].tolist() or neg[1] in group['TARGET_ID'].tolist():
                neg = random.choice(queries)
            negative_queries.append(neg[1])

    assert len(negative_queries) == len(positive_queries), f'Not the same number of negative and positive examples. {len(negative_queries)} negative examples, {len(positive_queries)} positive examples'

    with open(f'{pairs_filepath}', 'w', encoding="utf-8") as out:
        out.write('target_id'+'\t'+ 'source_id' + '\t' + 'label' + '\n')
        for i in range(len(target_queries)):
            out.write(f'{target_queries[i]}\t{positive_queries[i]}\t1\n')
            out.write(f'{target_queries[i]}\t{negative_queries[i]}\t0\n')


def generate_info_file(data_dict_filepath, info_filepath, queries, doc_sums, age_to_grade):

    with open(data_dict_filepath) as json_file:
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
                                              'query_term': label,
                                              'topic': topic['label'],
                                              'subject': subject['label'],
                                              'grade': subject['label'],
                                              'doc_titles': ' '.join(
                                                  [doc_info['title'] for doc_info in query['docs'].values() if
                                                   doc_info['pin']]),
                                              'doc_sums_1sent': '',
                                              'doc_sums_nsent': ''}

                                age = find_age(age_to_grade, cur['label'], grade['label'])
                                query_dict['age'] = age

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

    with open(info_filepath, 'w', encoding="utf-8") as out:
        headers = ['id','query_term','doc_titles','doc_sums_1sent','doc_sums_nsents','topic','subject','grade','age']
        headers = '\t'.join(headers) + '\n'
        # headers = headers.encode('UTF-8',errors='ignore')
        # headers = headers.decode('UTF-8')
        out.write(headers)
        for query_dict in rows:
            out.write(
                f'{query_dict["id"]}\t{query_dict["query_term"]}\t{query_dict["doc_titles"]}\t{query_dict["doc_sums_1sent"]}'
                f'\t{query_dict["doc_sums_nsent"]}\t{query_dict["topic"]}\t{query_dict["subject"]}\t{query_dict["grade"]}'
                f'\t{query_dict["age"]}\n')


def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss

    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def main():

    ###### PARAMETERS #######
    batch_size = 12
    bert_model = "distilbert-base-uncased"
    seed = 42
    n_epochs = 3
    age = True
    lr = 2e-5

    ###### DATA PREPARATION #######

    start_time = time.perf_counter()

    # train_inputs = {
    #     'target': ['Operations On Matrices Operations On Matrices', 'Volume as unit cubes Volume of a Combination of Solids'],
    #     'source': ['Multiplying matrices Matrix operations and use', 'Volume of 3D shapes Introduction to Perimeter'],
    #     'age': [('13', '14'), ('12', '13')],
    #     'subject': [("Math", "Science"), ("Physics", "Biology")],
    #     'y': [1, 1]}
    #
    # dev_inputs = {'target': ['Operations On Matrices Operations On Matrices', 'Volume as unit cubes Volume of a Combination of Solids'],
    #               'source': ['Multiplying matrices Matrix operations and use', 'Volume of 3D shapes Introduction to Perimeter'],
    #               'age': [('13', '14'), ('12', '13')],
    #               'subject': [("Math", "Science"), ("Physics", "Biology")],
    #               'y': [1, 1]}
    #

    DATA_DIR = f'../data'
    OUTPUT_DIR = f'../eval'
    MODEL_SAVE = '../models'

    TRAIN_FILE = f'{DATA_DIR}/train_query_pairs_13.csv'
    TRAIN_PAIR_FILE = f'{DATA_DIR}/train_pairs_13.csv'
    DEV_FILE = f'{DATA_DIR}/dev_query_pairs_13.csv'
    DEV_PAIR_FILE = f'{DATA_DIR}/dev_pairs_13.csv'
    DOC_SUMS = f'{DATA_DIR}/doc_sums.csv'
    DATA_DICT = f'{DATA_DIR}/data_dict.json'
    INFO_FILEPATH = f'{DATA_DIR}/info_queries.csv'
    curriculums = 'ICSE,CBSE,Cambridge,English,CCSS,NGSS,Scotland'
    val_eval_filepath = f'{OUTPUT_DIR}/val_train_classifier_13.csv'
    model_save_filepath = f'{MODEL_SAVE}'

    generate_pairs_file(TRAIN_FILE,TRAIN_PAIR_FILE,13)
    generate_pairs_file(DEV_FILE,DEV_PAIR_FILE,13)

    with open(TRAIN_PAIR_FILE) as f:
      train_pairs: List[Dict[str, str]] = [line for line in csv.DictReader(f,delimiter='\t')]

    with open(DEV_PAIR_FILE) as f:
      dev_pairs: List[Dict[str, str]] = [line for line in csv.DictReader(f,delimiter='\t')]

    all_queries = set()
    for pair_list in [train_pairs, dev_pairs]:
        for pair_dict in pair_list:
            all_queries.add(pair_dict['target_id'])
            all_queries.add(pair_dict['source_id'])

    doc_sums_df = pd.read_csv(DOC_SUMS, sep='\t', dtype={'queryId': str})
    doc_sums = defaultdict(list)
    for query_id, docs in doc_sums_df.groupby(['queryId']):
        doc_sums[query_id] = [doc_sum for doc_sum in list(docs['sumText'])]

    age_to_grade = grade_by_age(curriculums, f'{DATA_DIR}/MASTER Reading levels and age filter settings (pwd 123456).xlsx')

    generate_info_file(DATA_DICT, INFO_FILEPATH, all_queries, doc_sums, age_to_grade)

    with open(INFO_FILEPATH) as infile:
        info = [line for line in csv.DictReader(infile, delimiter='\t')]

    # index = int(len(train_pairs)/20)
    # train_inputs = input_features(info,train_pairs[len(train_pairs)-index:])
    # index = int(len(dev_pairs)/20)
    # dev_inputs = input_features(info,dev_pairs[len(dev_pairs)-index:])
    train_inputs = input_features(info,train_pairs)
    dev_inputs = input_features(info,dev_pairs)

    # for key, item in train_inputs.items():
    #     print(key, item[0])
    #     print(key, item[1])
    # for key, item in dev_inputs.items():
    #     print(key, item[0])
    #     print(key, item[1])

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    dataset = PairDataset(train_inputs,pre_processor,tokenizer)
    dev_dataset = PairDataset(dev_inputs,pre_processor,tokenizer)
    print(f'Length of training data: {len(dataset)}\n'
          f'Length of dev data: {len(dev_dataset)}\n')
    # print(dataset[0])
    # print(dataset[1])

    dim_age = 0
    if age:
        dim_age = dataset[0]['age'].size()[0]

    loader = DataLoader(dataset,sampler=RandomSampler(dataset),batch_size=batch_size)
    val_loader = DataLoader(dev_dataset,sampler=SequentialSampler(dev_dataset),batch_size=batch_size)

    ######## TRAINING ########

    # source: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

    # seed to make it reproducible
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # list to store training stats
    training_stats = []
    # measure training time
    total_t0 = time.time()

    model = Model(bert_model,
                  age=True,
                  dim_age=dim_age,
                  subject=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(),lr = lr)
    total_steps = len(loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    for epoch_i in range(0, n_epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, n_epochs))
        print('Training...')

        # since print statements not in slurm output
        # warnings.warn('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, n_epochs))
        # warnings.warn('Training...')

        # measure epoch time
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(loader):

            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(loader), elapsed))
                # warnings.warn('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(loader), elapsed))

            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["att_masks"].to(device)
            b_age = batch["age"].to(device)
            b_subj_ids = batch["sbj_input_ids"].to(device)
            b_subj_att = batch["sbj_att_masks"].to(device)
            b_labels = batch["y"].to(device)

            model.zero_grad()

            loss, logits = model.forward(b_input_ids,
                                         b_input_mask,
                                         b_age,
                                         b_subj_ids,
                                         b_subj_att,
                                         b_labels)

            total_train_loss += loss.item()

            loss.backward()

            # help prevent the "exploding gradients" problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # update parameters and take a step using the computed gradient
            optimizer.step()
            # update learning rate
            scheduler.step()

        # calculate the average loss over all of the batches
        avg_train_loss = total_train_loss / len(loader)

        # epoch training time
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        # warnings.warn("\n  Average training loss: {0:.2f}".format(avg_train_loss))
        # warnings.warn("  Training epoch took: {:}".format(training_time))

        #### VALIDATION ####

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in val_loader:
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["att_masks"].to(device)
            b_age = batch["age"].to(device)
            b_subj_ids = batch["sbj_input_ids"].to(device)
            b_subj_att = batch["sbj_att_masks"].to(device)
            b_labels = batch["y"].to(device)

            with torch.no_grad():

                loss, logits = model.forward(b_input_ids,
                                             b_input_mask,
                                             b_age,
                                             b_subj_ids,
                                             b_subj_att,
                                             b_labels)

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(val_loader)
        avg_val_loss = total_eval_loss / len(val_loader)
        validation_time = format_time(time.time() - t0)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        # warnings.warn("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        # warnings.warn("  Validation Loss: {0:.2f}".format(avg_val_loss))
        # warnings.warn("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': round(avg_train_loss,3),
                'Valid. Loss': round(avg_val_loss,3),
                'Valid. Accur.': round(avg_val_accuracy,3),
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    # warnings.warn('\n Training complete!')

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    # warnings.warn("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    if not os.path.exists(model_save_filepath):
        os.makedirs(model_save_filepath)
    torch.save(model.state_dict(), model_save_filepath + '/model_weights.pth')

    # # check if reloading for testing is working
    # model = Model(age=True,
    #               subject=True,
    #               dim_age=dim_age,
    #               dim_subj=dim_subj,
    #               dim_emb=dim_emb)
    # model.load_state_dict(torch.load(model_save_filepath + 'model_weights.pth'))
    # model.eval()
    # test_inputs = {
    #     'target': ['Operations On Matrices Operations On Matrices', 'Volume as unit cubes Volume of a Combination of Solids'],
    #     'source': ['Multiplying matrices Matrix operations and use', 'Volume of 3D shapes Introduction to Perimeter'],
    #     'age': ['13 14', '12 13'],
    #     'subject': ["Math Science", "Physics Biology"],
    #     'y': [1, 1]}
    # tokenizer = BertTokenizer.from_pretrained(bert_model)
    # dataset = PairDataset(test_inputs, pre_processor,tokenizer)
    # dim_age, dim_subj = 0,0
    # if age:
    #     dim_age = dataset[0]['age'].size()[0]
    #     # print(dim_age)
    # if subject:
    #     dim_subj = dataset[0]['subject'].size()[0]
    #     # print(dim_subj)
    # loader = DataLoader(dataset,sampler=RandomSampler(dataset),batch_size=batch_size)
    # predictions, true_labels, total_eval_accuracy = [], [], []
    # for batch in loader:
    #     b_input_ids = batch["bert_input_ids"].to(device)
    #     b_input_mask = batch["bert_att_masks"].to(device)
    #     b_type_ids = batch["bert_type_ids"].to(device)
    #     b_age = batch["age"].to(device)
    #     b_subj = batch["subject"].to(device)
    #     b_labels = torch.tensor(batch["y"]).to(device)
    #     with torch.no_grad():
    #         logits = model(b_input_ids,
    #                         b_input_mask,
    #                         b_type_ids,
    #                         b_age,
    #                         b_subj)
    #     logits = logits.detach().cpu().numpy()
    #     label_ids = b_labels.to('cpu').numpy()
    #     total_eval_accuracy += flat_accuracy(logits, label_ids)
    #     predictions.append(logits)
    #     true_labels.append(label_ids)
    # print(predictions)
    # print(true_labels)
    # print(total_eval_accuracy)
    # print('    DONE.')


    # saving out training stats
    df_stats = pd.DataFrame(data=training_stats)
    df_stats.to_csv(val_eval_filepath, sep='\t', index=False)

    end_time = time.perf_counter()
    print(f"TRAINING with {len(train_inputs['y'])} instances, {n_epochs} epochs and batch size {batch_size} took {end_time - start_time:0.4f} seconds")
    # warnings.warn(f"TRAINING with {len(train_inputs['y'])} instances, {n_epochs} epochs and batch size {batch_size} took {end_time - start_time:0.4f} seconds")

    # plotting training stats
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3])
    plt.savefig(f'{OUTPUT_DIR}/train_val_loss_13.png')
    # plt.show()

if __name__ == '__main__':
    main()
