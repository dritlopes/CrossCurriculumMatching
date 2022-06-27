import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel
import torch
from torch.utils.data import Dataset


def pre_processor(texts, tokenizer):

    inputs = tokenizer(texts[0],
                       texts[1],
                       padding='max_length',
                       truncation='only_second',
                       # max_length=,
                       return_tensors='pt')
                       # return_overflowing_tokens=True)
    # print(inputs['input_ids'][0])
    # print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

    return inputs

def clean_text(text):

    if type(text) == str:

        text = text.strip()
        text = text.lower()

    else:
        text = ''

    return text

def input_features(info, pairs):

    info_dict = dict()
    empty = 0
    for line in info:

        query = clean_text(line['query_term'])
        topic = clean_text(line['topic'])
        doc_titles = clean_text(line['doc_titles'])
        doc_sums_1sent = clean_text(line['doc_sums_1sent'])
        doc_sums_nsent = clean_text(line['doc_sums_nsents'])
        subj = clean_text(line['subject'])

        info_dict[line['id']] = {'query_term': query,
                                 'topic': topic,
                                 'subject': subj,
                                 'grade': line['grade'],
                                 'age': line['age'],
                                 'doc_titles': doc_titles,
                                 'doc_sums_1sent': doc_sums_1sent,
                                 'doc_sums_nsent': doc_sums_nsent}

        if '' in [query,topic,doc_titles,doc_sums_1sent,doc_sums_nsent,subj]: empty += 1

    # print(empty)

    source, target, age, subject, labels = [], [], [], [], []
    for line in pairs:
        target.append(f"{info_dict[line['target_id']]['query_term']} {info_dict[line['target_id']]['topic']}")
        source.append(f"{info_dict[line['source_id']]['query_term']} {info_dict[line['source_id']]['topic']} "
                      f"{info_dict[line['source_id']]['doc_titles']}")
        age.append((str(info_dict[line['target_id']]['age']),str(info_dict[line['source_id']]['age'])))
        subject.append((info_dict[line['target_id']]['subject'],info_dict[line['source_id']]['subject']))
        labels.append(int(line['label']))

    return {'target': target,
            'source': source,
            'age': age,
            'subject': subject,
            'y': labels}


def vectorizer(dimensions):

    vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    vec.fit(dimensions)
    # print(vec.get_feature_names())
    return vec


class PairDataset(Dataset):

    def __init__(self, inputs, pre_processor, tokenizer):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.bert_inputs = pre_processor((self.inputs['target'], self.inputs['source']), self.tokenizer)
        self.labels = None

        if 'age' in self.inputs.keys():
            vec = vectorizer(['4 5 6 7 8 9 10 11 12 13 14 15 16 17 18'])
            age_target = vec.transform([age[0] for age in self.inputs['age']])
            age_source = vec.transform([age[1] for age in self.inputs['age']])
            ages = []
            for t, s in zip(age_target,age_source):
                t = torch.from_numpy(t.toarray())
                s = torch.from_numpy(s.toarray())
                concat = torch.cat((t,s),1)
                ages.append(concat)
            ages = torch.stack(ages)
            ages = torch.squeeze(ages)
            self.ages = ages

        if 'subject' in self.inputs.keys():
            subject_target = [subj[0] for subj in self.inputs['subject']]
            subject_source = [subj[1] for subj in self.inputs['subject']]
            self.subj_inputs = pre_processor((subject_target, subject_source), self.tokenizer)

        if 'y' in self.inputs.keys():
            self.labels = torch.tensor(self.inputs['y'])

    def __getitem__(self, index):

        pair_dict = dict()

        pair_dict['input_ids'] = self.bert_inputs['input_ids'][index]
        pair_dict['att_masks'] = self.bert_inputs['attention_mask'][index]

        pair_dict['age'] = self.ages[index]

        pair_dict['sbj_input_ids'] = self.subj_inputs['input_ids'][index]
        pair_dict['sbj_att_masks'] = self.subj_inputs['attention_mask'][index]

        if self.labels != None:
            pair_dict['y'] = self.labels[index]

        return pair_dict

    def __len__(self):
        return len(self.inputs['target'])


class Model(nn.Module):

    def __init__(self, model='distilbert-base-uncased', input_dim = 768, hidden_dim = 256, age=False, subject=False, dim_age=30):
        super(Model, self).__init__()
        self.base_model = AutoModel.from_pretrained(model)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2) # https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
        if subject:
            input_dim = input_dim + input_dim
            self.fc1 = nn.Linear(input_dim, hidden_dim)
        if age:
            input_dim = input_dim + dim_age
            self.embedding_age = nn.Embedding(dim_age, dim_age)
            self.fc1 = nn.Linear(input_dim, hidden_dim)


    def forward(self, input_ids, att_masks, age=None, subj_ids=None, subj_att=None, labels=None):

        # getting input to classifier head
        bert_encodings = self.base_model(input_ids,attention_mask=att_masks)
        input_vectors = torch.mean(bert_encodings[0],1) # average last hidden representations across tokens

        if subj_ids != None:
            bert_encodings = self.base_model(subj_ids, attention_mask=subj_att)
            sbj_vectors = torch.mean(bert_encodings[0], 1)
            # embeddings = self.embedding_subject(subject)
            # avg_emb = torch.mean(embeddings, 1)
            # print(avg_emb.size())
            input_vectors = torch.cat((input_vectors, sbj_vectors),1)

        if age != None:
            embeddings = self.embedding_age(age)
            avg_emb = torch.mean(embeddings,2)
            input_vectors = torch.cat((input_vectors, avg_emb),1)
            # print(input_vectors.size())

        # classifier
        outputs = self.fc1(input_vectors)
        # print(outputs.size())
        outputs = self.activation(outputs)
        # print(outputs.size())
        outputs = self.dropout(outputs) # https://wandb.ai/authors/ayusht/reports/Implementing-Dropout-in-PyTorch-With-Example--VmlldzoxNTgwOTE
        # print(outputs.size())
        logits = self.fc2(outputs)
        # print(logits.size())
        # logits = self.softmax(logits)

        # loss computation
        if labels != None:
            # labels = labels.float()
            # loss_fct = nn.BCEWithLogitsLoss() # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
            loss_fct = nn.CrossEntropyLoss()
            # logits = torch.squeeze(logits)
            # print(logits)
            loss = loss_fct(logits, labels)
            # print(loss)
            # output = (logits,) + bert_encodings[2:]
            logits = torch.softmax(logits,dim=1)
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        else:
            logits = torch.softmax(logits,dim=1)

        return logits


