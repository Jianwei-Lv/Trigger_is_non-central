import numpy as np
import torch
from torch.utils import data
import json

from consts import NONE, PAD, CLS, SEP, UNK, TRIGGERS, ARGUMENTS, ENTITIES, POSTAGS
from utils import build_vocab
from pytorch_pretrained_bert import BertTokenizer

# init vocab
all_events, events2idx, idx2events = build_vocab(TRIGGERS,BIO_tagging=False,padding=True)
# all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)

glove_weight=np.load('data/100.utf8.npy')
with open('data/100.utf8.word2id.json','r',encoding='utf-8') as fw:
    word2id=json.load(fw)
with open('data/100.utf8.id2word.json','r',encoding='utf-8') as fi:
    id2word=json.load(fi)


tokenizer = BertTokenizer.from_pretrained('bert-base-cased/', do_lower_case=False, never_split=(PAD, CLS, SEP, UNK))

event_type_dict={}
class ACE2005Dataset(data.Dataset):
    def __init__(self, fpath,type=None):
        self.sent_li, self.entities_li, self.postags_li, self.triggers_li, self.arguments_li,self.sent_trigger_li = [], [], [], [], [],[]
        self.cut_off=60
        count=0
        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item['words'][:self.cut_off]
                entities = [[NONE] for _ in range(len(words))][:self.cut_off]
                triggers = [NONE] * len(words)
                postags = item['pos-tags'][:self.cut_off]
                sent_trigger_li = [events2idx[PAD]] * len(all_events)
                arguments = {
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }
                try:
                    for entity_mention in item['golden-entity-mentions']:
                        start = entity_mention['start']
                        if start >= self.cut_off:
                            continue
                        end = min(entity_mention["end"], self.cut_off)
                        arguments['candidates'].append((start, end, entity_mention['entity-type']))

                        for i in range(start, end):
                            entity_type = entity_mention['entity-type']
                            if i == start:
                                entity_type = 'B-{}'.format(entity_type)
                            else:
                                entity_type = 'I-{}'.format(entity_type)
                            # try:
                            if len(entities[i]) == 1 and entities[i][0] == NONE:
                                entities[i][0] = entity_type
                            else:
                                entities[i].append(entity_type)

                    for event_mention in item['golden-event-mentions']:
                        if event_mention['trigger']['start'] >= self.cut_off:
                            continue
                        trigger_type = event_mention['event_type']

                        if trigger_type in event_type_dict:
                            event_type_dict[trigger_type]+=1
                        else:
                            event_type_dict[trigger_type]=1

                        for i in range(event_mention['trigger']['start'], min(event_mention['trigger']['end'],self.cut_off)):

                            sent_trigger_li[events2idx[trigger_type]] = 1
                            triggers[i] =trigger_type
                            # if i == event_mention['trigger']['start']:
                            #     triggers[i] = 'B-{}'.format(trigger_type)
                            # else:
                            #     triggers[i] = 'I-{}'.format(trigger_type)

                        event_key = (event_mention['trigger']['start'], min(event_mention['trigger']['end'],self.cut_off), event_mention['event_type'])
                        arguments['events'][event_key] = []
                        for argument in event_mention['arguments']:
                            if argument['start']>=self.cut_off:
                                continue
                            role = argument['role']
                            if role.startswith('Time'):
                                role = role.split('-')[0]
                            arguments['events'][event_key].append((argument['start'], min(argument['end'],self.cut_off), argument2idx[role]))
                except:
                    continue
                if len(words)>4:
                    if type=='train':
                        if 1 in sent_trigger_li:
                            self.sent_li.append([CLS] + words + [SEP]+['there','exits','event']+[SEP])
                            self.entities_li.append([[PAD]] + entities + [[PAD]]*5)
                            self.postags_li.append([PAD] + postags + [PAD]*5)
                            self.triggers_li.append(triggers+['O']*3)
                            self.arguments_li.append(arguments)
                            self.sent_trigger_li.append(sent_trigger_li)
                        else:
                            self.sent_li.append([CLS] + words + [SEP]+['there','does','not','exits','event']+[SEP])
                            self.entities_li.append([[PAD]] + entities + [[PAD]]*7)
                            self.postags_li.append([PAD] + postags + [PAD]*7)
                            self.triggers_li.append(triggers+['O']*5)
                            self.arguments_li.append(arguments)
                            self.sent_trigger_li.append(sent_trigger_li)
                    else:
                        if 1 in sent_trigger_li:
                            self.sent_li.append([CLS] + words + [SEP])
                            self.entities_li.append([[PAD]] + entities + [[PAD]] )
                            self.postags_li.append([PAD] + postags + [PAD] )
                            self.triggers_li.append(triggers )
                            self.arguments_li.append(arguments)
                            self.sent_trigger_li.append(sent_trigger_li)
                    # else:
                    # self.sent_li.append([CLS] + words + [SEP])
                    # self.entities_li.append([[PAD]] + entities + [[PAD]])
                    # self.postags_li.append([PAD] + postags + [PAD])
                    # self.triggers_li.append(triggers)
                    # self.arguments_li.append(arguments)
                        # for key,value in arguments['events'].items():
                        #     if len(value)==n_array:
                        #         count+=1
                        #         self.sent_li.append([CLS] + words + [SEP])
                        #         self.entities_li.append([[PAD]] + entities + [[PAD]])
                        #         self.postags_li.append([PAD] + postags + [PAD])
                        #         self.triggers_li.append(triggers)
                        #         self.arguments_li.append(arguments)
                    # elif n_array==2:
                    #     for key,value in arguments['events'].items():
                    #         if len(value)==n_array:
                    #             count += 1
                    #             self.sent_li.append([CLS] + words + [SEP])
                    #             self.entities_li.append([[PAD]] + entities + [[PAD]])
                    #             self.postags_li.append([PAD] + postags + [PAD])
                    #             self.triggers_li.append(triggers)
                    #             self.arguments_li.append(arguments)
                    # elif n_array==3:
                    #     for key,value in arguments['events'].items():
                    #         if len(value)==n_array:
                    #             count += 1
                    #             self.sent_li.append([CLS] + words + [SEP])
                    #             self.entities_li.append([[PAD]] + entities + [[PAD]])
                    #             self.postags_li.append([PAD] + postags + [PAD])
                    #             self.triggers_li.append(triggers)
                    #             self.arguments_li.append(arguments)
                    # else:
                    #     for key,value in arguments['events'].items():
                    #         if len(value)>=n_array:
                    #             count += 1
                    #             self.sent_li.append([CLS] + words + [SEP])
                    #             self.entities_li.append([[PAD]] + entities + [[PAD]])
                    #             self.postags_li.append([PAD] + postags + [PAD])
                    #             self.triggers_li.append(triggers)
                    #             self.arguments_li.append(arguments)
        print(event_type_dict)
        # print('sss')
    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, entities, postags, triggers, arguments,sent_trigger = self.sent_li[idx], self.entities_li[idx], self.postags_li[idx], self.triggers_li[idx], self.arguments_li[idx],self.sent_trigger_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x, postags_x, is_heads ,glove_id= [], [], [], [],[]
        for w, e, p in zip(words, entities, postags):

            if w in [CLS,SEP]:
                id=word2id[w]
            else:
                id=word2id[w.lower()] if w.lower() in word2id else word2id[UNK]
            glove_id.append(id)


            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)
                if len(tokens)>1:
                    for t in tokens[1:]:
                        id = word2id[t.lower()] if t.lower() in word2id else word2id[UNK]
                        glove_id.append(id)

            p = [p] + [PAD] * (len(tokens) - 1)
            e = [e] + [[PAD]] * (len(tokens) - 1)  # <PAD>: no decision
            p = [postag2idx[postag] for postag in p]
            e = [[entity2idx[entity] for entity in entities] for entities in e]

            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head)

        triggers_y = [events2idx[t] for t in triggers]
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, postags_x, triggers_y, arguments, seqlen, head_indexes, words, triggers,glove_id,sent_trigger

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    event_type = trigger
                    not_none = True
                    break
            if not_none:
                if 0 <= event_type_dict[event_type] <= 10:
                    samples_weight.append(10.0)
                elif 11 <= event_type_dict[event_type] <= 30:
                    samples_weight.append(8.0)
                elif 31 <= event_type_dict[event_type] <= 50:
                    samples_weight.append(7.0)
                elif 51 <= event_type_dict[event_type] <= 100:
                    samples_weight.append(4.0)
                elif 101 <= event_type_dict[event_type] <= 500:
                    samples_weight.append(3.0)
                else:
                    samples_weight.append(1.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


def pad(batch):
    tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d ,glove_id_2d,sent_trigger_y_2d= list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()
    position_2d=[]
    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        postags_x_2d[i] = postags_x_2d[i] + [0] * (maxlen - len(postags_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [events2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_x_3d[i]))]
        glove_id_2d[i] = glove_id_2d[i] + [word2id[PAD]] * (maxlen - len(glove_id_2d[i]))
        position=[j for j  in range(maxlen)]
        position_2d.append(position)

    return tokens_x_2d, entities_x_3d, postags_x_2d, \
           triggers_y_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, \
           words_2d, triggers_2d,glove_id_2d,sent_trigger_y_2d,position_2d
