import numpy as np

from consts import NONE, PAD,UNK,CLS,SEP
import json


def gen_vec():

    f = open('data/100.utf8', "r", encoding='utf-8')
    contents = f.readlines()
    f.close()

    embedding_size = 100

    # id2word = {}
    # vec_dict = {}

    word2id={UNK:0,PAD:1,CLS:2,SEP:3}
    word_size=len(contents)+len(word2id)

    vec = np.random.standard_normal((word_size, embedding_size))

    for content in contents:
        content = content.strip().split()
        values=content[1:]
        for j in range(embedding_size):
            vec[len(word2id)][j] = (float)(values[j])
        word2id[content[0].strip()]=len(word2id)

    print(len(word2id))#400004
    id2word={id:word for word,id in word2id.items()}

    with open('data/100.utf8.word2id.json','w',encoding='utf-8') as fw:
        json.dump(word2id,fw)
    with open('data/100.utf8.id2word.json','w',encoding='utf-8') as fi:
        json.dump(id2word,fi)

    np.save('data/100.utf8.npy', vec)

def build_vocab(labels, BIO_tagging=True,padding=True):
    if padding:
        all_labels = [PAD, NONE]
    else:
        all_labels = [NONE]
    # all_labels = [NONE]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label


def calc_metric(y_true, y_pred):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


def find_triggers(labels,sen_event=None):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result = []
    # labels = [label.split('-') for label in labels]

    # for i in range(len(labels)):
    #     if labels[i][0] == 'B':
    #         result.append([i, i + 1, labels[i][1]])
    #
    # for item in result:
    #     j = item[1]
    #     while j < len(labels):
    #         if labels[j][0] == 'I':
    #             j = j + 1
    #             item[1] = j
    #         else:
    #             break

    flag=[]
    for i in range(len(labels)):
        if i in flag:
            continue
        if labels[i]!='O' :
            index=i
            j=i+1
            while j<len(labels) and labels[j]==labels[i]:
                flag.append(j)
                j+=1
            result.append([index, j, labels[i]])
            # if sen_event ==None:
            #     result.append([index,j,labels[i]])
            # else:
            #     if labels[i] in sen_event:
            #         result.append([index, j, labels[i]])

    return [tuple(item) for item in result]


if __name__ == '__main__':
    gen_vec()

# To watch performance comfortably on a telegram when training for a long time
def report_to_telegram(text, bot_token, chat_id):
    try:
        import requests
        requests.get('https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(bot_token, chat_id, text))
    except Exception as e:
        print(e)
