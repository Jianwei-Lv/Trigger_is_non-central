import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import numpy as np
from model import Net

from data_load import ACE2005Dataset, pad, all_triggers, all_entities, all_postags, all_arguments, tokenizer,glove_weight
from utils import report_to_telegram
from eval import eval


def train(model, iterator, optimizer, criterion,epoch,test_iter,fname,trigger_F1,argument_F1):
    model.train()
    for i, batch in enumerate(iterator):
        tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d ,glove_id_2d= batch
        optimizer.zero_grad()
        trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.module.predict_triggers(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                      postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                                                                                                                      triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d,glove_id_2d=glove_id_2d ,epoch=epoch)

        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))

        if len(argument_keys) > 0:
            argument_hidden_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.predict_arguments(argument_hidden, argument_keys, arguments_2d)
            # argument_loss = criterion(argument_logits, arguments_y_1d)
            argument_loss = criterion(argument_hidden_logits, arguments_y_1d)


            loss = trigger_loss +  argument_loss
            # if i == 0:
            #     print("=====sanity check for arguments======")
            #     print('arguments_y_1d:', arguments_y_1d)
            #     print("arguments_2d[0]:", arguments_2d[0]['events'])
            #     print("argument_hat_2d[0]:", argument_hat_2d[0]['events'])
            #     print("=======================")
        else:
            loss = trigger_loss

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        loss.backward()

        optimizer.step()

        # if i == 0:
        #     print("=====sanity check======")
        #     print("tokens_x_2d[0]:", tokenizer.convert_ids_to_tokens(tokens_x_2d[0])[:seqlens_1d[0]])
        #     print("entities_x_3d[0]:", entities_x_3d[0][:seqlens_1d[0]])
        #     print("postags_x_2d[0]:", postags_x_2d[0][:seqlens_1d[0]])
        #     print("head_indexes_2d[0]:", head_indexes_2d[0][:seqlens_1d[0]])
        #     print("triggers_2d[0]:", triggers_2d[0])
        #     print("triggers_y_2d[0]:", triggers_y_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
        #     print('trigger_hat_2d[0]:', trigger_hat_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
        #     print("seqlens_1d[0]:", seqlens_1d[0])
        #     print("arguments_2d[0]:", arguments_2d[0])
        #     print("=======================")

        if i % 300 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))
            metric_test, trigger_f1, argument_f1 = eval(model, test_iter, fname)
            if trigger_F1 < trigger_f1:
                trigger_F1 = trigger_f1
                torch.save(model, "latest_model.pt")
            if argument_F1 < argument_f1:
                argument_F1 = argument_f1
                torch.save(model, "argument_latest_model.pt")
            print('best trigger F1:')
            print(trigger_F1)
            print('best argument F1:')
            print(argument_F1)
            model.train()
    return trigger_F1, argument_F1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument("--trainset", type=str, default="data/train.json")
    parser.add_argument("--devset", type=str, default="data/dev.json")
    parser.add_argument("--testset", type=str, default="data/test.json")

    parser.add_argument("--telegram_bot_token", type=str, default="")
    parser.add_argument("--telegram_chat_id", type=str, default="")

    hp = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    np.random.seed(200)
    torch.manual_seed(200)
    torch.cuda.manual_seed_all(200)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    model = Net(
        device=device,
        trigger_size=len(all_triggers),
        entity_size=len(all_entities),
        all_postags=len(all_postags),
        argument_size=len(all_arguments),
        glove_weight=glove_weight
    )
    if device == 'cuda:0':
        print('ssssssss')
        model = model.cuda(0)

    model = nn.DataParallel(model)
    model = torch.load("argument_latest_model.pt")

    # train_dataset = ACE2005Dataset(hp.trainset)
    # dev_dataset = ACE2005Dataset(hp.devset)
    test_dataset = ACE2005Dataset(hp.testset,n_event=2,n_argument=5)

    # samples_weight = train_dataset.get_samples_weight()
    # sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    #
    # train_iter = data.DataLoader(dataset=train_dataset,
    #                              batch_size=hp.batch_size,
    #                              shuffle=False,
    #                              # sampler=sampler,
    #                              num_workers=4,
    #                              collate_fn=pad)
    # dev_iter = data.DataLoader(dataset=dev_dataset,
    #                            batch_size=hp.batch_size,
    #                            shuffle=False,
    #                            num_workers=4,
    #                            collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    trigger_F1=0
    argument_F1=0

    fname = os.path.join(hp.logdir, str(1))
    print(f"=========eval test at epoch={1}=========")
    metric_test, trigger_f1, argument_f1 = eval(model, test_iter, fname + '_test')

    # for epoch in range(1, hp.n_epochs + 1):
    #     fname = os.path.join(hp.logdir, str(epoch))
    #     trigger_F1, argument_F1 =train(model, train_iter, optimizer, criterion,epoch,test_iter,fname+'test',trigger_F1,argument_F1)
    #
    #     fname = os.path.join(hp.logdir, str(epoch))
    #     print(f"=========eval dev at epoch={epoch}=========")
    #     metric_dev ,_,_= eval(model, dev_iter, fname + '_dev')
    #
    #     print(f"=========eval test at epoch={epoch}=========")
    #     metric_test ,trigger_f1,argument_f1= eval(model, test_iter, fname + '_test')
    #
    #     if hp.telegram_bot_token:
    #         report_to_telegram('[epoch {}] dev\n{}'.format(epoch, metric_dev), hp.telegram_bot_token, hp.telegram_chat_id)
    #         report_to_telegram('[epoch {}] test\n{}'.format(epoch, metric_test), hp.telegram_bot_token, hp.telegram_chat_id)
    #
    #     if trigger_F1<trigger_f1:
    #         trigger_F1=trigger_f1
    #         torch.save(model, "latest_model.pt")
    #     if argument_F1<argument_f1:
    #         argument_F1=argument_f1
    #         torch.save(model, "argument_latest_model.pt")
    #     print('best trigger F1:')
    #     print(trigger_F1)
    #     print('best argument F1:')
    #     print(argument_F1)
    #     # print(f"weights were saved to {fname}.pt")
