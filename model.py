import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import BertModel
from data_load import  argument2idx,all_events,idx2events
from consts import NONE,ENTITIES
from utils import find_triggers
from transformer.layers import EncoderLayer
from transformer.modules import PosEncoding
event_list = ['Conflict:Attack', 'Movement:Transport', 'Life:Die', 'Contact:Meet', 'Personnel:End', 'Personnel:Elect', 'Life:Injure', 'Transaction:Transfer', 'Contact:Phone', 'Justice:Trial', 'Justice:Charge', 'Transaction:Transfer', 'Personnel:Start', 'Justice:Sentence', 'Justice:Arrest', 'Justice:Sue', 'Life:Marry', 'Conflict:Demonstrate', 'Justice:Convict', 'Life:Be', 'Justice:Release', 'Business:Declare', 'Business:End', 'Justice:Appeal', 'Business:Start', 'Justice:Fine', 'Life:Divorce', 'Justice:Execute', 'Business:Merge', 'Personnel:Nominate', 'Justice:Acquit', 'Justice:Extradite', 'Justice:Pardon']

def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


class Attention_label(nn.Module):
    def __init__(self, input_dim_1,input_dim_2,output_dim):
        super(Attention_label, self).__init__()
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.output_dim=output_dim
        # self.linear_ctx = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.linear_1 = nn.Linear(self.input_dim_1, self.output_dim, bias=True)
        self.linear_2 = nn.Linear(self.input_dim_2, self.output_dim, bias=True)
        # self.v = nn.Linear(self.input_dim, 1)

    def forward(self, input_1,input_2):
        batch=input_1.shape[0]
        # output2 = self.linear_2(input_2)
        output2 = input_2.transpose(0, 1).contiguous()
        trigger_indexs=[]
        for i in range(batch):

            output1=self.linear_1(input_1[i])


            c = torch.mm(output1, output2)
            trigger_index = c.argmax(-1)
            trigger_indexs.append(trigger_index)
        trigger_indexs=torch.stack(trigger_indexs)
        return trigger_indexs
        # softmax = torch.softmax(c, 0)
        #
        # output2 = output2.transpose(0, 1).contiguous()
        # output2 = output2.repeat(seq_len, 1)
        #
        # result = output2.mul(softmax)
        #
        # # return torch.cat([input_1,result],-1)
        # return result

class Net(nn.Module):
    def __init__(self, trigger_size=None,glove_weight=None, entity_size=None, all_postags=None, postag_embedding_dim=50, argument_size=None, entity_embedding_dim=50, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        # self.bert = BertModel.from_pretrained('bert-base-cased/')
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.entity_embed = MultiLabelEmbeddingLayer(num_embeddings=entity_size, embedding_dim=entity_embedding_dim, device=device)
        # self.postag_embed = nn.Embedding(num_embeddings=all_postags, embedding_dim=postag_embedding_dim)
        # self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=768+800 , hidden_size=768 , batch_first=True).to(device)
        self.pos_embsize = 100
        glove_weight = torch.FloatTensor(glove_weight)
        self.glove_embed = nn.Embedding(num_embeddings=20138, embedding_dim=100)
        self.glove_embed = self.glove_embed.from_pretrained(embeddings=glove_weight, freeze=False)
        self.glove_embedding_size = 100

        self.attention_label=Attention_label(768+self.glove_embedding_size,800,800).to(device)
        hidden_size_cnn = 768 +self.glove_embedding_size+800
        hidden_size = 768+self.glove_embedding_size
        self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=hidden_size, hidden_size=int(hidden_size/2),
                           batch_first=True).to(device)
        self.cnn_1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2).to(device)
        self.w1=nn.Linear(hidden_size,hidden_size,bias=True)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cnn_2_1 = nn.Conv1d(hidden_size * 2, int(hidden_size*1.5), kernel_size=5, padding=2).to(device)
        self.cnn_2_2 = nn.Conv1d(int(hidden_size*1.5), hidden_size, kernel_size=7, padding=3).to(device)

        self.cnn_3 = nn.Conv1d(hidden_size*2 , hidden_size,kernel_size=1).to(device)
        #(d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        # self.transformer_encoder=EncoderLayer(d_k=hidden_size,d_v=hidden_size,d_ff=int(hidden_size/2),d_model=hidden_size,n_heads=12)
        self.layers=2
        self.transformer_layers = nn.ModuleList([EncoderLayer(d_k=hidden_size,d_v=hidden_size,d_ff=int(hidden_size/2),d_model=hidden_size,n_heads=12,dropout=0.1) for _ in range(self.layers)])
        self.transformer_layers_type = nn.ModuleList([EncoderLayer(d_k=hidden_size, d_v=hidden_size,
                                                              d_ff=int(hidden_size/2), d_model=hidden_size,
                                                              n_heads=12, dropout=0.1) for _ in range(self.layers)])
        self.transformer_layers_type_cross = nn.ModuleList([EncoderLayer(d_k=hidden_size, d_v=hidden_size,
                                                                   d_ff=int(hidden_size / 2), d_model=hidden_size,
                                                                   n_heads=12, dropout=0.1) for _ in
                                                      range(8)])
        self.transformer_layer = nn.ModuleList([EncoderLayer(d_k=hidden_size,d_v=hidden_size,d_ff=int(hidden_size/2),d_model=hidden_size,n_heads=12,dropout=0.1) for _ in range(self.layers)])

        self.fc1 = nn.Sequential(
            nn.Linear(768, 300, bias=True),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 125, bias=True),
            nn.Dropout(0.2),
            nn.Linear(125, len(all_events), bias=True),

        ).to(device)
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size),
            # nn.BatchNorm1d(400),
            # nn.Dropout(0.1),
            # nn.Linear(400, 125),
            # nn.BatchNorm1d(125),
            # nn.Dropout(0.1),
            # nn.Linear(125, trigger_size),
        ).to(device)
        self.fc_argument = nn.Sequential(
            nn.Linear(2236+868, 500),
            nn.BatchNorm1d(30),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear( 500,125),
            nn.BatchNorm1d(30),
            nn.Dropout(0.5),
            nn.Linear(125, 1)
        ).to(device)
        # self.fc_argument_event = nn.Sequential(
        #     nn.Linear(hidden_size * 2, 500),
        #     nn.Linear(500, argument_size),
        # )
        # self.fc_argument_eventtype = nn.Sequential(
        #     nn.Linear(hidden_size + 500, 500),
        #     nn.Linear(500, argument_size),
        # )
        # self.fc_argument_entity = nn.Sequential(
        #     nn.Linear(hidden_size + 500, 500),
        #     nn.Linear(500, argument_size),
        # )

        self.bert = BertModel.from_pretrained('bert-base-cased/').to(device)
        # for param in self.bert.parameters():
        #     param.requires_grad = True

        self.tri_embsize = 500
        self.arg_embsize = 500

        # self.postag_embed = nn.Embedding(num_embeddings=all_postags, embedding_dim=self.pos_embsize)
        self.argument_size=argument_size
        # self.trigger_embed = nn.Embedding(num_embeddings=len(event_list), embedding_dim=self.tri_embsize).to(device)
        self.argument_embed = nn.Embedding(num_embeddings=argument_size, embedding_dim=self.arg_embsize).to(device)

        # self.position_embed = nn.Embedding(num_embeddings=100, embedding_dim=self.pos_embsize)
        self.entity_embeds = nn.Embedding(num_embeddings=len(ENTITIES), embedding_dim=500).to(device)
        self.trigger_embedings=nn.Embedding(num_embeddings=trigger_size,embedding_dim=hidden_size).to(device)
        # self.trigger_embedings_without_bi = nn.Embedding(num_embeddings=len(event_list)+1, embedding_dim=800).to(device)


        self.v=torch.nn.Parameter(torch.FloatTensor(np.random.uniform(-1,1,868)))

        self.data_weight = torch.LongTensor([i for i in range(len(all_events))]).to(self.device)
        self.data_mask = torch.LongTensor([(i + 1) for i in range(len(all_events))]).to(self.device)

        # self.pos_emb = PosEncoding(50 * 10, 100)
    def predict_triggers(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d,epoch,glove_id_2d=None,sent_trigger_y_2d = None,position_2d=None,seqlens_1d=None):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        # postags_x_2d = torch.LongTensor(postags_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)
        glove_id_2d = torch.LongTensor(glove_id_2d).to(self.device)
        sent_trigger_y_2d = torch.FloatTensor(sent_trigger_y_2d).to(self.device)
        position_2d = torch.LongTensor(position_2d).to(self.device)

        seqlens_1d=torch.LongTensor(seqlens_1d).to(self.device)


        # xlen = [max(x) for x in head_indexes_2d]
        # batch_size = tokens_x_2d.shape[0]
        # seqlen = tokens_x_2d.shape[1]
        # mask = np.zeros(shape=[batch_size, seqlen], dtype=np.uint8)
        # mask = torch.ByteTensor(mask).to(self.device)
        # for i in range(len(xlen)):
        #     mask[i, :xlen[i]] = 1
        # self.mask = mask

        # postags_x_2d = self.postag_embed(postags_x_2d)
        # entity_x_2d = self.entity_embed(entities_x_3d)

        if self.training:
            self.bert.train()
            encoded_layers, pooled = self.bert(tokens_x_2d)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, pooled = self.bert(tokens_x_2d)
                enc = encoded_layers[-1]

        # event_type_predict
        enc_c = torch.chunk(enc, enc.shape[1], dim=1)
        enc_c = enc_c[0]
        enc_c = enc_c.squeeze(1)
        event_logits = self.fc1(enc_c)
        event_sigmoid = torch.sigmoid(event_logits)

        glove_embedding = self.glove_embed(glove_id_2d)
        enc = torch.cat([enc, glove_embedding], -1)
        # enc += self.pos_emb(seqlens_1d)

        # for layer in self.transformer_layers:
        #     temp=enc
        #     enc, attn = layer(enc, enc_self_attn_mask)
        #     enc+=temp
        # enc, attn = self.transformer_encoder(enc, enc_self_attn_mask)

        # x = torch.cat([enc, entity_x_2d, postags_x_2d], 2)
        # x = self.fc1(enc)  # x: [batch_size, seq_len, hidden_size]

        # logits = self.fc2(x + enc)
        # glove_embedding = self.glove_embed(glove_id_2d)
        # x = torch.cat([enc, glove_embedding], -1)
        batch_size = tokens_x_2d.shape[0]

        # x = x.transpose(1, 2).contiguous()
        # x = self.cnn_1(x)
        # x = x.transpose(1, 2).contiguous()


        # (enc_inputs, self_attn_mask):
        # x, attn = self.transformer_encoder(x, enc_self_attn_mask)


        x=enc
        # x = F.max_pool1d(x, x.size(2))
        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])

        # x, _ = self.rnn(x)
        # x = x.transpose(1, 2).contiguous()
        # x = self.cnn_1(x)
        # x = x.transpose(1, 2).contiguous()


        data_weight=self.data_weight.unsqueeze(0)
        data_weight = data_weight.repeat(batch_size, 1)
        data = self.trigger_embedings(data_weight)



        # x=torch.cat([x,data],1)


        data_mask = self.data_mask.unsqueeze(0)
        data_mask = data_mask.repeat(batch_size, 1)
        # data_mask=torch.cat([tokens_x_2d,data_mask],-1)
        enc_attn_mask = get_attn_pad_mask(tokens_x_2d, data_mask)

        # tag=self.v.unsqueeze(0)
        # tag=tag.unsqueeze(0)
        # tag=tag.repeat(x.shape[0],1,1)
        # x=torch.cat([x,tag],1)
        enc_self_attn_mask = get_attn_pad_mask(tokens_x_2d, tokens_x_2d)
        # enc_attn_mask = get_attn_pad_mask(data_mask, data_mask)


        # event_logits_temp=event_logits.unsqueeze(-1)
        # data=data*event_logits_temp


        attn_=None
        temp = x
        for layer in self.transformer_layers_type_cross:

            # x, attn = layer(x, enc_attn_mask)
            temp, attn_ = layer(temp,data, enc_attn_mask,scale_factor=False,attn_=attn_)

        trigger_hat_2d = attn_.argmax(-1)
        trigger_logits = attn_

        for layer in self.transformer_layers_type:
            temp=x
            # x, attn = layer(x, enc_attn_mask)
            x, attn = layer(x,x, enc_self_attn_mask)
            x+=temp




        # trigger_index = self.attention_label(x, data)
        # trigger_embedd=self.trigger_embedings(trigger_index)

        # x=torch.cat([x,trigger_embedd],-1)

        # x = x.transpose(1, 2).contiguous()
        # x = self.cnn_1(x)
        # x = x.transpose(1, 2).contiguous()
        # x, _ = self.rnn(x)
        # pooled = pooled.unsqueeze(1)
        # pooled = pooled.repeat(1, x.shape[1], 1)
        # x=torch.cat([x,pooled],-1)

        # trigger_logits = self.fc_trigger(x)
        # trigger_hat_2d = trigger_logits.argmax(-1)
        x=x[:,0:tokens_x_2d.shape[1],:]
        # trigger_hat_2d,trigger_logits=[],[]

        argument_hidden,argument_keys=[],[]
        # argument_hidden_event,argument_hidden_eventtype,argument_hidden_entity, argument_keys = [], [],[], []
        sentence_event=[]
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            entity_tensor=[]
            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = x[i, e_start:e_end, ].mean(dim=0)
                entity_tensor.append(x[i, e_start:e_end, ].mean(dim=0))

            # if entity_tensor==[]:
            #     v=[self.v]
            #     v = torch.stack(v)
            #     v = v.squeeze(0)
            # else:
            #     entity_tensor.insert(0,self.v)
            #
            #     data_mask_1 = torch.LongTensor([(i + 1) for i in range(len(entity_tensor))]).to(self.device)
            #     data_mask_1 = data_mask_1.unsqueeze(0)
            #     # data_mask = data_mask.repeat(batch_size, 1)
            #     # data_mask = torch.cat([tokens_x_2d, data_mask], -1)
            #     enc_attn_mask_1 = get_attn_pad_mask(data_mask_1, data_mask_1)
            # #
            #     entity_tensor = torch.stack(entity_tensor)
            #     entity_tensor = entity_tensor.unsqueeze(0)
            #     for layer in self.transformer_layer:
            #         temp=entity_tensor
            #         entity_tensor,att=layer(entity_tensor,entity_tensor, enc_attn_mask_1)
            #         entity_tensor+=temp
            #     # entity_t=0
            #     # for index,e_tensor in enumerate(entity_tensor):
            #     #     if index==0:
            #     #         entity_t=e_tensor
            #     #     else:
            #     #
            #     #         temp = torch.cat([entity_t, e_tensor], -1)
            #     #         temp = temp.unsqueeze(0)
            #     #         temp = temp.unsqueeze(0)
            #     #         temp = temp.transpose(1, 2).contiguous()
            #     #         temp = self.cnn_3(temp)
            #     #         temp = temp.transpose(1, 2).contiguous()
            #     #         temp = temp.squeeze(0)
            #     #         temp = temp.squeeze(0)
            #     #         sigmoid = torch.sigmoid(temp)
            #     #         entity_t = entity_t * sigmoid + e_tensor * (1 - sigmoid)
            #
            #     # entity_tensor=entity_t
            #
            #     # entity_tensor=torch.stack(entity_tensor)
            #     # entity_tensor=torch.mean(entity_tensor,dim=0)
            #
                # v=entity_tensor.squeeze(0)
                # v=v[0]
            #     entity_tensor=entity_tensor.repeat(x.shape[1],1)
            #
            #     x_temp=torch.cat([x[i],entity_tensor],-1)
            #     x_temp = x_temp.unsqueeze(0)
            #
            #
            #     x_temp = x_temp.transpose(1, 2).contiguous()
            #     x_temp=self.cnn_2_1(x_temp)
            #     x_temp = self.cnn_2_2(x_temp)
            #     x_temp = x_temp.transpose(1, 2).contiguous()
            #     x_temp=x_temp.squeeze(0)
            #     sigmoid=torch.sigmoid(x_temp)
            #     x_temp=x_temp*sigmoid+entity_tensor*(1-sigmoid)
            #
            # trigger_logits_temp=self.fc_trigger(x_temp)
            # trigger_hat_2d_temp = trigger_logits_temp.argmax(-1)
            # trigger_logits.append(trigger_logits_temp)
            # trigger_hat_2d.append(trigger_hat_2d_temp)

            # predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d_temp.tolist()])
            event_sigmoid_i = event_sigmoid[i].cpu().detach().numpy().tolist()
            event_sigmoid_dict={}
            for index,value in enumerate(event_sigmoid_i):
                event_sigmoid_dict[index]=value
            event_sigmoid_dict=sorted(event_sigmoid_dict.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
            sen_event=[]

            # v = v.squeeze()
            # v = v.repeat(self.argument_size, 1)

            if entity_tensor != []:
                entity_tensor.insert(0, self.v)

                data_mask_1 = torch.LongTensor([(i + 1) for i in range(len(entity_tensor))]).to(self.device)
                data_mask_1 = data_mask_1.unsqueeze(0)
                # data_mask = data_mask.repeat(batch_size, 1)
                # data_mask = torch.cat([tokens_x_2d, data_mask], -1)
                enc_attn_mask_1 = get_attn_pad_mask(data_mask_1, data_mask_1)
                #
                entity_tensor = torch.stack(entity_tensor)
                entity_tensor = entity_tensor.unsqueeze(0)
                for layer in self.transformer_layer:
                    temp = entity_tensor
                    entity_tensor, att = layer(entity_tensor, entity_tensor, enc_attn_mask_1)
                    entity_tensor += temp
                v = entity_tensor.squeeze(0)
                v = v[0]
                v = v.unsqueeze(0)
                v = v.repeat(self.argument_size, 1)


            for value  in event_sigmoid_dict:
                event_type_index, event_type_value=value

                if event_type_index==0 :break
                # print(event_type_index)
                sen_event.append(idx2events[event_type_index])
                if event_type_value>0.4:
            # for predicted_trigger in predicted_triggers:
            #     t_start, t_end, t_type_str = predicted_trigger

                # eventtype_emb = self.trigger_embedings_without_bi(torch.LongTensor([event_list.index(t_type_str)]).to(self.device))
                    eventtype_emb = self.trigger_embedings(torch.LongTensor([event_type_index]).to(self.device))
                    eventtype_emb = eventtype_emb.squeeze()
                    eventtype_emb = eventtype_emb.repeat(self.argument_size, 1)



                    t_type_str=all_events[event_type_index]

                    for j in range(len(candidates)):
                        e_start, e_end, e_type_str = candidates[j]
                        # entity_emb = self.entity_embeds(torch.LongTensor([ENTITIES.index(e_type_str)]).to(self.device))
                        # entity_emb = entity_emb.squeeze()

                        entity_t = golden_entity_tensors[candidates[j]]
                        entity_t=entity_t.unsqueeze(0)
                        entity_t = entity_t.repeat(self.argument_size, 1)
                        # entity_event_tensor=torch.cat([entity_tensor,event_tensor])
                        # entity_event_tensor=entity_event_tensor.unsqueeze(0)
                        # entity_event_tensor = entity_event_tensor.repeat(self.argument_size, 1)

                        argument_weight=torch.LongTensor([ i for i in range(30)]).to(self.device)
                        argument_embedding=self.argument_embed(argument_weight)


                        # argument_hidden_event.append(torch.cat([ entity_tensor,event_tensor]))
                        # argument_hidden_eventtype.append(torch.cat([entity_tensor,eventtype_emb]))
                        # argument_hidden_entity.append(torch.cat([entity_tensor,entity_emb]))
                        try:
                            argument_hidden.append(torch.cat([entity_t,argument_embedding,eventtype_emb,v],-1))
                        except:
                            print()
                        # argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))
                        argument_keys.append((i, t_type_str,e_start, e_end, e_type_str))
            sentence_event.append(sen_event)
        # trigger_hat_2d=torch.stack(trigger_hat_2d)
        # trigger_logits=torch.stack(trigger_logits)
        return trigger_logits, triggers_y_2d, trigger_hat_2d,  argument_hidden, argument_keys,event_logits,event_sigmoid,sent_trigger_y_2d,sentence_event

    def predict_arguments(self, argument_hidden, argument_keys, arguments_2d):

        argument_hidden = torch.stack(argument_hidden)
        argument_logits = self.fc_argument(argument_hidden)
        argument_logits=argument_logits.squeeze(-1)
        argument_hat_1d = argument_logits.argmax(-1)

        # argument_hidden_event = torch.stack(argument_hidden_event)
        # argument_hidden_event_logits = self.fc_argument_event(argument_hidden_event)
        # argument_hidden_eventtype = torch.stack(argument_hidden_eventtype)
        # argument_hidden_eventtype_logits = self.fc_argument_eventtype(argument_hidden_eventtype)
        # argument_hidden_entity = torch.stack(argument_hidden_entity)
        # argument_hidden_entity_logits = self.fc_argument_entity(argument_hidden_entity)

        # argument_hidden_event_hat_1d = argument_hidden_event_logits.argmax(-1)
        # argument_hidden_eventtype_hat_1d = argument_hidden_eventtype_logits.argmax(-1)
        # argument_hidden_entity_hat_1d = argument_hidden_entity_logits.argmax(-1)

        # mid = (argument_hidden_eventtype_hat_1d == argument_hidden_entity_hat_1d)
        #
        # argument_hat_1d = torch.where(mid == 1, argument_hidden_eventtype_hat_1d,argument_hidden_event_hat_1d)

        arguments_y_1d = []
        # for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
        for i, t_type_str,e_start, e_end, e_type_str in argument_keys:
            a_label = argument2idx[NONE]
            # if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
            #     for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
            flag=False
            for key in arguments_2d[i]['events'].keys():
                (t_start, t_end, t_type)=key
                if flag:break
                if t_type_str==t_type:
                    for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type)]:
                        if e_start == a_start and e_end == a_end:
                            a_label = a_type_idx
                            flag=True
                            break
            # for (t_start, t_end, t_type_str),(a_start, a_end, a_type_idx) in arguments_2d[i]['events'].items():#[(t_start, t_end, t_type_str)]:
            #     if e_start == a_start and e_end == a_end:
            #         a_label = a_type_idx
            #         break
            arguments_y_1d.append(a_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]

        # for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys, argument_hat_1d.cpu().numpy()):
        for (i, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys,argument_hat_1d.cpu().numpy()):
            if a_label == argument2idx[NONE]:
                continue
            if (event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(event_type_str)] = []
            argument_hat_2d[i]['events'][(event_type_str)].append((e_st, e_ed, a_label))

        return argument_logits,arguments_y_1d, argument_hat_1d, argument_hat_2d


# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
class MultiLabelEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_embeddings=None, embedding_dim=None,
                 dropout=0.5, padding_idx=0,
                 max_norm=None, norm_type=2,
                 device=torch.device("cpu")):
        super(MultiLabelEmbeddingLayer, self).__init__()

        self.matrix = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type)
        self.dropout = dropout
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(batch_size)
             for j in range(seq_len)]
        x = torch.stack(x).view(batch_size, seq_len, -1)

        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x
