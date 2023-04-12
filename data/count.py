import json

f=open('test.json','r',encoding='utf-8')
data=json.load(f)

arguments_dict={}
arguments_dict['>3']=0

trigger_dict={}
trigger_dict[0]=0
trigger_dict[1]=0
trigger_dict[2]=0
trigger_dict[3]=0
trigger_dict['>3']=0

count=0
for item in data:
    gold_event_mentions=item['golden-event-mentions']
    tri_len=len(gold_event_mentions)
    # if tri_len in [0,1,2,3]:
    #     trigger_dict[tri_len]+=1
    # else:
    #     trigger_dict['>3'] += 1


    for event_mention in gold_event_mentions:
        arguments=event_mention['arguments']
        arg_len=len(arguments)

        if tri_len>3 and arg_len==0:
            print(gold_event_mentions)
            count+=1
print(count)

    #     if arg_len == 0:
    #         if arg_len not in arguments_dict:
    #             arguments_dict[arg_len] = 1
    #         else:
    #             arguments_dict[arg_len] += 1
    #     elif arg_len == 1:
    #         if arg_len not in arguments_dict:
    #             arguments_dict[arg_len] = 1
    #         else:
    #             arguments_dict[arg_len] += 1
    #     elif arg_len == 2:
    #         if arg_len not in arguments_dict:
    #             arguments_dict[arg_len] = 1
    #         else:
    #             arguments_dict[arg_len] += 1
    #     elif arg_len == 3:
    #         if arg_len not in arguments_dict:
    #             arguments_dict[arg_len] = 1
    #         else:
    #             arguments_dict[arg_len] += 1
    #     elif arg_len > 3:
    #         arguments_dict['>3'] += 1

# print(arguments_dict)
#train.json    {'>3': 345,  3: 706,  2: 1223,  1: 1072,  0: 542}
#dev.json      {'>3': 58,   3: 93,   2: 125,   1: 124   ,0: 71 }
#test.json     {'>3': 59,   3: 110,  2: 101,   1: 93,    0: 49}

# print(trigger_dict)
#train.json      {0: 11728, 1: 2214, 2: 558, 3: 140, '>3': 32}
#dev.json        {0: 551,   1: 212,  2: 84,  3: 17,  '>3': 9}
#test.json       {0: 426,   1: 181,  2: 84,  3: 18,  '>3': 2}

'''
train.json
    tri_len     1           1       1       1       2       2       2       2       3       3       3       3       3+      3+
    arg_len     0           1       2       3+      0       1       2       3+      0       1       2       3+      0       1+
    num         338         654     705     517     148     271     372     325     44      116     116     144     12      126
dev.json
    tri_len     1       1       1       1       2       2       2       2       3       3       3       3       3+      3+
    arg_len     0       1       2       3+      0       1       2       3+      0       1       2       3+      0       1+
    num         33      61      66      52      31      35      40      62      5       16      12      18      2       38
test.json
    tri_len     1       1       1       1       2       2       2       2       3       3       3       3       3+      3+
    arg_len     0       1       2       3+      0       1       2       3+      0       1       2       3+      0       1+
    num         27      49      40      65      18      29      43      78      1       12      15      26      3       6
'''