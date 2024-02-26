import random
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def error_type(predict_df, target_df):

    t_f = target_df
    p_f = predict_df
    merged_df =[]

    print("init: target=",len(t_f)," predict=",len(p_f))
    # ACOS
    ACOS = pd.merge(t_f, p_f, left_on=['sent_id', 'Category_t', 'Aspect_t', 'Opinion_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Category_p', 'Aspect_p', 'Opinion_p', 'Sentiment_p'])
    ACOS = ACOS.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    ACOS['score']='OOOO'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(ACOS.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(ACOS.set_index(['sent_id', 'quad_ord_p']).index)]
    print("ACOS:",len(ACOS)," + target",len(t_f)," = ",len(ACOS) + len(t_f),"  :: predict=",len(p_f))
    
    #AOS_gold
    AOS_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t', 'Opinion_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Aspect_p', 'Opinion_p', 'Sentiment_p'])
    AOS_gold = AOS_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    AOS_gold['score']='OXOO'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(AOS_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(AOS_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("AOS_gold:",len(AOS_gold)," + target",len(t_f)," = ",len(AOS_gold) + len(t_f),"  :: predict=",len(p_f))
    
    #COS_gold
    COS_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Category_t', 'Opinion_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Category_p', 'Opinion_p', 'Sentiment_p'])
    COS_gold = COS_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    COS_gold['score']='XOOO'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(COS_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(COS_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("COS_gold:",len(COS_gold)," + target",len(t_f)," = ",len(COS_gold) + len(t_f),"  :: predict=",len(p_f))
    
    #ACS_Gold
    ACS_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t', 'Category_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Aspect_p', 'Category_p', 'Sentiment_p'])
    ACS_gold = ACS_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    ACS_gold['score']='OOXO'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(ACS_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(ACS_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("ACS_gold:",len(ACS_gold)," + target",len(t_f)," = ",len(ACS_gold) + len(t_f),"  :: predict=",len(p_f))
    
    #ACO_gold
    ACO_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t', 'Opinion_t', 'Category_t'],
                         right_on=['sent_id', 'Aspect_p', 'Opinion_p', 'Category_p'])
    ACO_gold = ACO_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    ACO_gold['score']='OOOX'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(ACO_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(ACO_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("ACO_gold:",len(ACO_gold)," + target",len(t_f)," = ",len(ACO_gold) + len(t_f),"  :: predict=",len(p_f))
    
    #AO_gold  AO, AC, AS, CO, CS, OS
    AO_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t', 'Opinion_t'],
                         right_on=['sent_id', 'Aspect_p', 'Opinion_p'])
    AO_gold = AO_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    AO_gold['score']='OXOX'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(AO_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(AO_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("AO_gold:",len(AO_gold)," + target",len(t_f)," = ",len(AO_gold) + len(t_f),"  :: predict=",len(p_f))

    #AC_gold  AO, AC, AS, CO, CS, OS
    AC_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t', 'Category_t'],
                         right_on=['sent_id', 'Aspect_p', 'Category_p'])
    AC_gold = AC_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    AC_gold['score']='OOXX'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(AC_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(AC_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("AC_gold:",len(AC_gold)," + target",len(t_f)," = ",len(AC_gold) + len(t_f),"  :: predict=",len(p_f))

    #AS_gold  AO, AC, AS, CO, CS, OS
    AS_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Aspect_p', 'Sentiment_p'])
    AS_gold = AS_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    AS_gold['score']='OXXO'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(AS_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(AS_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("AS_gold:",len(AS_gold)," + target",len(t_f)," = ",len(AS_gold) + len(t_f),"  :: predict=",len(p_f))

    #CO_gold  AO, AC, AS, CO, CS, OS
    CO_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Category_t', 'Opinion_t'],
                         right_on=['sent_id', 'Category_p', 'Opinion_p'])
    CO_gold = CO_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    CO_gold['score']='XOOX'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(CO_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(CO_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("CO_gold:",len(CO_gold)," + target",len(t_f)," = ",len(CO_gold) + len(t_f),"  :: predict=",len(p_f))

    #CS_gold  AO, AC, AS, CO, CS, OS
    CS_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Category_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Category_p', 'Sentiment_p'])
    CS_gold = CS_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    CS_gold['score']='XOXO'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(CS_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(CS_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("CS_gold:",len(CS_gold)," + target",len(t_f)," = ",len(CS_gold) + len(t_f),"  :: predict=",len(p_f))

    #OS_Gold  AO, AC, AS, CO, CS, OS
    OS_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Opinion_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Opinion_p', 'Sentiment_p'])
    OS_gold = OS_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    OS_gold['score']='OOXX'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(OS_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(OS_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("OS_gold:",len(OS_gold)," + target",len(t_f)," = ",len(OS_gold) + len(t_f),"  :: predict=",len(p_f))

    #A_gold
    A_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t'],
                         right_on=['sent_id', 'Aspect_p'])
    A_gold = A_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    A_gold['score']='OXXX'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(A_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(A_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("A_gold:",len(A_gold)," + target",len(t_f)," = ",len(A_gold) + len(t_f),"  :: predict=",len(p_f))

    #O_gold
    O_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Opinion_t'],
                         right_on=['sent_id', 'Opinion_p'])
    O_gold = O_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    O_gold['score']='XXOX'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(O_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(O_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("O_gold:",len(O_gold)," + target",len(t_f)," = ",len(O_gold) + len(t_f),"  :: predict=",len(p_f))

    #C_gold
    C_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Category_t'],
                         right_on=['sent_id', 'Category_p'])
    C_gold = C_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    C_gold['score']='XOXX'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(C_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(C_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("C_gold:",len(C_gold)," + target",len(t_f)," = ",len(C_gold) + len(t_f),"  :: predict=",len(p_f))

    #S_Gold
    S_gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Sentiment_t'],
                         right_on=['sent_id', 'Sentiment_p'])
    S_gold = S_gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    S_gold['score']='XXXO'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(S_gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(S_gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("S_gold:",len(S_gold)," + target",len(t_f)," = ",len(S_gold) + len(t_f),"  :: predict=",len(p_f))


    #etc_error
    t_f['score']='XXXX'
    p_f['score']='XXXX'

    merged_df = pd.concat([ACOS, ACO_gold, ACS_gold, AOS_gold, COS_gold,
                           AC_gold, AO_gold, AS_gold, CO_gold, CS_gold, OS_gold,
                           A_gold, C_gold, O_gold, S_gold,
                           t_f, p_f ])
    merged_df = merged_df.sort_values(by=['sent_id', 'quad_ord_t'])

    return merged_df

def trans_tuple(input_list):
    indexed_data = [(i, item) for i, sublist in enumerate(input_list, start=1) for item in sublist]
    df = pd.DataFrame(indexed_data, columns=['Index', 'Data'])
    df[[ 'Aspect','Category', 'Sentiment', 'Opinion']] = pd.DataFrame(df['Data'].tolist(), index=df.index)
    df.drop('Data', axis=1, inplace=True)
    df.rename(columns={'Index': 'sent_id'}, inplace=True)
    df['quad_ord'] = df.groupby('sent_id').cumcount() + 1
    df['max_ord'] = df.groupby('sent_id')['quad_ord'].transform('max').astype(int)
    df = df[['sent_id', 'quad_ord', 'max_ord','Aspect', 'Category',  'Opinion', 'Sentiment']]
    return df


def extract_quad(quard_list, seq_type='gold'):
    target = []
    for seq in quard_list:
        quads = []
        sents = [s.strip() for s in seq.split('[SSEP]')]
        for s in sents:
            try:
                tok_list = ["[C]", "[S]", "[A]", "[O]"]
    
                for tok in tok_list:
                    if tok not in s:
                        s += " {} null".format(tok)
                index_ac = s.index("[C]")
                index_sp = s.index("[S]")
                index_at = s.index("[A]")
                index_ot = s.index("[O]")
    
                combined_list = [index_ac, index_sp, index_at, index_ot]
                arg_index_list = list(np.argsort(combined_list))
    
                result = []
                for i in range(len(combined_list)):
                    start = combined_list[i] + 4
                    sort_index = arg_index_list.index(i)
                    if sort_index < 3:
                        next_ = arg_index_list[sort_index + 1]
                        re = s[start:combined_list[next_]]
                    else:
                        re = s[start:]
                    result.append(re.strip())
    
                ac, sp, at, ot = result
    
                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'null'
            except ValueError:
                try:
                    print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                ac, at, sp, ot = '', '', '', ''
    
            quads.append((ac, at, sp, ot))
    
        target.append(quads)
    return target

def split_read(file_name):
    inputs, targets =[], []
    with open(file_name, 'r', encoding='UTF-8') as fp:
        for line in fp:
            input, target = [], []
            input, target = line.strip().split('####')
            if line != '':
                inputs.append(input)
                targets.append(target)
    print('Data read. Total count: ',len(targets))
    return inputs, targets

