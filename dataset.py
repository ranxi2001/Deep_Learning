import random
import pandas as pd
import tqdm
import numpy as np


data = pd.read_csv('./data/数据1.csv', usecols=['data_id', 'daima', 'chengjiaoliang', 'chengjiaojine', 'huanshoulv', 'zhangdie'])  ##usecols函数实现读取某些列
#print(data)
raw_number = data.daima.unique().tolist()  ##去掉重复的daima
num_stock = len(raw_number)
print("number of gupiaos: %d" % num_stock)
stocks = {p: i for i, p in enumerate(raw_number)}####给股票进行编号
#print(stocks)


# %%

def parse_all_seq(stocks):    ###解析所有的句子
    all_sequences = []
    for stock_id in tqdm.tqdm(stocks, 'parse stock sequence:\t'):
        stock_sequence = parse_stock_seq(data[data.daima == stock_id])
        all_sequences.extend([stock_sequence])  ###表示每个股票的记录，每个学生的练习再用括号表示题号和答对与否
    return all_sequences


def parse_stock_seq(stock):    ##按顺序剥离出信息
    seq = stock.sort_values('data_id')  ##按Row的顺序排列
    cjl = seq.chengjiaoliang.tolist()
    cjl = np.array(cjl)
    cjl = cjl/1000
    cjl = cjl.astype(int)
    cjje = seq.chengjiaojine.tolist()
    hsl = seq.huanshoulv.tolist()
    zd = seq.zhangdie.tolist()
    zd = np.array(zd)
    zd[zd > 0.0] = 1
    zd[zd == 0.0] = 0
    zd[zd < 0.0] = 0
    zd = zd.astype(int)
    return cjl, cjje, hsl, zd


# [(question_sequence_0, answer_sequence_0), ..., (question_sequence_n, answer_sequence_n)]
sequences = parse_all_seq(data.daima.unique())###表示所有股票的记录
# %%

def train_test_split(data, train_size=.7, shuffle=True):     ###分离训练集和测试集
    if shuffle:
        random.shuffle(data)     ##随机打乱
    boundary = round(len(data) * train_size)
    return data[: boundary], data[boundary:]    ##从前往后的数据


train_sequences, test_sequences = train_test_split(sequences)  ###使用的是只包含输出的张量


# %%

def sequences2tl(sequences, trgpath):    ###将练习日志变成tl的文件形式
    with open(trgpath, 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, 'write into file: '):
            cjls, cjjes, hsls, zds = seq    ##股票信息按Row排
            seq_len = len(cjls)     ###对于一个股票的日志
            f.write(str(seq_len) + '\n')  ###\n换行符
            f.write(','.join([str(cjl) for cjl in cjls]) + '\n')
            #f.write(','.join([str(cjje) for cjje in cjjes]) + '\n')
           # f.write(','.join([str(hsl) for hsl in hsls]) + '\n')
            f.write(','.join([str(zd) for zd in zds]) + '\n')

# save triple line format for other tasks
sequences2tl(train_sequences, './data/train.txt')
sequences2tl(test_sequences, './data/test.txt')

