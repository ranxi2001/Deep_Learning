import logging
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from load_data import DATA
import math
from tqdm import tqdm
from sklearn import metrics
from transformer_demo import AKTNet


###-------生成知识概念图
from graph import transition_graph
from format import tl2json
from graph import posterior_correct_probability_graph
#--------------
tl2json(src='../data_09/train.txt',tar='../data_09/train.json') ##将tl文件转为json文件
graph = transition_graph(198, '../data_09/train.json',tar='../data_09/transition_graph.json')
###这里得到的graph是list



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_question = 198
seqlen = 198
bata = 0.01   ###调整知识概念关联性的超参数



dat = DATA(n_question=n_question, seqlen=seqlen, separate_char=',')

train_data_q, train_data_a, _ = dat.load_data('../data_09/train.txt')
print(_)
test_data_q, test_data_a, _ = dat.load_data('../data_09/test.txt')
print(len(train_data_q))
print(_)
print(len(test_data_q))


def binary_entropy(target, pred):     ###定义损失函数的计算，，，评价指标
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0   ##乘1，损失函数变负数，，，计算公式决定需要负号


def compute_auc(all_target, all_pred):     ###定义AUC的计算
    return metrics.roc_auc_score(all_target, all_pred, multi_class='ovo')


def compute_accuracy(all_target, all_pred):     ###定义ACC的计算
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train_one_epoch(net, params, optimizer, q_data, qa_data, graph1):
    net.train()
    batch_size = params['batch_size']
    maxgradnorm = params['maxgradnorm']###maxgradnorm梯度截断
    n = int(math.ceil(len(q_data) / batch_size))   ###计算了所需要的批次，，向上取整
    q_data = q_data.T  ###转置
    qa_data = qa_data.T
    # shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])   ##shape[1]矩阵的列数
    np.random.shuffle(shuffled_ind)   ###重排0-shape[1]矩阵的列数
    q_data = q_data[:, shuffled_ind]    ###交换对应的列
    qa_data = qa_data[:, shuffled_ind]   ###交换对应的列，故学生的练习和作答依旧对应


    pred_list = []
    target_list = []

    for idx in tqdm(range(n)):
        optimizer.zero_grad()

        q_one_seq = q_data[:, idx * batch_size: (idx + 1) * batch_size]
        qa_one_seq = qa_data[:, idx * batch_size: (idx + 1) * batch_size]

        input_q = np.transpose(q_one_seq[:, :])    ###转置
        input_qa = np.transpose(qa_one_seq[:, :])

        len_q = len(input_q)
        #print('len_q', len_q)

        input_q = torch.from_numpy(input_q).int().to(device)    ###不同模型的时候这里需要改
        input_qa = torch.from_numpy(input_qa).int().to(device)

        graph = torch.tensor(graph1)
        one = torch.ones_like(graph)
        zero = torch.zeros_like(graph)
        graph = torch.where(graph > bata, one, graph)
        graph = torch.where(graph <= bata, zero, graph)
        graph2 = nn.Linear(198, len_q)
        graph = graph2(graph) * 1000 + 2000
        graph = graph.int()
###在这里出现了图数据的变形

        loss, pred, _ = net(input_q, input_qa, input_qa, graph)   ####预测指标从net中得出
        pred = pred.detach().cpu().numpy()
        #print('迭代：loss', loss.sum())
        loss.sum().backward()    ###得到预测指标，进行反向传播
        ##print('loss.sum', loss.sum())  ###打印损失函数

        if maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=maxgradnorm)

        optimizer.step()   ####进行迭代

        # correct: 1.0; wrong 0.0; padding -1.0
        input_qa = input_qa.cpu().numpy()
        pred_list.append(pred)
        target_list.append(input_qa)

    all_pred = np.concatenate(pred_list, axis=None)
    all_target = np.concatenate(target_list, axis=None)


    print('all_target', all_target)
    print('all_pred', all_pred)
    print(max(all_pred))

    loss = binary_entropy(all_target, all_pred)  ##计算损失函数,,,总的损失函数，不是用于迭代器的损失
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


def test_one_epoch(net, params, q_data, qa_data, graph1):
    batch_size = params['batch_size']
    net.eval()
    n = int(math.ceil(len(q_data) / batch_size))
    q_data = q_data.T
    qa_data = qa_data.T

    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []

    count = 0

    for idx in tqdm(range(n)):
        q_one_seq = q_data[:, idx * batch_size: (idx + 1) * batch_size]
        qa_one_seq = qa_data[:, idx * batch_size: (idx + 1) * batch_size]

        input_q = np.transpose(q_one_seq[:, :])
        input_qa = np.transpose(qa_one_seq[:, :])

        len_q = len(input_q)
        #print('len_q', len_q)

        graph = torch.tensor(graph1)
        one = torch.ones_like(graph)
        zero = torch.zeros_like(graph)
        graph = torch.where(graph > bata, one, graph)
        graph = torch.where(graph <= bata, zero, graph)
        graph2 = nn.Linear(198, len_q)
        graph = graph2(graph) * 1000 + 2000
        graph = graph.int()


        input_q = torch.from_numpy(input_q).int().to(device)   ###不同模型需要更改
        input_qa = torch.from_numpy(input_qa).int().to(device)

        loss, pred, _ = net(input_q, input_qa, input_qa, graph)
        pred = pred.detach().cpu().numpy()

        if (idx + 1) * batch_size > seq_num:
            real_batch_size = seq_num - idx * batch_size
            count += real_batch_size
        else:
            count += batch_size

        input_qa = input_qa.cpu().numpy()
        pred_list.append(pred)
        target_list.append(input_qa)

    assert count == seq_num, 'Seq not matching'

    all_pred = np.concatenate(pred_list, axis=None)
    all_target = np.concatenate(target_list, axis=None)
    


    print('all_target', all_target)
    print('all_pred', all_pred)
    print(max(all_pred))
    np.set_printoptions(threshold=np.inf) ###用于显示所有打印结果
    f = open ('../all_target.txt','w')
    print (all_target,file = f)
    f.close()
    f = open ('../all_pred.txt','w')
    print (all_pred,file = f)
    f.close()

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


class AKT:
    def __init__(self, n_question, n_blocks, d_model, dropout, kq_same, batch_size, maxgradnorm,
                 separate_qa=False):
        super(AKT, self).__init__()
        self.params = {
            'batch_size': batch_size,
            'n_question': n_question,
            'maxgradnorm': maxgradnorm,
        }
        self.akt_net = AKTNet(n_question=n_question, n_blocks=n_blocks, d_model=d_model, dropout=dropout,
                              kq_same=kq_same, seqlen=seqlen, separate_qa=separate_qa).to(device)

    tl2json(src='../data_09/train.txt', tar='../data_09/train.json')  ##将tl文件转为json文件
    graph = transition_graph(198, '../data_09/train.json', tar='../data_09/transition_graph.json')

    def train(self, train_data_q, train_data_a, test_data_q=None, test_data_a=None, *, epoch: int, lr=0.02) -> ...:

        for idx in range(epoch):
            optimizer = torch.optim.Adam(self.akt_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            train_loss, train_auc, train_accuracy = train_one_epoch(self.akt_net, self.params, optimizer, train_data_q, train_data_a, graph)
            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))

            if test_data_q is not None:
                valid_loss, valid_auc, valid_accuracy = self.eval(test_data_q, test_data_a)     ###每一次都是验证集的结果
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (idx, valid_auc, valid_accuracy))

    def eval(self, test_data_q, test_data_qa) -> ...:
        self.akt_net.eval()
        return test_one_epoch(self.akt_net, self.params, test_data_q, test_data_qa, graph)    ##更新演化后的权重，，self.akt_net, self.params即存储的数据

    def save(self, filepath) -> ...:
        torch.save(self.akt_net.state_dict(), filepath)    ###存储模型中的参数
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.akt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)


batch_size = 64
HIDDEN_SIZE = 10
NUM_LAYERS = 1
INPUT_SIZE = 100
maxgradnorm = -1
n_blocks = 1
d_model = 256
dropout = 0.5
kq_same = 1
n_question = 198
n_question = 198
seqlen = 198
bata = 0.01   ###调整知识概念关联性的超参数


logging.getLogger().setLevel(logging.INFO)

akt = AKT(n_question, n_blocks, d_model, dropout, kq_same, batch_size, maxgradnorm)
akt.train(train_data_q, train_data_a, epoch=5)
akt.save("akt.params")   ##保存权重的文件
akt.load("akt.params")    ##载入权重文件
loss, auc, accuracy = akt.eval(test_data_q, test_data_a)
print("loss: %.6f, auc: %.6f, accuracy: %.6f" % (loss, auc, accuracy))
