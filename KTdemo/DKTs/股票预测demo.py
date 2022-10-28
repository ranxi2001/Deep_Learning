import logging
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from load_data import DATA
import math
from tqdm import tqdm
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
HIDDEN_SIZE = 10
NUM_LAYERS = 1
INPUT_SIZE = 200
seqlen = 200
n_question = 100
maxgradnorm = -1

dat = DATA(n_question=n_question, seqlen=seqlen, separate_char=',')

train_data_q, train_data_a, _ = dat.load_data('../data/train.txt')
test_data_q, test_data_a, _ = dat.load_data('../data/test.txt')



def binary_entropy(target, pred):     ###定义损失函数的计算
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0   ##乘1，损失函数变负数


def compute_auc(all_target, all_pred):     ###定义AUC的计算
    return metrics.roc_auc_score(all_target, all_pred, multi_class='ovo')


def compute_accuracy(all_target, all_pred):     ###定义ACC的计算
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train_one_epoch(net, params, optimizer, q_data, qa_data):
    net.train()
    batch_size = params['batch_size']
    n_question = params['n_question']
    maxgradnorm = params['maxgradnorm']###maxgradnorm梯度截断
    n = int(math.ceil(len(q_data) / batch_size))   ###计算了所需要的批次
    q_data = q_data.T  ###转置
    qa_data = qa_data.T
    # shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])   ##shape[1]矩阵的列数
    np.random.shuffle(shuffled_ind)   ###重排
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]


    pred_list = []
    target_list = []

    for idx in tqdm(range(n)):
        optimizer.zero_grad()

        q_one_seq = q_data[:, idx * batch_size: (idx + 1) * batch_size]
        qa_one_seq = qa_data[:, idx * batch_size: (idx + 1) * batch_size]

        input_q = np.transpose(q_one_seq[:, :])    ###转置
        input_qa = np.transpose(qa_one_seq[:, :])

        input_q = torch.from_numpy(input_q).float().to(device)
        input_qa = torch.from_numpy(input_qa).float().to(device)

        loss, pred = net(input_q, input_qa)
        pred = pred.detach().cpu().numpy()
        #print('迭代：loss', loss.sum())
        loss.sum().backward()    ###得到预测指标，进行反向传播

        if maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=maxgradnorm)

        optimizer.step()   ####进行迭代

        # correct: 1.0; wrong 0.0; padding -1.0

        pred_list.append(pred)
        target_list.append(input_qa)

    all_pred = np.concatenate(pred_list, axis=None)
    all_target = np.concatenate(target_list, axis=None).detach().cpu().numpy()
    print('all_target', all_target)
    print('all_pred', all_pred)
    print(max(all_pred))

    loss = binary_entropy(all_target, all_pred)  ##计算损失函数,,,总的损失函数，不是用于迭代器的损失
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


def test_one_epoch(net, params, q_data, qa_data):
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

        input_q = torch.from_numpy(input_q).float().to(device)
        input_qa = torch.from_numpy(input_qa).float().to(device)

        loss, pred = net(input_q, input_qa)
        pred = pred.detach().cpu().numpy()

        if (idx + 1) * batch_size > seq_num:
            real_batch_size = seq_num - idx * batch_size
            count += real_batch_size
        else:
            count += batch_size

        # correct: 1.0; wrong 0.0; padding -1.0

        pred_list.append(pred)
        target_list.append(input_qa)

    assert count == seq_num, 'Seq not matching'

    all_pred = np.concatenate(pred_list, axis=None)
    all_target = np.concatenate(target_list, axis=None)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


class Net(nn.Module):##定义了循环神经网络
    def __init__(self, input_size, hidden_size, num_layers):   ###新加了批次的属性，要不要加？
        super(Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  ##bidirectional=True，判断是否双向
        self.fc = nn.Linear(self.hidden_dim, input_size)

    def forward(self, x, y):##定义了残差块
        h0 = torch.zeros(self.layer_dim,  self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim,  self.hidden_dim).to(device)


        loss = nn.BCEWithLogitsLoss(reduction='none')
        out, _ = self.rnn(x, (h0, c0))
        out = out + 1    ###怎么修改，使其结果更合理，现在的预测值始终太小了
        res = self.fc(out)    ##输出的为input_size，并通过sigmoid函数回到【0，1】上，，，然而这里的res始终很小
        Loss = loss(res, y)
        return Loss, res


class DL:
    def __init__(self, input_size, hidden_size, num_layers):  ##重载属性
        super(DL, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.params = {
            'batch_size': batch_size,
            'n_question': n_question,
            'maxgradnorm': maxgradnorm,
        }
        self.dl_model = Net(input_size, hidden_size, num_layers).to(device)     ###调用RNN

    def train(self, train_data_q, train_data_qa, test_data=None, *, epoch: int, lr=0.001) -> ...:
        optimizer = torch.optim.Adam(self.dl_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        for idx in range(epoch):
            train_loss, train_auc, train_accuracy = train_one_epoch(self.dl_model, self.params, optimizer, train_data_q, train_data_a)
            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))

            if test_data is not None:
                valid_loss, valid_auc, valid_accuracy = self.eval(test_data)     ###每一次都是验证集的结果
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (idx, valid_auc, valid_accuracy))

    def eval(self, test_data_q, test_data_qa) -> ...:
        self.dl_model.eval()
        return test_one_epoch(self.dl_model, self.params, test_data_q, test_data_qa)    ##更新演化后的权重，，self.akt_net, self.params即存储的数据

    def save(self, filepath) -> ...:
        torch.save(self.dl_model.state_dict(), filepath)    ###存储模型中的参数
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.dl_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)





logging.getLogger().setLevel(logging.INFO)

dl = DL(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
dl.train(train_data_q, train_data_a, epoch=30)
dl.save("dl.params")
dl.load("dl.params")
loss, auc, accuracy = dl.eval(test_data_q, test_data_a)
print('loss', loss)
print('auc', auc)
print('acc', accuracy)
