import numpy as np
import math


class DATA(object):
    def __init__(self, n_question, seqlen, separate_char):   ###n_question：用于评价指标的辅助；；；；seqlen：用于统一输入的长度，transformer中不需要
        self.separate_char = separate_char    ###separate_char：：分隔符
        self.n_question = n_question
        self.seqlen = seqlen

    # data format
    # length
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []   ###生成空列表
        qa_data = []
        idx_data = []
        for lineID, line in enumerate(f_data):  ###enumerate：：枚举
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 0:
                student_id = lineID // 3
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)  ##每一行按分隔符分离元素
                if len(Q[len(Q) - 1]) == 0:
                    Q = Q[:-1]     ###删除最后一个元素，因为分隔符的原因，可能最后一个是空的
                # print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data   分离数据，即针对较长的序列
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)  ###向下取整
                    if len(Q) % self.seqlen:   ###判断是否整除
                        n_split = n_split + 1
                # print('n_split:',n_split)    ###确定n_split的大小
                for k in range(n_split):    ##从0到n_split - 1
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k + 1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):   ###例如创建一个101-200的序列
                        if len(Q[i]) != 0:
                            # int(A[i]) is in {0,1}
                            Xindex = int(float(A[i]))# * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)   ###每次是添加一个句子相当于
                    qa_data.append(answer_sequence)
                    idx_data.append(student_id)   ###标记同一个学生的不同记录
        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))    ###用0填充长度不足seqlen的
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_dataArray, qa_dataArray, np.asarray(idx_data)


class PID_DATA(object):
    def __init__(self, n_question, seqlen, separate_char):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_question = n_question

    # data format
    # length
    # pid1, pid2, ...
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        qa_data = []
        p_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 4 == 2:
                Q = line.split(self.separate_char)
                if len(Q[len(Q) - 1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            if lineID % 4 == 1:
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]

            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    problem_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k + 1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            problem_sequence.append(int(P[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    p_data.append(problem_sequence)

        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat
        return q_dataArray, qa_dataArray, p_dataArray