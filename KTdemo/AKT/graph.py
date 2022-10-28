# coding: utf-8
# 2019/12/12 @ tongshiwei

import fileinput
import json
import warnings

import numpy as np
from longling import wf_open
from scipy.spatial.distance import cdist
from scipy.special import softmax
from tqdm import tqdm
###本文件构建了各种图的生成过程
__all__ = [
    "dense_graph",
    "correct_transition_count_graph",
    "correct_transition_graph", "transition_graph",
    "posterior_correct_probability_graph", "posterior_correct_transition_graph",
    "similarity_graph",
    "concurrence_graph", "correct_co_influence_graph"
]


def dense_graph(ku_num: int, tar=None, undirected: bool = False): ##undirected，无向图
    """
    Dense graph where any two vertex have a link  ##任何两个顶点都有连接的密集图

    No self loop is reserved.

    Parameters
    ----------
    ku_num: int
    tar
    undirected

    Examples
    --------
    Target file is a json file, json.load can be used to read it.

    Demo of target file with undirected tag is False:
    [
        [0, 1],
        [0, 2],
        [1, 0],
        ...
        [2, 0],
        [2, 1]
    ]

    Demo of target file with undirected tag is True:
    [
        [0, 1],
        [1, 2],
        [0, 2]
    ]

    >>> dense_graph(3)
    [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
    >>> dense_graph(3, undirected=True)
    [[0, 1], [0, 2], [1, 2]]
    """
    _graph = []

    if undirected:  ###判断两个知识是否有连接，有的话加入到列表里，，这里做的是有向图
        for i in range(ku_num):
            for j in range(i + 1, ku_num):
                _graph.append([i, j])
    else:
        for i in range(ku_num):
            for j in range(ku_num):
                if i != j:
                    _graph.append([i, j])

    if tar is not None:
        with wf_open(tar) as wf:
            json.dump(_graph, wf, indent=2)   ##打开图文件，，，如果 ``indent`` 是一个非负整数，那么 JSON 数组元素和对象成员将使用该缩进级别进行漂亮打印。 缩进级别 0 只会插入换行符。 ``None`` 是最紧凑的表示
            print(json.dump)
    return _graph

##下述定义了一个两个图的除法过程函数
def _nan_divide(dividend_graph, divisor_graph, fill_na_with: (float, None) = 0., to_list=True):
    """

    Parameters
    ----------
    dividend_graph
    divisor_graph
    fill_na_with: float or None
    to_list

    Examples
    -------
    >>> a = [[10, 20], [30, 50]]
    >>> b = [[100, 200], [50, 100]]
    >>> _nan_divide(a, b)
    [[0.1, 0.1], [0.6, 0.5]]
    >>> a = [[0, 20], [30, 50]]
    >>> b = [[0, 200], [50, 100]]
    >>> _nan_divide(a, b)
    [[0.0, 0.1], [0.6, 0.5]]
    >>> _nan_divide(a, b, to_list=False)
    array([[0. , 0.1],
           [0.6, 0.5]])
    """
    _dividend_graph = np.asarray(dividend_graph)
    _divisor_graph = np.asarray(divisor_graph)

    _quotient_graph = _dividend_graph / _divisor_graph   ##商的图

    if fill_na_with is not None:
        _quotient_graph[np.isnan(_quotient_graph)] = fill_na_with  ###isnan判断是否为NAN

    if to_list:  ###判断是否加入list
        _quotient_graph = _quotient_graph.tolist()
    return _quotient_graph

###以下定义标准化记数图，即对列表中的数据归一化
def _row_normalize(count_graph, diagonal_value=None, normalized_by_softmax=False, skip_zero_row=False):
    """
    Normalize the count graph

    Parameters
    ----------
    count_graph
    diagonal_value: float or None
    normalized_by_softmax    ###经softmax函数得到的标准化图
    skip_zero_row  ##跳过为0的行

    Examples
    --------
    >>> _count_graph = [[0, 0, 0], [2, 8, 10], [82, 8, 0]]
    >>> _row_normalize(_count_graph, diagonal_value=0, skip_zero_row=True)
    [[0.0, 0.0, 0.0], [0.16666666666666666, 0.0, 0.8333333333333334], [0.9111111111111111, 0.08888888888888889, 0.0]]
    >>> _count_graph = [[0, 0], [2, 8]]
    >>> _row_normalize(_count_graph)
    [[0.5, 0.5], [0.2, 0.8]]
    >>> _count_graph = [[2, 3], [2, 8]]
    >>> _row_normalize(_count_graph)
    [[0.4, 0.6], [0.2, 0.8]]
    >>> _count_graph = [[0, 0], [2, 8]]
    >>> _row_normalize(_count_graph, normalized_by_softmax=True)   ##使用激活函数归一化
    [[0.5, 0.5], [0.002472623156634775, 0.9975273768433656]]
    >>> _count_graph = [[2, 3], [2, 8]]
    >>> _row_normalize(_count_graph, normalized_by_softmax=True)
    [[0.26894142136999505, 0.7310585786300048], [0.002472623156634775, 0.9975273768433656]]
    >>> _count_graph = [[0, 0, 0], [2, 8, 10], [82, 8, 0]]
    >>> _row_normalize(_count_graph, diagonal_value=0)  ###类似矩阵的主对角线设置为0，再归一化计算
    [[0.0, 0.5, 0.5], [0.16666666666666666, 0.0, 0.8333333333333334], [0.9111111111111111, 0.08888888888888889, 0.0]]
    >>> _count_graph = [[0, 0, 0], [2, 8, 10], [82, 8, 0]]
    >>> import numpy as np
    >>> np.asarray(_row_normalize(_count_graph, normalized_by_softmax=True))  ##以矩阵形式保存
    array([[3.33333333e-01, 3.33333333e-01, 3.33333333e-01],
           [2.95387223e-04, 1.19167711e-01, 8.80536902e-01],
           [1.00000000e+00, 7.28129018e-33, 2.44260074e-36]])
    >>> np.asarray(_row_normalize(_count_graph, normalized_by_softmax=True, diagonal_value=0.0)) ##主对角线为0的矩阵保存
    array([[0.00000000e+00, 5.00000000e-01, 5.00000000e-01],
           [3.35350130e-04, 0.00000000e+00, 9.99664650e-01],
           [1.00000000e+00, 7.28129018e-33, 0.00000000e+00]])
    """
    _graph = np.asarray(count_graph)
    zero_rows_indices = None

    if skip_zero_row:   ##判断是否跳过为0的信息，对求和做不同处理
        zero_rows_indices = np.sum(_graph, axis=-1) == 0

    if normalized_by_softmax:##判断是否使用激活函数
        if diagonal_value is None:  ##判断对角线数值是否为空
            _graph = softmax(_graph, axis=-1)  ##是的话，对所以的元素使用激活函数
        else:
            _graph = _graph.astype(float)
            np.fill_diagonal(_graph, -np.inf)
            _graph = softmax(_graph, axis=-1)
            if diagonal_value is not None:
                np.fill_diagonal(_graph, diagonal_value)
    else:
        _offset_graph = _graph + 1e-50
        if diagonal_value is not None:
            np.fill_diagonal(_offset_graph, 0.0)
            _graph = (_offset_graph.T / _offset_graph.sum(axis=-1)).T
            _graph[_graph <= 1e-40] = 0.0
            np.fill_diagonal(_graph, diagonal_value)
        else:
            _graph = (_offset_graph.T / _offset_graph.sum(axis=-1)).T
            _graph[_graph <= 1e-40] = 0.0

    if skip_zero_row:
        _graph[zero_rows_indices] = 0.0

    return _graph.tolist()
###输出为list的形式，可利用numpy库，变为矩阵保存

def _output_graph(graph, tar):
    ku_num = len(graph)

    _graph = []

    for i in range(ku_num):
        for j in range(ku_num):
            if i != j and not np.isnan(graph[i][j]) and graph[i][j] > 0:
                _graph.append([i, j, graph[i][j]])

    with wf_open(tar) as wf:
        json.dump(_graph, wf, indent=2)
###输出知识概念间的连通关系

def _correct_transition_count_graph(count_graph, seq): ###定义基于正确情况的转移过程，即上一题做对，下一题即做对的过渡
    pre_c = None
    for eid, r in seq:
        if pre_c is not None:
            if eid != pre_c and r == 1:
                count_graph[pre_c][eid] += 1
            elif r == 1:
                pass
        if r == 1:
            pre_c = eid
        else:
            pre_c = None


def correct_transition_count_graph(ku_num, *src, tar=None, input_is_file=True):  ###和自己的想法相关，用于前后序概念的认知
    """

    Parameters
    ----------
    ku_num
    src
    tar
    input_is_file

    Examples
    -------
    >>> _seq = [[[0, 1], [1, 0], [1, 1], [2, 1]], [[2, 0], [1, 0], [0, 1], [2, 1]]]##[1, 1], [2, 1]； [0, 1], [2, 1]由于这两个产生的转移矩阵
    >>> correct_transition_count_graph(3, _seq, input_is_file=False)
    [[0, 0, 1], [0, 0, 1], [0, 0, 0]]
    >>> _seq = [[[0, 1], [1, 1], [1, 1], [2, 1]]]
    >>> correct_transition_count_graph(3, _seq, input_is_file=False)
    [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    """
    count_graph = [[0] * ku_num for _ in range(ku_num)]  ##生成一个知识概念×知识概念大小的n×n的矩阵，并且元素都为0

    if input_is_file:    ###判断是否是输入的文件
        with fileinput.input(files=src) as f:
            for line in tqdm(f, "constructing coorect transition graph"):
                if not line.strip():  # pragma: no cover
                    continue
                seq = json.loads(line)
                _correct_transition_count_graph(count_graph, seq)
    else:
        for seqs in src:
            for seq in seqs:
                _correct_transition_count_graph(count_graph, seq)

    if tar is not None:
        _output_graph(count_graph, tar)

    return count_graph


def correct_transition_graph(ku_num, *src, tar=None, input_is_file=True, diagonal_value=0.):##*src即输入的练习日志
    """
    When a concept is mastered, how much probability is it to be transferred to another concept.
    ##当一个概念被掌握后，它转移到另一个概念的概率有多大。， 只考虑下一题是什么，不考虑下一题的正确与否

    For example,

    ```
    [[0, 1], [1, 0], [1, 1], [2, 1]]
    [[2, 0], [1, 0], [0, 1], [2, 1]]
    ```
    When concept #0 is mastered (i.e., 1st in seq #1, 3rd in seq #2),
    only concept # 2 can be mastered (4th in seq #2).
    Thus, the transition probabilty for concept #0 is [0, 0, 1],
    which mastering concept #0 can influence mastering concept #2 more thant concept #1.#这里表明概念0比概念1更能影响概念2

    Parameters
    ----------
    ku_num
    src
    tar
    input_is_file
    diagonal_value

    Returns
    -------

    Examples
    -------
    >>> _seq = [[[0, 1], [1, 0], [1, 1], [2, 1]], [[2, 0], [1, 0], [0, 1], [2, 1]]]
    >>> correct_transition_graph(3, _seq, input_is_file=False)
    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    >>> _seq = [[[0, 1], [1, 1], [1, 1], [2, 1]]]
    >>> correct_transition_graph(3, _seq, input_is_file=False)
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    """
    count_graph = correct_transition_count_graph(ku_num, *src, tar=None, input_is_file=input_is_file)
    _transition_graph = _row_normalize(count_graph, diagonal_value, skip_zero_row=True)

    if tar is not None:
        _output_graph(_transition_graph, tar)

    return _transition_graph


def transition_graph(ku_num, *src, tar=None, input_is_file=True, diagonal_value=0.):  ##过渡图，这里不考虑做对的情况，只考虑做完什么题继续做什么题目
    """
    When a concept is learned, how much probability does another concept appear.
    ##当学习一个概念时，另一个概念出现的概率有多大。
    For example,

    ```
    [[0, 1], [1, 0], [1, 1], [2, 1]]
    [[2, 0], [1, 0], [0, 1], [2, 1]]
    ```
    When concept #0 is learned (i.e., 1st in seq #1, 3rd in seq #2),
    concept #2 and #1 could appear (2nd in seq #1, 4th in seq #2)
    Thus, the transition probabilty for concept #0 is [0, 0.5, 0.5].

    Parameters
    ----------
    ku_num
    src
    tar
    input_is_file
    diagonal_value

    Returns
    -------

    Examples
    --------
    >>> _seq = [[[0, 1], [1, 0], [1, 1], [2, 1]], [[2, 0], [1, 0], [0, 1], [2, 1]]]
    >>> transition_graph(3, _seq, input_is_file=False)
    [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]
    >>> _seq = [[[0, 1], [1, 1], [1, 1], [2, 1]]]
    >>> transition_graph(3, _seq, input_is_file=False)
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    """
    count_graph = [[0] * ku_num for _ in range(ku_num)]     ###生成ku_num*ku_num类型的列表

    def __transition_graph(_seq):
        pre = None
        for eid, _ in _seq:
            if pre is not None:
                if eid != pre:
                    count_graph[pre][eid] += 1
                else:
                    pass
            pre = eid

    if input_is_file:
        with fileinput.input(files=src) as f:
            for line in tqdm(f, "constructing transition graph"):
                if not line.strip():  # pragma: no cover
                    continue
                seq = json.loads(line)
                __transition_graph(seq)

    else:
        for seqs in src:
            for seq in seqs:
                __transition_graph(seq)

    _transition_graph = _row_normalize(count_graph, diagonal_value, skip_zero_row=True)   ###归一化
    if tar is not None:    ###存储图文件
        _output_graph(_transition_graph, tar)

    return _transition_graph


def posterior_correct_probability_graph(ku_num, *src, tar=None, input_is_file=True, fill_na_with=0.):
    """
    When a concept is mastered, how much probability is another concept correctly answered.
    ##当一个概念被掌握时，另一个概念被正确回答的概率有多大。考虑另一个概念的正确情况
    For example,

    ```
    [[0, 1], [1, 1], [1, 1], [2, 1]]
    [[2, 0], [1, 0], [0, 1], [2, 1]]
    ```
    When concept #0 is mastered (i.e., 1st in seq #1, 3rd in seq #2),
    concept #1 and # 2 can both be mastered (1th in seq # 1, 4th in seq #2).
    Thus, the posterior correct probability for concept #0 is [0, 1, 1].

    Parameters
    ----------
    ku_num
    src
    tar
    input_is_file
    fill_na_with

    Returns
    -------
    >>> _seq = [[[0, 1], [1, 0], [1, 1], [2, 1]], [[2, 0], [1, 0], [0, 1], [2, 1]]]
    >>> posterior_correct_probability_graph(3, _seq, input_is_file=False)
    [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    """
    count_graph = [[0] * ku_num for _ in range(ku_num)]
    correct_count_graph = [[0] * ku_num for _ in range(ku_num)]

    def __posterior_correct_graph(seq):
        pre_c = None
        for eid, r in seq:
            if pre_c is not None:
                if eid != pre_c:
                    count_graph[pre_c][eid] += 1
                    if r == 1:
                        correct_count_graph[pre_c][eid] += 1
                    elif r == 0:
                        correct_count_graph[pre_c][eid] += 1
            if r == 1:
                pre_c = eid
            else:
                pre_c = None

    assert src

    if input_is_file:
        with fileinput.input(files=src) as f:
            for line in tqdm(f, "constructing correct transition graph"):
                if not line.strip():  # pragma: no cover
                    continue
                seq = json.loads(line)
                __posterior_correct_graph(seq)
    else:
        for seqs in src:
            for seq in seqs:
                __posterior_correct_graph(seq)

    _posterior_correct_graph = _nan_divide(correct_count_graph, count_graph, fill_na_with)
    if tar is not None:
        _output_graph(_posterior_correct_graph, tar)

    return _posterior_correct_graph


def posterior_correct_transition_graph(ku_num, *src, tar=None, input_is_file=True, diagonal_value=None):
    """
    Correct transition graph based on posterior correct graph
   ###基于后验正确图的正确转换图，，，包含了转移和正确两种概率
    For example,

    ```
    [[0, 1], [1, 1], [1, 1], [2, 1]]
    [[2, 0], [1, 0], [0, 1], [2, 1]]
    ```
    When concept #0 is mastered (i.e., 1st in seq #1, 3rd in seq #2),
    concept #1 and # 2 can both be mastered (1th in seq # 1, 4th in seq #2).
    Thus, the posterior correct probability for concept #0 is [0, 1, 1].

    Parameters
    ----------
    ku_num
    src
    tar
    input_is_file
    diagonal_value

    Returns
    -------
    >>> _seq = [[[0, 1], [1, 0], [1, 1], [2, 1]], [[2, 0], [1, 0], [0, 1], [2, 1]]]
    >>> posterior_correct_transition_graph(3, _seq, input_is_file=False)
    [[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    >>> _seq = [[[0, 1], [1, 0], [1, 1], [2, 1]], [[2, 0], [1, 0], [0, 1], [2, 1]]]
    >>> posterior_correct_transition_graph(3, _seq, input_is_file=False)
    [[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    """
    return _row_normalize(
        posterior_correct_probability_graph(
            ku_num,
            *src,
            tar=tar,
            input_is_file=input_is_file
        ),
        diagonal_value=diagonal_value,
        skip_zero_row=True
    )


def correct_co_influence_graph(ku_num, *src, tar=None, input_is_file=True):  # pragma: no cover
    """

    Co-influence graph
    ##共同影响图 共同影响对被定义为转移计数之和较大且差异较小的两个顶点。 对角线值始终为 0
    A co-influence pair is defined as two vertexes
    that the sum of transition count is large and the difference is small.

    Diagonal_value is always 0

    Parameters
    ----------
    ku_num
    src
    tar
    input_is_file

    Examples
    --------
    >>> _seq = [
    ...     [[0, 1], [1, 0], [1, 1], [2, 0]],
    ...     [[0, 1], [1, 1], [2, 0], [2, 1]],
    ...     [[2, 1], [2, 1], [1, 1], [2, 0]],
    ...     [[1, 0], [0, 1], [0, 1], [2, 0]],
    ...     [[2, 0], [1, 1], [0, 1], [2, 1]],
    ... ]
    >>> correct_co_influence_graph(3, _seq, input_is_file=False)
    array([[0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 0.]])
    """
    warnings.warn("do not use this function due to the lack of support from theory")
    count_graph = correct_transition_count_graph(ku_num, *src, tar=None, input_is_file=input_is_file)

    for i in range(ku_num):
        for j in range(i + 1, ku_num):
            count_graph[i][j] = (count_graph[i][j] + count_graph[j][i]) / (
                    abs(count_graph[i][j] - count_graph[j][i]) + 1e-8)
            count_graph[j][i] = count_graph[i][j]

    count_graph = np.asarray(count_graph)  ###asarray：将输入转换为数组。
    _concurrence_graph = softmax(count_graph) * 2

    if tar is not None:
        _output_graph(_concurrence_graph, tar)

    return _concurrence_graph


def concurrence_graph(ku_num, *src, tar):  # pragma: no cover
    warnings.warn("do not use this function due to the lack of support from theory")

    count_graph = [[0] * ku_num for _ in range(ku_num)]

    with fileinput.input(files=src) as f:
        for line in tqdm(f, "constructing concurrence graph"):  ##构建并发图
            if not line.strip():  # pragma: no cover
                continue
            seq = json.loads(line)
            pre = None
            for eid, _ in seq:
                if pre is not None:
                    if eid != pre:
                        try:
                            count_graph[pre][eid] += 1
                            count_graph[eid][pre] += 1
                        except IndexError:
                            print(pre, eid)
                            exit(-1)
                    else:
                        # count_graph[pre][eid] += 1
                        pass
                pre = eid

    count_graph = np.asarray(count_graph)
    _concurrence_graph = softmax(count_graph) * 2
    _output_graph(_concurrence_graph, tar)


def similarity_graph(ku_num, src_graph, tar):
    """construct similarity graph based on transition graph"""   ##基于转移图构建相似图
    with open(src_graph) as f:
        _transitions = json.load(f)

    _transition_graph = [[0] * ku_num for _ in range(ku_num)]

    for id1, id2, value in _transitions:
        _transition_graph[id1][id2] = float(value)

    _transition_graph = np.asarray(_transition_graph)
    _dist_graph = 1 - cdist(_transition_graph, _transition_graph, 'cosine')

    _output_graph(_dist_graph, tar)
