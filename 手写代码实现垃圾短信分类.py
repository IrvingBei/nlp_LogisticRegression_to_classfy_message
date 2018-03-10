#-*- coding: UTF-8 -*-
# @Time    : 2018/3/10 10:59
# @Author  : xiongzongyang
# @Site    : 
# @File    : 手写代码实现垃圾短信分类.py
# @Software: PyCharm

import pandas as pd
from pandas import Series, DataFrame

import os
import sys
import jieba
from sklearn import metrics
import numpy as np
from random import randrange
import time


class logistic_regression():
    """
       手写逻辑回归，用于二分类。
       注意事项：
           1）sklearn自带的lr可以做多分类，这里我们实现的版本只能做二分类
           2）这里我们支持的label是1,0。如果您的label是1,-1，请先通过预处理进行转换
    """

    def fit(self, train_X_in, train_Y, learning_rate=0.5, batch_size=20, eps=1e-3):
        #case_cnt：表示短信的条数， feature_cnt表示特征的数目
        case_cnt, feature_cnt = np.array(train_X_in).shape
        #设置初始化参数theta
        self.theta = np.zeros([feature_cnt + 1, 1])
        # np.c_将多个对象连接到第二个数轴上
        train_X = np.c_[train_X_in, np.ones(case_cnt, )]

        step = 0
        max_iteration_times = sys.maxsize
        past_best_likelihood = -sys.maxsize - 1
        past_step = 0
        stay_times = 0
        X = train_X.T
        while step < max_iteration_times:
            #总的训练数据是case_cnt，以batch_size大小来进行一次训练
            for b in range(0, case_cnt, batch_size):
                #计算预测值
                pred = 1.0 / (1 + np.exp(-self.theta.T.dot(X[:, b: b + batch_size])))
                #更新theta
                self.theta = self.theta + learning_rate * 1.0 / case_cnt * (train_Y[b: b + batch_size] - pred).dot(X[:, b: b + batch_size].T).T

            pred = 1.0 / (1 + np.exp(-self.theta.T.dot(X)))
            likelihood = 1.0 / case_cnt * sum((train_Y * np.log(pred) +(1 - train_Y) * np.log(1 - pred)).flatten())
            if likelihood > past_best_likelihood + eps:
                past_best_likelihood = likelihood
                past_step = step
            elif step - past_step >= 20:
                sys.stderr.write("training finished. total step %s: %.6f\n" % (step, likelihood))
                break
            if step % 1000 == 0:
                sys.stderr.write("step %s: %.6f\n" % (step, likelihood))
            step += 1
        return 1

    def predict_proba(self, X):
        case_cnt = X.shape[0]
        X = np.c_[X, np.ones(case_cnt, )]
        return 1. / (1 + np.exp(-self.theta.T.dot(X.T)))

    def predict(self, X):
        case_cnt = X.shape[0]
        X = np.c_[X, np.ones(case_cnt, )]
        prob = 1. / (1 + np.exp(-self.theta.T.dot(X.T)))
        return (prob >= 0.5).astype(np.int32).flatten()

    def score(self, X, label):
        """ Returns the mean accuracy on the given test data and labels """
        pred = self.predict(X)
        return sum((pred == label).astype(np.int32).flatten()) * 1.0 / pred.shape[0]


# 创建词汇表，并统计各个词出现的次数
def create_vocab_dict(dataSet):
    #这个词汇字典是针对整个短信数据集的
    vocab_dict = {}
    # 对于每一条短信
    for document in dataSet:
        # 判断这条短信中的每一个单词是否在词汇表中出现过，如果没有出现过，则加入字典，如果出现过，则相应的次数加一
        for term in document:
            if term in vocab_dict:
                vocab_dict[term] += 1
            else:
                vocab_dict[term] = 1
    #返回字典，格式为{“单词”：出现次数}
    return vocab_dict


def BOW_feature(vocabList, inputSet):
    #初始化词带，得到一个长度为单词表长度，元素为全0的列表
    #参数inputset为一条短信
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            #如果当前词在字典中，则将词带对应位置+1
            returnVec[vocabList.index(word)] += 1
    return returnVec


def get_dataset(data_path):
    # 数据由三部分组成，第一部分是标签，第二部分是短信内容，第三部分是分词后的短信内容
    text_list = []
    labels = []
    for line in open(data_path, "r",encoding="utf-8"):
        #先按照tab将串分隔开，然后再去掉多余的空白符
        arr = line.rstrip().split('\t')
        #如果读取的该行不是标准的格式，则丢弃这行数据，处理下一条
        if len(arr) < 3:
            continue

        # 把标签从(1,-1)改为(1,0)
        if int(arr[0]) == 1:
            label = arr[0]
        elif int(arr[0]) == -1:
            label = 0
        #---------------------------------------
        elif int(arr[0])==0:
            label=0
        else:  # illegal label
            continue
        text = arr[2]
        # 将每条短信以空格为分隔符分开成列表
        text_list.append(list(text.split()))
        labels.append(int(label))
    # text_list是一个二维列表，其中的每一个元素是一条短信分词后所形成的列表
    # print(text_list)
    return text_list, labels


if __name__ == "__main__":

    # 数据路径
    train_file_path = './open_spam_data/mudou_spam.train'
    test_file_path = './open_spam_data/mudou_spam.test'
    #获取数据
    train_data, train_label = get_dataset(train_file_path)
    test_data, test_label = get_dataset(test_file_path)

    # 构造词典
    min_freq = 5
    vocab_dict = create_vocab_dict(train_data)
    # 将词汇表按照词汇多少来排序
    sorted_vocab_list = sorted(vocab_dict.items(), key=lambda d: d[1], reverse=True)
    # 删除那些稀少词，生成的只是一个单词表，没有出现的次数
    vocab_list = [v[0] for v in sorted_vocab_list if int(v[1]) > min_freq]

    # 生成文本的词袋（BOW）特征
    train_X = []
    # 对于每一个短信，生成一条词带记录
    for one_msg in train_data:
        train_X.append(BOW_feature(vocab_list, one_msg))

    test_X = []
    for one_msg in test_data:
        test_X.append(BOW_feature(vocab_list, one_msg))

    #将标签列表转化为numpy数组
    test_label = np.array(test_label)
    train_label = np.array(train_label)
    #将词带特征转化为numpy数组
    train_X = np.array(train_X)
    test_X = np.array(test_X)

    # 训练模型
    model = logistic_regression()
    model.fit(train_X[:100], train_label[:100], 0.6, 100, 1e-4)

    # 模型评估
    accuracy_train = model.score(train_X, train_label)
    print('训练集accuracy:', accuracy_train)
    accuracy_test = model.score(test_X, test_label)
    print('测试集accuracy: ', accuracy_test)

    # pred = model.predict(test_X)
    print(type(test_label))
    print(test_label.shape)
    print(test_label)
    predict_prob_y = model.predict_proba(test_X)
    print(type(predict_prob_y))
    print(predict_prob_y.shape)
    print(predict_prob_y)
    test_auc = metrics.roc_auc_score(test_label.flatten(), predict_prob_y.flatten())
    print('测试集AUC:', test_auc)