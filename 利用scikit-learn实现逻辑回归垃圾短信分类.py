#-*- coding: UTF-8 -*-
# @Time    : 2018/3/10 14:52
# @Author  : xiongzongyang
# @Site    : 
# @File    : 利用scikit-learn实现逻辑回归垃圾短信分类.py
# @Software: PyCharm

import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

#读入数据
def getdata(filepath):
    #创建空的文本列表和标签列表
    text_list=[]
    labels=[]
    #开始读入数据，每一行为一条短信数据，内容依次为，标签，短信原文本，短信分词后的文本
    for line in open(filepath,'r',encoding="utf-8"):
        #将这条短信数据分隔开
        arr=line.rstrip().split('\t')
        #如果不是标准格式的文本，则跳过这条数据
        if len(arr)<3:
            continue

        #修改标签为1，0形式
        if int(arr[0])==1:
            label=1
        elif int(arr[0])==0:
            label=0
        elif int(arr[0])==-1:
            label=0
        else:
            continue

        #将文本和标签分别加入相应的列表
        labels.append(label)
        text_list.append(list(arr[2].split()))
    return text_list,labels

#创建词典
def create_vocab_dict(dataset):
    vocab_dict={}
    #对于数据集中的每一条短信
    for item in dataset:
        #对于每一条短信中的每一个词语，判断其在词汇字典中是否存在
        for word in item:
            if word in vocab_dict:
                vocab_dict[word]+=1
            else:
                vocab_dict[word]=1
    #返回词汇字典，格式为：{“单词”：出现次数}
    return vocab_dict

#词带特征
def BOW_feature(vocablist,one_msg):
    returnvol=[0]*len(vocablist)
    #判断一条短信中的每一个词在词汇表中是否出现
    for word in one_msg:
        if word in vocablist:
            returnvol[vocablist.index(word)]+=1
    return returnvol

#主函数
def main():
    #读取数据
    train_data_path="./open_spam_data/mudou_spam.train"
    test_data_path="./open_spam_data/mudou_spam.test"
    train_data,train_label=getdata(train_data_path)
    text_data,text_label=getdata(test_data_path)

    # 构建词典，去掉低频词
    min_freq=5
    vocab_dict=create_vocab_dict(train_data)
    sorted_vocab_dict=sorted(vocab_dict.items(),key=lambda d:d[1],reverse=True)
    vocab_list=[a[0] for a in sorted_vocab_dict if int(a[1])>min_freq]

    #生成词带特征
    train_x=[]
    #对于训练集中的每一条短信，生成对应的词带特征
    for one_msg in train_data:
        train_x.append(BOW_feature(vocab_list,one_msg))
    text_x=[]
    for one_msg in text_data:
        text_x.append(BOW_feature(vocab_list,one_msg))

    #训练模型
    modeL=LogisticRegression()
    modeL.fit(train_x,train_label)
    pre=modeL.predict(text_x)

    #模型评估
    accuracy_train=modeL.score(train_x,train_label)
    print("训练集上的准确率为：",accuracy_train)
    accuracy_text=modeL.score(text_x,text_label)
    print("测试集上的准确率为：",accuracy_text)

    text_label=np.array(text_label)
    predict_prob_y=modeL.predict_proba(text_x)[:,1]
    predict_prob_y=predict_prob_y.reshape((1,1379))
    text_auc=metrics.roc_auc_score(text_label.flatten(),predict_prob_y.flatten())
    print("测试集AUC：",text_auc)

#程序入口
if __name__ == '__main__':
    main()