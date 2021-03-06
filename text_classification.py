import os
import shutil
import zipfile

import jieba

import time
import warnings
import xgboost
import lightgbm
import numpy as np
import pandas as pd
# from keras import models
# from keras import layers
# from keras.utils.np_utils import to_categorical
# from keras.preprocessing.text import Tokenizer
from sklearn import svm
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class Classifier:
    def __init__(self):
        self.filepath = '../data/text_classification.zip'
        self.savepath = '../data/'
        self.delectpath = '../data/text_classification'

    def unzip(self):
        if os.path.exists(self.delectpath):
            print('\n存在该文件夹，正在进行删除，防止解压重命名失败......\n')
            shutil.rmtree(self.delectpath)
        else:
            print('\n不存在该文件夹, 请放心处理......\n')

        # -----------------------------------------
        # 解压并处理中文名字乱码的问题
        z = zipfile.ZipFile(self.filepath, 'r')
        for file in z.namelist():
            # 中文乱码需处理
            # 先使用 cp437 编码，然后再使用 gbk 解码
            filename = file.encode('cp437').decode('gbk')
            # 解压 ZIP 文件
            z.extract(file, self.savepath)
            # 解决乱码问题
            os.chdir(self.savepath)  # 切换到目标目录
            os.rename(file, filename)  # 将乱码重命名文件

    # 读取文件
    def read_text(self, path, text_list):
        '''
        path: 必选参数，文件夹路径
        text_list: 必选参数，文件夹 path 下的所有 .txt 文件名列表
        return: 返回值
            features 文本(特征)数据，以列表形式返回;
            labels 分类标签，以列表形式返回
        '''

        features, labels = [], []
        for text in text_list:
            if text.split('.')[-1] == 'txt':
                try:
                    with open(path + text, encoding='gbk') as fp:
                        features.append(fp.read())  # 特征
                        labels.append(path.split('/')[-2])  # 标签
                except Exception as erro:
                    print('\n>>>发现错误, 正在输出错误信息...\n', erro)

        return features, labels

    # 合并文件
    def merge_text(self, train_or_test, label_name):
        '''
        train_or_test: 必选参数，train 训练数据集 or test 测试数据集
        label_name: 必选参数，分类标签的名字
        return: 返回值
            merge_features 合并好的所有特征数据，以列表形式返回;
            merge_labels   合并好的所有分类标签数据，以列表形式返回
        '''

        print('\n>>>文本读取和合并程序已经启动, 请稍候...')

        merge_features, merge_labels = [], []  # 函数全局变量
        for name in label_name:
            path = '../data/text_classification/' + train_or_test + '/' + name + '/'
            text_list = os.listdir(path)
            features, labels = self.read_text(path=path, text_list=text_list)  # 调用函数
            merge_features += features  # 特征
            merge_labels += labels  # 标签

        # 可以自定义添加一些想要知道的信息
        print('\n>>>你正在处理的数据类型是...\n', train_or_test)
        print('\n>>>[', train_or_test, ']数据具体情况如下...')
        print('样本数量\t', len(merge_features), '\t类别名称\t', set(merge_labels))
        print('\n>>>文本读取和合并工作已经处理完毕...\n')

        return merge_features, merge_labels

    def get_text_classification(self, estimator, X, y, X_test, y_test):
        '''
        estimator: 分类器，必选参数
                X: 特征训练数据，必选参数
                y: 标签训练数据，必选参数
           X_test: 特征测试数据，必选参数
            y_tes: 标签测试数据，必选参数
           return: 返回值
               y_pred_model: 预测值
                 classifier: 分类器名字
                      score: 准确率
                          t: 消耗的时间
                      matrix: 混淆矩阵
                      report: 分类评价函数

        '''
        start = time.time()

        print('\n>>>算法正在启动，请稍候...')
        model = estimator

        print('\n>>>算法正在进行训练，请稍候...')
        model.fit(X, y)
        print(model)

        print('\n>>>算法正在进行预测，请稍候...')
        y_pred_model = model.predict(X_test)
        print(y_pred_model)

        print('\n>>>算法正在进行性能评估，请稍候...')
        score = metrics.accuracy_score(y_test, y_pred_model)
        matrix = metrics.confusion_matrix(y_test, y_pred_model)
        report = metrics.classification_report(y_test, y_pred_model)

        print('>>>准确率\n', score)
        # >>> 准确率
        # 0.89
        print('\n>>>混淆矩阵\n', matrix)
        # >> > 混淆矩阵
        # [[94 6   13  2]
        #  [0  34  2   2]
        #  [3  3   22  3]
        #  [1  1   7   7]]
        print('\n>>>召回率\n', report)
#       >>>召回率
#       precision    recall     f1-score     support
#           0       0.94      0.94      0.94       115
#           1       0.92      0.89      0.91        38
#           2       0.73      0.87      0.79        31
#           3       0.82      0.56      0.67        16
#     accuracy                           0.89       200
#   macro avg        0.85      0.82      0.83       200
# weighted avg       0.89      0.89      0.89       200
        print('>>>算法程序已经结束...')

        end = time.time()
        t = end - start
        print('\n>>>算法消耗时间为：', t, '秒\n')
        classifier = str(model).split('(')[0]

        return y_pred_model, classifier, score, round(t, 2), matrix, report


if __name__ == '__main__':
    classifer = Classifier()
    classifer.unzip()

    train_or_test = 'train'
    label_name = ['女性', '体育', '校园', '文学']
    X_train, y_train = classifer.merge_text(train_or_test, label_name)
    # # 获取测试集
    train_or_test = 'test'
    label_name = ['女性', '体育', '校园', '文学']
    X_test, y_test = classifer.merge_text(train_or_test, label_name)

    X_train_word = [jieba.cut(words) for words in X_train]
    X_train_cut = [' '.join(word) for word in X_train_word]

    print(X_train_cut)

    X_test_word = [jieba.cut(words) for words in X_test]
    X_test_cut = [' '.join(word) for word in X_test_word]

    print(X_test_cut)

    stoplist = [word.strip() for word in open('../data/text_classification/stop/stopword.txt', \
                                              encoding='utf-8').readlines()]
    print(stoplist)

    le = LabelEncoder()

    y_train_le = le.fit_transform(y_train)
    y_test_le = le.fit_transform(y_test)

    print(y_train_le)  # [1 1 1 ... 2 2 2]
    print(y_test_le)  # [0,1,2,3]

    count = CountVectorizer(stop_words=stoplist)

    '''注意：
    这里要先 count.fit() 训练所有训练和测试集，保证特征数一致，
    这样在算法建模时才不会报错
    '''

    count.fit(list(X_train_cut) + list(X_test_cut))
    X_train_count = count.transform(X_train_cut)
    X_test_count = count.transform(X_test_cut)

    X_train_count = X_train_count.toarray()
    X_test_count = X_test_count.toarray()

    print(X_train_count.shape, X_test_count.shape)  # (3305, 23732) (200, 23732)
    print(X_train_count)
    print(X_test_count)

    estimator_list, score_list, time_list = [], [], []

    # K近邻
    print('>>>K-NN算法')
    # KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #                      metric_params=None, n_jobs=None, n_neighbors=5, p=2,
    #                      weights='uniform')
    knc = KNeighborsClassifier(n_neighbors=5)
    result = classifer.get_text_classification(knc, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    print('>>>决策树')
    # DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
    #                        max_depth=None, max_features=None, max_leaf_nodes=None,
    #                        min_impurity_decrease=0.0, min_impurity_split=None,
    #                        min_samples_leaf=1, min_samples_split=2,
    #                        min_weight_fraction_leaf=0.0, presort='deprecated',
    #                        random_state=None, splitter='best')
    dtc = DecisionTreeClassifier()
    result = classifer.get_text_classification(dtc, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    print('>>>多层感知机')
    # MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
    #               beta_2=0.999, early_stopping=False, epsilon=1e-08,
    #               hidden_layer_sizes=(100,), learning_rate='constant',
    #               learning_rate_init=0.001, max_fun=15000, max_iter=200,
    #               momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
    #               power_t=0.5, random_state=None, shuffle=True, solver='adam',
    #               tol=0.0001, validation_fraction=0.1, verbose=False,
    #               warm_start=False)
    mlpc = MLPClassifier()
    result = classifer.get_text_classification(mlpc, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    print('>>>高斯贝叶斯算法')
    # GaussianNB(priors=None, var_smoothing=1e-09)
    gnb = GaussianNB()
    result = classifer.get_text_classification(gnb, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    print('>>>多项式朴素贝叶斯算法')
    # MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    mnb = MultinomialNB()
    result = classifer.get_text_classification(mnb, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    # 逻辑回归
    print('>>>逻辑回归算法')
    # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    #                    intercept_scaling=1, l1_ratio=None, max_iter=100,
    #                    multi_class='auto', n_jobs=None, penalty='l2',
    #                    random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
    #                    warm_start=False)
    lgr = LogisticRegression()
    result = classifer.get_text_classification(lgr, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    print('>>>svm算法')
    svc = svm.SVC()
    # SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)
    result = classifer.get_text_classification(svc, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    print('>>>集成学习算法>>>随机森林算法')
    # RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
    #                        criterion='gini', max_depth=None, max_features='auto',
    #                        max_leaf_nodes=None, max_samples=None,
    #                        min_impurity_decrease=0.0, min_impurity_split=None,
    #                        min_samples_leaf=1, min_samples_split=2,
    #                        min_weight_fraction_leaf=0.0, n_estimators=100,
    #                        n_jobs=None, oob_score=False, random_state=None,
    #                        verbose=0, warm_start=False)
    rfc = RandomForestClassifier()
    result = classifer.get_text_classification(rfc, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    print('>>>集成学习算法>>>自增强算法')
    abc = AdaBoostClassifier()
    # AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
    #                    n_estimators=50, random_state=None)
    result = classifer.get_text_classification(abc, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    print('>>>集成学习算法>>>lightgbm算法')
    # LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
    #                importance_type='split', learning_rate=0.1, max_depth=-1,
    #                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
    #                n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
    #                random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
    #                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
    gbm = lightgbm.LGBMClassifier()
    result = classifer.get_text_classification(gbm, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    print('>>>集成学习算法>>>xgboost算法')
    # XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
    #               colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
    #               importance_type='gain', interaction_constraints=None,
    #               learning_rate=0.300000012, max_delta_step=0, max_depth=6,
    #               min_child_weight=1, missing=nan, monotone_constraints=None,
    #               n_estimators=100, n_jobs=0, num_parallel_tree=1,
    #               objective='multi:softprob', random_state=0, reg_alpha=0,
    #               reg_lambda=1, scale_pos_weight=None, subsample=1,
    #               tree_method=None, validate_parameters=False, verbosity=None)
    xgb = xgboost.XGBClassifier()
    result = classifer.get_text_classification(xgb, X_train_count, y_train_le, X_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

    df = pd.DataFrame()
    df['分类器'] = estimator_list
    df['准确率'] = score_list
    df['消耗时间/s'] = time_list
    print(df)
    '''
            分类器         准确率     消耗时间/s
0     KNeighborsClassifier  0.665   38.12
1   DecisionTreeClassifier  0.735   28.43
2            MLPClassifier  0.880  234.18
3               GaussianNB  0.890    1.73
4            MultinomialNB  0.905    4.67
5       LogisticRegression  0.890    7.01
6                      SVC  0.850  390.35
7   RandomForestClassifier  0.890   45.82
8       AdaBoostClassifier  0.570  124.02
9           LGBMClassifier  0.785    1.07
10           XGBClassifier  0.815  333.82
    
    '''


