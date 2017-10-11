import pandas as pd
from sklearn.metrics import (precision_score, recall_score,
            f1_score)
#首先也是准备好测试数据，进行归一化处理
test_set = pd.read_csv('TestSet.csv')
train_set = pd.read_csv('TrainingSet.csv')
test = test_set.drop(['EbayID', 'QuantitySold', 'SellerName'],axis=1)
test_target = test_set['QuantitySold']
test = scaler.fit_transform(test)
#利用训练号的分类器进行预测
test_pred = clf.predict(test)
print("SGDClassifier training performance on testing dataset:")
print("\tPrecision: %1.3f " % precision_score(test_target, test_pred))
print("\tF1: %1.3f \n" % f1_score(test_target, test_pred))
