import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
from numpy import mean,median
from scipy.stats import mode
import operator
test_set = pd.read_csv('TestSet.csv')
train_set = pd.read_csv('TrainingSet.csv')
test_subset = pd.read_csv('TestSubset.csv')
train_subset = pd.read_csv('TrainingSubset.csv')
#train_set.info()
mean(train_set)
print(mean)
train = train_set.drop(["EbayID", 'QuantitySold', "SellerName"],axis=1)
train_target = train_set['QuantitySold']
_, n_features = train.shape
#print (train[:3])
#print (n_features)
#print (train_target)
df = DataFrame(np.hstack((train,train_target[:, None])), columns=list(range(n_features)) + ["isSold"])
#print(df)

# mean(df)
# median(df)
# mode(df)#计算众数
# #_ = sns.pairplot(df[:50], vars=[2,3,4,10,13], hue="isSold", size=1.5)

plt.figure(figsize=(10,10))
# compute the correlation matrix
corr = df.corr()
#print(corr)
# generate a mask for the upper triangle
mask = np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#print(mask)
# generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax = .3,
                 square=True, xticklabels=5, yticklabels=2,
                linewidths=.5, cbar_kws={"shrink":.5})

plt.yticks(rotation=0)

plt.show()
#利用数据预测拍卖是否会成功






