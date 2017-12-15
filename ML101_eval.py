# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:20:58 2016

@author: jf186031
"""




import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

os.getcwd()

df=pd.read_csv('../data/datasets/fake_data_1.csv', sep=',')
df.values


#simple hold-out set
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)

model1_tree = tree.DecisionTreeClassifier()
model1_tree = model1_tree.fit(train[['pred1','pred2','pred3']],train['resp'])

expected = test['resp']
predicted = model_1tree.predict(test[['pred1','pred2','pred3']])

confusion_matrix(expected,predicted)



accuracy = 0
for i in (expected == predicted):
    if i:
        accuracy+=1
accuracy = accuracy/len(expected==predicted)    
print(accuracy)


sum(expected==predicted)

print(classification_report(expected, predicted))
print(confusion_matrix(expected, predicted))



model2_lr = LogisticRegression()
model2_lr = model2_lr.fit(train[['pred1','pred2','pred3']],train['resp'])

expected = test['resp']
predicted = model2_lr.predict(test[['pred1','pred2','pred3']])

confusion_matrix(expected,predicted)


print(classification_report(expected, predicted))
print(confusion_matrix(expected, predicted))





m1_report = classification_report(expected, predicted)

dir(m1_report)



recall(confusion_matrix(expected, predicted))


