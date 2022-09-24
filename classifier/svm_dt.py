import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

train_path="Dataset/train"
test_path="Dataset/test"
val_path="Dataset/val"

data=[]
labels=['Natural','Synthetic']
path='Dataset' 

for j in labels:
    for img in os.listdir(os.path.join(path,j)):
        imgpath=os.path.join(path,j,img)
        print(imgpath)
        ct_img=cv2.imread(imgpath,0)
        ct_img=cv2.resize(ct_img,(350,350))
        image=np.array(ct_img).flatten()
        label=labels.index(j)
        
        data.append([image,label])


import random

random.shuffle(data)
features=[]
labels=[]

for feature,label in data:
    features.append(feature)
    labels.append(label)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(features,labels,test_size=0.20)

from sklearn.svm import LinearSVC

model=LinearSVC(max_iter=5000)
model.fit(xtrain,ytrain)

import pickle
 
pickle.dump(model, open('./svm_best_model.h5', 'wb'))
# Loading model from disk

saved_model_svm = pickle.load(open('svm_best_model.h5', 'rb'))

#Confution Matrix
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

y_pred = saved_model_svm.predict(xtest)
# print('\t\tConfusion Matrix')
# cf_matrix=confusion_matrix(ytest, y_pred)
# group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
# group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
# labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
# labels = np.asarray(labels).reshape(4,4)
# sns.set(font_scale=1.4)
# sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

# print('\t\tClassification Report')
target_names = ['Natural','Synthetic']
# svm_class_report=classification_report(ytest, y_pred, target_names=target_names,output_dict=True)
# precision_svm,recall_svm,f1_svm=svm_class_report['macro avg']['precision'],svm_class_report['macro avg']['recall'],svm_class_report['macro avg']['f1-score']
# print(classification_report(ytest, y_pred, target_names=target_names))

print(classification_report(ytest,y_pred, target_names=target_names))
svm_accuracy=saved_model_svm.score(xtest,ytest)
print(svm_accuracy)

## Decision tree model

from sklearn import tree

#building the model

model_dt=tree.DecisionTreeClassifier(criterion="entropy")
model_dt.fit(xtrain,ytrain)

import pickle
pickle.dump(model_dt, open('dt_best_model.h5', 'wb'))

# Loading model from disk

saved_model_dt = pickle.load(open('dt_best_model.h5', 'rb'))

y_pred2 = saved_model_dt.predict(xtest)
print(classification_report(ytest,y_pred2,target_names=target_names))
dt_accuracy=saved_model_dt.score(xtest,ytest)
print(dt_accuracy)