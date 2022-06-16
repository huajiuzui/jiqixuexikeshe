from sklearn import svm
import xlrd2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#获取数据
wb=xlrd2.open_workbook('机器学习数据.xlsx')

sheet1_content1=wb.sheet_by_index(0)

target=[]#标记列表
for i in range(1,53):
   # print(sheet1_content1.cell(i,0).value)
   target.append(round(sheet1_content1.cell(i,0).value,2))

print(target)

data=[]
# print(sheet1_content1.cell(1,6).value)
for i in range(1,53):
    temp=[]
    for j in range(8,57):
        temp.append(round(sheet1_content1.cell(i,j).value,2))
    data.append(temp)
print(data)

x=np.array(data)
y=np.array(target)




train_data,test_data=train_test_split(x,random_state=1,train_size=0.8,test_size=0.2)
train_label,test_label=train_test_split(y,random_state=1,train_size=0.8,test_size=0.2)

classifier=svm.SVC(C=0.65,kernel='rbf',gamma='auto',decision_function_shape='ovo')
classifier.fit(train_data,train_label.ravel())

#模型预测
pre_train=classifier.predict(train_data)
pre_test=classifier.predict(test_data)
print("train准确率：",accuracy_score(train_label,pre_train))
print("test准确率：",accuracy_score(test_label,pre_test))

#混淆矩阵
cm=confusion_matrix(test_label,pre_test)
print("混淆矩阵：",cm)
