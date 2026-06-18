#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree


# In[2]:


data = pd.read_csv(r"C:\Users\jenish\Downloads\archive (4)\WA_Fn-UseC_-HR-Employee-Attrition.csv")
a = data.info()
print(a)
data = data.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'])
print(data.shape)
print((data["Attrition"]).value_counts())
b = ["Attrition"]
for col in b:
    data[col] = data[col].map({"Yes":1,"No":0})
df = pd.get_dummies(data,drop_first = True,)
df = df.astype(int)
print(df.head())
print(df.describe())
print(df.info())
print(df.shape)
x = df.drop("Attrition",axis = 1)
y = df["Attrition"]
p = x.columns
print(y.name)


# In[4]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
st = StandardScaler()
x_train = st.fit_transform(x_train)
x_test = st.transform(x_test)


# In[5]:


#logisticregression
logistic_model = LogisticRegression()
logistic_model.fit(x_train,y_train)
log_pred = logistic_model.predict(x_test)
cm = confusion_matrix(y_test,log_pred)
print("confusion matrix :",cm)
print("logistic_accuracy:",round(accuracy_score(y_test, log_pred),2))
#knn model
knn_model = KNeighborsClassifier(n_neighbors = 7, metric = "minkowski")
knn_model.fit(x_train,y_train)
knn_pred = knn_model.predict(x_test)
print("knn_accuracy:", round(accuracy_score(y_test, knn_pred),2))


# In[6]:


#naviebias
naive_model = GaussianNB()
naive_model.fit(x_train,y_train)
naive_pred = naive_model.predict(x_test)
print("naive_accuracy:", round(accuracy_score(y_test, naive_pred),2))
#support vector
svm_model = SVC()
svm_model.fit(x_train,y_train)
svm_pred = svm_model.predict(x_test)
print("svm_accuracy:", round(accuracy_score(y_test,svm_pred),2))


# In[14]:


#decision tree
decision1_model = DecisionTreeClassifier(criterion = "entropy",max_depth = 5,random_state = 0,class_weight='balanced')
decision1_model.fit(x_train,y_train)
decision1_pred = decision1_model.predict(x_test)
print("decision1_accuracy:",round(accuracy_score(y_test,decision1_pred),2))
print(decision1_model.get_depth())
print(decision1_model.get_depth())
cm1 = confusion_matrix(y_test,decision1_pred)
print("confusion matrix :",cm1)
z = decision1_model.feature_importances_
print(z)
print("length of the feature:",len(z))
stack =pd.DataFrame({
    "Feature": p,
    "Importance": z
})
sorted_stack = stack.sort_values(by = "Importance",ascending = False)
m = sorted_stack.head(10)
print(m)
n = m.columns
print(n)
x1 = m["Feature"]
y1 = m["Importance"]
plt.title("Top 10 Most Important Features")
plt.xlabel("Important")
plt.ylabel("Feature")
plt.barh(x1,y1,color = "purple")
plt.gca().invert_yaxis()
plt.show()


# In[11]:


print("SVM Classification Report:")
print(classification_report(y_test, svm_pred))
print("Decision Tree (Depth=5) Classification Report:")
print(classification_report(y_test, decision1_pred))
print("knn Classification Report:")
print(classification_report(y_test, knn_pred))
print("navie Classification Report:")
print(classification_report(y_test, naive_pred))
print("logistic Classification Report:")
print(classification_report(y_test, log_pred))


# In[10]:


results = pd.DataFrame({"model":["logistic","knn","svm","naivebias","decision"],"Accuracy":[0.88,0.88,0.90,0.69,0.79],"recall_class1":[0.46,0.13,0.26,0.67,0.38],"F1score_class1":[0.51,0.22,0.40,0.36,0.32]})
print(results)


# In[91]:


plt.figure(figsize = (20,10))
plot_tree(decision1_model,feature_names = p.tolist(),filled = True,max_depth = 3)
plt.show()


# In[94]:


log_employee = pd.DataFrame(
    [[35,800,5,3,4,70,3,2,4,6000,15000,2,15,3,4,1,10,3,3,5,3,1,4,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0]],
    columns=p)
log_employee_scaled = st.transform(log_employee)
log_prediction = logistic_model.predict(log_employee_scaled)
print(log_prediction)
knn_employee = pd.DataFrame(
    [[35,800,5,3,4,70,3,2,4,6000,15000,2,15,3,4,1,10,3,3,5,3,1,4,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0]],
    columns=p)
knn_employee_scaled = st.transform(knn_employee)
knn_prediction = knn_model.predict(knn_employee_scaled)
print(knn_prediction)
navie_employee = pd.DataFrame(
    [[35,800,5,3,4,70,3,2,4,6000,15000,2,15,3,4,1,10,3,3,5,3,1,4,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0]],
    columns=p)
navie_employee_scaled = st.transform(navie_employee)
navie_prediction = naive_model.predict(navie_employee_scaled)
print(navie_prediction)
svm_employee = pd.DataFrame(
    [[35,800,5,3,4,70,3,2,4,6000,15000,2,15,3,4,1,10,3,3,5,3,1,4,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0]],
    columns=p)
svm_employee_scaled = st.transform(svm_employee)
svm_prediction = svm_model.predict(svm_employee_scaled)
print(svm_prediction)


# In[ ]:




