import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
data=pd.read_csv(r"C:\Users\Mostafa\Downloads\healthcare-dataset-stroke-data.csv")
#removing useless data mainly ID because it has nothing to do with heart strokes
data=data.drop("id",axis=1)
#filling missing bmi values with the most common one
data["bmi"]=data["bmi"].fillna(data["bmi"].median())
# some interesting graphs
sns.barplot(data,y="stroke",color="red")
plt.show()
sns.countplot(data=data,x="smoking_status")
plt.show()
#converting strings to binaries so training is possible
data=pd.get_dummies(data,columns=["gender","ever_married","work_type","Residence_type","smoking_status"])
x=data.drop("stroke",axis=1)
y=data["stroke"]
#a small graph on the side
plt.figure(figsize=(14,10))
sns.heatmap(data.corr(),annot=True,cmap="coolwarm")
plt.tight_layout()
plt.show()
#splitting training data and testing data
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.20,random_state=42)
smote=SMOTE(random_state=42)
x_train,y_train=smote.fit_resample(x_train,y_train)
#training the first model
model1=Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(penalty="l2",C=1))])
model1.fit(x_train,y_train)
y_est_train1=model1.predict(x_train)
y_est_val1=model1.predict(x_val)
y_est_test1=model1.predict_proba(x_val)[:,1]
#testing how well it trained and its performance with new data
print("model 1:")
print("training evaluation:")
print(classification_report(y_train,y_est_train1))
print(confusion_matrix(y_train,y_est_train1))
print("validation evaluation:")
print(classification_report(y_val,y_est_val1))
print(confusion_matrix(y_val,y_est_val1))
print("AUC evaluation:")
print(roc_auc_score(y_val,y_est_test1))
#training the second model
model2=Pipeline([("scaler",StandardScaler()),("model",RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=42,max_depth=4,min_samples_leaf=6,ccp_alpha=0.001))])
model2.fit(x_train,y_train)
y_est_train2=model2.predict(x_train)
y_est_val2=model2.predict(x_val)
y_est_test2=model2.predict_proba(x_val)[:,1]
#testing how well it trained and its performance with new data
print("model 2:")
print("training evaluation:")
print(classification_report(y_train,y_est_train2))
print(confusion_matrix(y_train,y_est_train2))
print("validation evaluation:")
print(classification_report(y_val,y_est_val2))
print(confusion_matrix(y_val,y_est_val2))
print("AUC evaluation:")
print(roc_auc_score(y_val,y_est_test2))
joblib.dump(model1,"model1")
joblib.dump(model2,"model2")