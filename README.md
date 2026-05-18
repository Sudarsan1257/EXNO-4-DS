# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df=pd.read_csv("income.csv",na_values=[ " ?"])
df
```
<img width="1656" height="681" alt="image" src="https://github.com/user-attachments/assets/5ddd0005-18df-4de0-9ac8-96045c295d82" />
```
df.isnull().sum()
```
<img width="612" height="615" alt="image" src="https://github.com/user-attachments/assets/6907344e-0541-4faf-9b66-d44384c96860" />

```
missing=df[df.isnull().any(axis=1)]
missing
```
<img width="1614" height="606" alt="image" src="https://github.com/user-attachments/assets/af790bb2-3fdc-44c1-a582-291e77503360" />

```
df2=df.dropna(axis=0)
df2
```
<img width="1626" height="553" alt="image" src="https://github.com/user-attachments/assets/90ab315f-b388-4420-857f-ef349bc71a42" />

```
 sal=df["SalStat"]
 df2["SalStat"]=df["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(df2['SalStat'])
```
<img width="1281" height="499" alt="image" src="https://github.com/user-attachments/assets/e33c470e-f9f6-4cea-8368-a30daead4a20" />

```
sal2=df2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="1033" height="559" alt="image" src="https://github.com/user-attachments/assets/9c0136b2-317f-4f00-843e-a4d6cdfcac71" />

```
df2
```
<img width="1521" height="529" alt="image" src="https://github.com/user-attachments/assets/2fd55017-cb21-491a-ac29-33aa8e1d905d" />

```
new_data=pd.get_dummies(df2, drop_first=True)
new_data
```
<img width="1779" height="631" alt="image" src="https://github.com/user-attachments/assets/48870eff-09c9-4c0a-bdce-da157ca976a7" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1796" height="147" alt="image" src="https://github.com/user-attachments/assets/94283170-b434-47a0-9d48-7de083f14378" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1780" height="199" alt="image" src="https://github.com/user-attachments/assets/04f70408-cf90-4d02-8be7-38103cee63ae" />

```
y=new_data['SalStat'].values
print(y)
```
<img width="644" height="161" alt="image" src="https://github.com/user-attachments/assets/25e6bfbc-70a8-4c8b-91bf-62bccbb1a98a" />

```
x=new_data[features].values
print(x)
```
<img width="682" height="267" alt="image" src="https://github.com/user-attachments/assets/36b2fa23-77ff-4fc2-9e36-b62a74f3ca2f" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="932" height="250" alt="image" src="https://github.com/user-attachments/assets/ed8b6d9f-3190-4d5f-953c-7cd3cebaf2e7" />
```
 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)
```
<img width="715" height="222" alt="image" src="https://github.com/user-attachments/assets/ed5a9dd2-7f09-4269-8e58-c1014930b459" />

```

accuracy = accuracy_score(test_y,prediction)
print(accuracy)
```
<img width="577" height="131" alt="image" src="https://github.com/user-attachments/assets/a55d58fc-7d77-4e42-bc6e-f43818658dbd" />
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="826" height="114" alt="image" src="https://github.com/user-attachments/assets/7579a172-bcea-49bf-a6cc-c774199c560a" />

```
df.shape
```
<img width="312" height="89" alt="image" src="https://github.com/user-attachments/assets/31d60390-ab03-48af-8eb7-65e3ff937731" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
df={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
df=pd.DataFrame(df)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="1793" height="553" alt="image" src="https://github.com/user-attachments/assets/9dbfe4a2-61a2-4371-b27a-33235792a963" />

```
from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```
<img width="922" height="358" alt="image" src="https://github.com/user-attachments/assets/3892fb42-52e2-4968-a4d7-1450e7a2762f" />

```
tips.time.unique()
```
<img width="579" height="144" alt="image" src="https://github.com/user-attachments/assets/7ef26441-c10a-4404-ad78-79e697369bf9" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="648" height="183" alt="image" src="https://github.com/user-attachments/assets/c348fa21-2aef-4799-8523-86bde7421ce1" />
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="678" height="166" alt="image" src="https://github.com/user-attachments/assets/f238e8aa-ff00-4c4e-9ad2-ecc57434223c" />

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
