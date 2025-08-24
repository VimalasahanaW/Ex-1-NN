<H3>ENTER YOUR NAME : VIMALA SAHANA W</H3>
<H3>ENTER YOUR REGISTER NO:212223040241</H3>
<H3>EX. NO.1</H3>
<H3>DATE:24-08-25</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM AND OUTPUT:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
```
df=pd.read_csv("/content/Churn_Modelling.csv")
df
```
<img width="1157" height="399" alt="image" src="https://github.com/user-attachments/assets/9786d5a4-5e22-464f-95b1-c2006dcf70d1" />

```
df.isnull().sum()
```
<img width="155" height="489" alt="image" src="https://github.com/user-attachments/assets/6c1030b8-e36c-409c-80ba-1411a7ba9df8" />

```
df.duplicated()
```
<img width="164" height="425" alt="image" src="https://github.com/user-attachments/assets/a9318e0e-ebf6-43c9-93f4-ba1a707e417b" />

```
print(df['CreditScore'].describe())
```
<img width="720" height="445" alt="image" src="https://github.com/user-attachments/assets/e1c280eb-add5-4cc0-bbc8-40de1a6ea532" />

```
df.info()
```
<img width="643" height="663" alt="image" src="https://github.com/user-attachments/assets/23720078-9b10-4551-b72c-de600db01be2" />


```
scaler=MinMaxScaler()
df['CreditScore']=scaler.fit_transform(df[['CreditScore']])
df
```
<img width="1244" height="422" alt="image" src="https://github.com/user-attachments/assets/6fb469d6-1a1e-432b-97ee-5ecc2d2e39e3" />

```
X=df.iloc[:, :-1].values
X
```
<img width="800" height="191" alt="image" src="https://github.com/user-attachments/assets/e71364cf-ec75-4bde-958f-4704942eac65" />

```
y=df.iloc[:,-1].values
y
```
<img width="349" height="54" alt="image" src="https://github.com/user-attachments/assets/56e53144-7fe9-4b7b-a7ab-5fe42ca2ec7b" />


```
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
print(X_train)
```
<img width="474" height="184" alt="image" src="https://github.com/user-attachments/assets/223551d9-4af5-45cf-95dd-c601f1352d70" />

```
print(len(X_train))
```
<img width="80" height="36" alt="image" src="https://github.com/user-attachments/assets/0451ff35-6347-4bc7-a2bd-ceb3dadb7c3d" />

```
print(X_test)
```
<img width="524" height="193" alt="image" src="https://github.com/user-attachments/assets/e1c94013-3212-425e-a3c2-c3dd96a42cef" />

```
print(len(X_test))
```
<img width="113" height="40" alt="image" src="https://github.com/user-attachments/assets/954c6dfd-1d1b-4313-980a-500f489082da" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


.


