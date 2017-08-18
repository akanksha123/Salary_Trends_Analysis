import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_2011.csv')
df = df.dropna()
df['HireDate'] = pd.to_datetime(df['HireDate'])
df['HireDate'] = df['HireDate'].dt.year
df['Experience'] = df['HireDate'].sub(2011, axis=0)
df['Experience'] = df['Experience'].abs()
df = df[(df.HireDate >= 1972) & (df.Experience <= 2011)]
df.drop_duplicates(inplace = True)
df1 = df.copy()
df1 = df1.drop(['Name', 'AgencyID', 'GrossPay'], axis=1)
df1['AnnualSalary'] = df1['AnnualSalary'].dropna().apply(lambda time: (time[1:]))
df1.columns
## Start of the dataset year = 2012
df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2012.csv')
df = df.dropna()
df['HireDate'] = pd.to_datetime(df['HireDate'])
df['HireDate'] = df['HireDate'].dt.year
df['Experience'] = df['HireDate'].sub(2012, axis=0)
df['Experience'] = df['Experience'].abs()
df = df[(df.Experience >= 0) & (df.Experience <= 40)]
df = df[(df.HireDate >= 1972) & (df.Experience <= 2012)]
df.drop_duplicates(inplace = True)
df.rename(columns={'name': 'Name'}, inplace=True)
df['AnnualSalary'] = df['AnnualSalary'].dropna().apply(lambda time: (time[1:]))
df2 = df.copy()
df2.columns
df2 = df2.drop(['Name', 'AgencyID', 'GrossPay'], axis=1)
## Start for the year dataset - 2013
df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2013.csv')
df = df.dropna()
df['HIRE_DT'] = pd.to_datetime(df['HIRE_DT'])
df['DESCR'] = df['DESCR'].replace(['COMP-Audits (001)', 'COMP-Audits (002)'], ['COMP-Audits', 'COMP-Audits'])
df['DESCR'].replace(regex=True,inplace=True,to_replace=r'\d',value=r'')
df['DESCR'] = df['DESCR'].str.replace(r'(',"")
df['DESCR'] = df['DESCR'].str.replace(r')',"")
df['HIRE_DT'] = df['HIRE_DT'].dt.year
df['Experience'] = df['HIRE_DT'].sub(2013, axis=0)
df['Experience'] = df['Experience'].abs()
df = df[(df.Experience >= 0) & (df.Experience <= 40)]
df = df[(df.HIRE_DT >= 1972) & (df.Experience <= 2013)]
df.rename(columns={'NAME': 'Name','JOBTITLE': 'JobTitle','DESCR': 'Agency','HIRE_DT': 'HireDate','ANNUAL_RT': 'AnnualSalary'}, inplace=True)
df3 = df.copy()
df3.columns
df3 = df3.drop(['Name', 'DEPTID', 'Gross'], axis=1)
df3['AnnualSalary'] = df3['AnnualSalary'].dropna().apply(lambda time: (time[1:]))
sorted(df3['Agency'].unique())
df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2014.csv')
df = df.dropna()
df['HireDate'] = pd.to_datetime(df['HireDate'])
df['HireDate'] = df['HireDate'].dt.year
df['Experience'] = df['HireDate'].sub(2014, axis=0)
df['Experience'] = df['Experience'].abs()
df = df[(df.Experience >= 0) & (df.Experience <= 40)]
df = df[(df.HireDate >= 1972) & (df.Experience <= 2014)]
df.rename(columns={' Name' :'Name'},inplace=True)
df4 = df.copy()
df4.columns
df4 = df4.drop(['Name', 'AgencyID', 'GrossPay'], axis=1)
df4['AnnualSalary'] = df4['AnnualSalary'].dropna().apply(lambda time: (time[1:]))
### Start of the dataset year = 2015
df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2015.csv')
df = df.dropna()
df['HireDate'] = pd.to_datetime(df['HireDate'])
df['HireDate'] = df['HireDate'].dt.year
df['Experience'] = df['HireDate'].sub(2015, axis=0)
df['Experience'] = df['Experience'].abs()
df = df[(df.Experience >= 0) & (df.Experience <= 40)]
df = df[(df.HireDate >= 1972) & (df.Experience <= 2015)]
df5 = df.copy()
df5 = df5.drop(['name', 'AgencyID', 'GrossPay'], axis=1)
df5['Agency'].replace(regex=True,inplace=True,to_replace=r'\d',value=r'')
df5['Agency'] = df5['Agency'].str.replace(r'(',"")
df5['Agency'] = df5['Agency'].str.replace(r')',"")
df5['AnnualSalary'] = df5['AnnualSalary'].dropna().apply(lambda time: (time[1:]))
frames = [df1,df2,df3,df4]
result = pd.concat(frames)
result.columns
result.head(1)
sorted(result['Agency'].unique())
len(result['Agency'].unique())
result['Agency'] = result['Agency'].map(lambda x: x.strip())
sorted(result['Agency'].unique())
result['Agency'] = result['Agency'].replace(['COMP-Communication Ser', 
"COMP-Comptroller's O",
"Civil Rights & Wage Enfor",
'DPW-Solid Waste (wkly)',
'DPW-Solid Waste wkly',
'HLTH-Health Dept Locatio',
'HLTH-Health Dept Location',
'HLTH-Health Dept. Locatio',
'HLTH-Health Dept. Location',
'HLTH-Heatlh',
'HLTH-Heatlh Dept.',
'HLTH-Heatlh Dept. Locatio',
'HLTH-Heatlh Dept. Location',
'HLTH-Heatlh Dept. Locatio',
'HLTH-Heatlh Dept. Location',
'Mayors Office',
'Municipal & Zoning Appeals',
'Municipal & Zoning Appeals 001',
'R&P-Parks (wkly)',
'R&P-Parks wkly',
'R&P-Recreation (part-ti',
'R&P-Recreation part-time',
'TRANS-Cross Guard-S',
'TRANS-Cross Guard-Summer',
'TRANS-Highways (wkly)',
'TRANS-Highways wkly',
'Youth Cust A'],
['COMP-Communication Services',
"COMP-Comptroller's Office",
"Civil Rights & Wage Enforce",
'DPW-Solid Waste',
'DPW-Solid Waste',
'HLTH-Health Department',
'HLTH-Health Department',
'HLTH-Health Department',
'HLTH-Health Department',
'HLTH-Health Department',
'HLTH-Health Department',
'HLTH-Health Department',
'HLTH-Health Department',
'HLTH-Health Department',
'HLTH-Health Department',
'Mayors Office',
'Municipal & Zoning Appeal',
'Municipal & Zoning Appeal',
'R&P-Parks',
'R&P-Parks',
'R&P-Recreation',
'R&P-Recreation',
'TRANS-Crossing Guards',
'TRANS-Crossing Guards',
'TRANS-Highways',
'TRANS-Highways',
'Youth Cust'])

## Considering X = Jobtitle,Agency,Experience and Y = Salary
result = result.dropna()
result2 = result.copy()
X = result2
X.drop(['HireDate'],axis=1,inplace=True)
##X = X[X.Agency.isin(["HEALTH","COMP","TRANS","Elections","Fire Department"])]
X['JobTitle'] = X['JobTitle'].astype('category').cat.codes
X['Agency'] = X['Agency'].astype('category').cat.codes
X = pd.concat([X['JobTitle'], X['Agency'], X['Experience']], axis=1)
Y = result2['AnnualSalary']
X_new = X[:10000]
Y_new = Y[:10000]
print ('about to start exec')
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

# #Logistic regression :
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X_new, Y_new)
# accuracy = model.score(X_new, Y_new) * 100
# print ('accuracy of logistic linear regression : %f' % accuracy)
# # End of logistic regression

### Naive Bayes
# model = GaussianNB()
# model.fit(X_new, Y_new)
# accuracy = model.score(X_new, Y_new) * 100
# print ('accuracy of logistic linear regression : %f' % accuracy)
#  Support vector regression
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
svr_rbf.fit(X_new, Y_new)
accuracy = svr_rbf.score(X_new, Y_new) * 100
print ('accuracy of support vector regression : %f' % accuracy)
# Support vector regression end
# Considering X = Jobtitle,Agency,Experience and Y = Salary

## Considering X = Jobtitle and Y = Salary
# result = result.dropna()
# result2 = result.copy()
# X = result2
# X.drop(['HireDate'],axis=1,inplace=True)
# X.drop(['Agency'],axis=1,inplace=True)
# X.drop(['Experience'],axis=1,inplace=True)
# X['JobTitle'] = X['JobTitle'].astype('category').cat.codes
# X = pd.concat([X['JobTitle']], axis=1)
# Y = result2['AnnualSalary']
# X_new = X[:10000]
# Y_new = Y[:10000]
# ## Support vector regression
# from sklearn.svm import SVR
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
# svr_rbf.fit(X_new, Y_new)
# accuracy = svr_rbf.score(X_new, Y_new) * 100
# print ('accuracy of support vector regression : %f' % accuracy)
#Support vector regression end
# Considering X = Jobtitle and Y = Salary



# ## Considering X = Agency and Y = Salary
# result = result.dropna()
# result2 = result.copy()
# X = result2
# X.drop(['HireDate'],axis=1,inplace=True)
# X.drop(['JobTitle'],axis=1,inplace=True)
# X.drop(['Experience'],axis=1,inplace=True)
# X['Agency'] = X['Agency'].astype('category').cat.codes
# X = pd.concat([X['Agency']], axis=1)
# Y = result2['AnnualSalary']
# X_new = X[:10000]
# Y_new = Y[:10000]
# ## Support vector regression
# from sklearn.svm import SVR
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
# svr_rbf.fit(X_new, Y_new)
# accuracy = svr_rbf.score(X_new, Y_new) * 100
# print ('accuracy of support vector regression : %f' % accuracy)
#Support vector regression end
# Considering X = Jobtitle and Y = Salary



## Considering X = Experience and Y = Salary
# result = result.dropna()
# result2 = result.copy()
# X = result2
# X.drop(['HireDate'],axis=1,inplace=True)
# X.drop(['JobTitle'],axis=1,inplace=True)
# X.drop(['Agency'],axis=1,inplace=True)
# X['Agency'] = X['Experience'].astype('category').cat.codes
# X = pd.concat([X['Experience']], axis=1)
# Y = result2['AnnualSalary']
# X_new = X[:10000]
# Y_new = Y[:10000]
# ## Support vector regression
# from sklearn.svm import SVR
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
# svr_rbf.fit(X_new, Y_new)
# accuracy = svr_rbf.score(X_new, Y_new) * 100
# print ('accuracy of support vector regression : %f' % accuracy)
#Support vector regression end
# Considering X = Jobtitle and Y = Salary

# ####### Considering only five departments
# Z=result.copy();
# Z.dropna()
# Z = Z[Z.Agency.isin(["HEALTH","COMP","TRANS","Elections","Fire Department"])]
# Z.drop(['HireDate'],axis=1,inplace=True)
# Z['JobTitle'] = Z['JobTitle'].astype('category').cat.codes
# Z['Agency'] = Z['Agency'].astype('category').cat.codes
# Z = pd.concat([Z['JobTitle'], Z['Agency'], Z['Experience']], axis=1)
# zcopy = result.copy()
# zcopy = zcopy[zcopy.Agency.isin(["HEALTH","COMP","TRANS","Elections","Fire Department"])]
# zcopy.drop(['Agency'],axis=1,inplace=True)
# zcopy.drop(['Experience'],axis=1,inplace=True)
# zcopy.drop(['JobTitle'],axis=1,inplace=True)
# zcopy.drop(['HireDate'],axis=1,inplace=True)

# from sklearn.svm import SVR
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
# svr_rbf.fit(Z, zcopy)
# accuracy = svr_rbf.score(Z, zcopy) * 100
# print ('accuracy of support vector regression : %f' % accuracy)
# plt.scatter(X_new, Y_new, color='darkorange', label='data')
# plt.hold('on')
# plt.plot(Z, zcopy, color='navy',label='RBF model')
# plt.show()

# In[ ]:
# X_new = X
# Y_new = Y
# len(X_new)
#X_new = pd.concat([X_new['Experience']], axis=1)
#X = pd.concat([pd.get_dummies(result1['Agency'], prefix = 'Agency_'), result1['Experience']], axis=1)
# X_new.columns
# le = preprocessing.LabelEncoder()


# # In[ ]:

# ##dt = DecisionTreeClassifier()
# X_test = df5.copy()
# X_test['JobTitle'] = X_test['JobTitle'].astype('category').cat.codes
# X_test['Agency'] = X_test['Agency'].astype('category').cat.codes
# X_test['Agency'] = X_test['Agency'].astype('category').cat.codes
# X_test.drop(['AnnualSalary'],axis=1,inplace=True)
# Y_test = df5['AnnualSalary']


# # In[ ]:

# X_test.head(2)


# # In[ ]:

# ##Y_test.drop(['JobTitle', 'Agency', 'Experience'],axis=1,inplace=True)
# ##X_test.drop(['AnnualSalary'],axis=1,inplace=True)


# # In[ ]:

# model = LogisticRegression(penalty='l2',C=1)


# # In[ ]:




# # In[ ]:

# model.fit(X,Y)


# # In[ ]:

# print"Logistic regression is %2.2f" % accuracy_score(Y_test,model.predict(X_test))


# # In[ ]:

# Y.head(2)


# # In[ ]:

# X.head(2)


# # In[ ]:

# Y.shape()


# # In[ ]:



