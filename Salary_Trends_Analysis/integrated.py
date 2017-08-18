
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[2]:

## Dataset year - 2011


# In[3]:

df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_2011.csv')


# In[4]:

df = df.dropna()


# In[5]:

df['HireDate'] = pd.to_datetime(df['HireDate'])


# In[6]:


df['HireDate'] = df['HireDate'].dt.year


# In[7]:

df['Experience'] = df['HireDate'].sub(2011, axis=0)
df['Experience'] = df['Experience'].abs()


# In[8]:

df = df[(df.HireDate >= 1972) & (df.Experience <= 2011)]


# In[9]:

df.drop_duplicates(inplace = True)


# In[10]:

df1 = df.copy()


# In[11]:

df1 = df1.drop(['Name', 'AgencyID', 'GrossPay'], axis=1)


# In[12]:

df1['AnnualSalary'] = df1['AnnualSalary'].dropna().apply(lambda time: (time[1:]))


# In[13]:

df1.columns


# In[14]:

## Start of the dataset year = 2012


# In[15]:

df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2012.csv')


# In[16]:

df = df.dropna()


# In[17]:

df['HireDate'] = pd.to_datetime(df['HireDate'])


# In[18]:

df['HireDate'] = df['HireDate'].dt.year


# In[19]:

df['Experience'] = df['HireDate'].sub(2012, axis=0)
df['Experience'] = df['Experience'].abs()


# In[20]:

df = df[(df.Experience >= 0) & (df.Experience <= 40)]


# In[21]:

df = df[(df.HireDate >= 1972) & (df.Experience <= 2012)]


# In[22]:

df.drop_duplicates(inplace = True)


# In[23]:

df.rename(columns={'name': 'Name'}, inplace=True)


# In[24]:

df['AnnualSalary'] = df['AnnualSalary'].dropna().apply(lambda time: (time[1:]))


# In[25]:

df2 = df.copy()


# In[26]:

df2.columns


# In[27]:

df2 = df2.drop(['Name', 'AgencyID', 'GrossPay'], axis=1)


# In[28]:

## Start for the year dataset - 2013


# In[ ]:




# In[ ]:




# In[29]:

df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2013.csv')


# In[30]:

df = df.dropna()


# In[31]:

df['HIRE_DT'] = pd.to_datetime(df['HIRE_DT'])


# In[32]:

df['DESCR'] = df['DESCR'].replace(['COMP-Audits (001)', 'COMP-Audits (002)'], ['COMP-Audits', 'COMP-Audits'])


# In[33]:

df['DESCR'].replace(regex=True,inplace=True,to_replace=r'\d',value=r'')
df['DESCR'] = df['DESCR'].str.replace(r'(',"")


# In[34]:

df['DESCR'] = df['DESCR'].str.replace(r')',"")


# In[35]:

df['HIRE_DT'] = df['HIRE_DT'].dt.year


# In[36]:

df['Experience'] = df['HIRE_DT'].sub(2013, axis=0)
df['Experience'] = df['Experience'].abs()


# In[37]:

df = df[(df.Experience >= 0) & (df.Experience <= 40)]


# In[38]:

df = df[(df.HIRE_DT >= 1972) & (df.Experience <= 2013)]


# In[39]:

df.rename(columns={'NAME': 'Name','JOBTITLE': 'JobTitle','DESCR': 'Agency','HIRE_DT': 'HireDate','ANNUAL_RT': 'AnnualSalary'}, inplace=True)


# In[40]:

df3 = df.copy()


# In[41]:

df3.columns


# In[42]:

df3 = df3.drop(['Name', 'DEPTID', 'Gross'], axis=1)


# In[43]:

df3['AnnualSalary'] = df3['AnnualSalary'].dropna().apply(lambda time: (time[1:]))


# In[44]:

sorted(df3['Agency'].unique())


# In[45]:

df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2014.csv')


# In[46]:

df = df.dropna()


# In[47]:

df['HireDate'] = pd.to_datetime(df['HireDate'])


# In[48]:

df['HireDate'] = df['HireDate'].dt.year


# In[49]:

df['Experience'] = df['HireDate'].sub(2014, axis=0)
df['Experience'] = df['Experience'].abs()


# In[50]:

df = df[(df.Experience >= 0) & (df.Experience <= 40)]


# In[51]:

df = df[(df.HireDate >= 1972) & (df.Experience <= 2014)]


# In[52]:

df.rename(columns={' Name' :'Name'},inplace=True)


# In[53]:

df4 = df.copy()


# In[54]:

df4.columns


# In[55]:

df4 = df4.drop(['Name', 'AgencyID', 'GrossPay'], axis=1)


# In[56]:

df4['AnnualSalary'] = df4['AnnualSalary'].dropna().apply(lambda time: (time[1:]))


# In[57]:

### Start of the dataset year = 2015


# In[58]:

df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2015.csv')


# In[59]:

df = df.dropna()


# In[60]:

df['HireDate'] = pd.to_datetime(df['HireDate'])


# In[61]:

df['HireDate'] = df['HireDate'].dt.year


# In[62]:

df['Experience'] = df['HireDate'].sub(2015, axis=0)
df['Experience'] = df['Experience'].abs()


# In[63]:

df = df[(df.Experience >= 0) & (df.Experience <= 40)]


# In[64]:

df = df[(df.HireDate >= 1972) & (df.Experience <= 2015)]


# In[65]:

df5 = df.copy()


# In[66]:

df5 = df5.drop(['name', 'AgencyID', 'GrossPay'], axis=1)


# In[67]:

df5['Agency'].replace(regex=True,inplace=True,to_replace=r'\d',value=r'')
df5['Agency'] = df5['Agency'].str.replace(r'(',"")


# In[68]:

df5['Agency'] = df5['Agency'].str.replace(r')',"")


# In[69]:

df5['AnnualSalary'] = df5['AnnualSalary'].dropna().apply(lambda time: (time[1:]))


# In[70]:

##frames = [df1,df2,df3,df4,df5]
frames = [df1,df2,df3,df4]


# In[71]:

result = pd.concat(frames)


# In[ ]:

result.columns


# In[ ]:

result.head(1)


# In[ ]:

sorted(result['Agency'].unique())
len(result['Agency'].unique())
#len(result['JobTitle'].unique())
#len(result['HireDate'].unique())


# In[ ]:

##result['Agency'] = result['Agency'].str.replace(r' ',"")
result['Agency'] = result['Agency'].map(lambda x: x.strip())


# In[ ]:

sorted(result['Agency'].unique())


# In[ ]:

##sorted(df1['Agency'].unique())


# In[ ]:

##sorted(df2['Agency'].unique())


# In[ ]:

##sorted(df3['Agency'].unique())


# In[ ]:

##sorted(df4['Agency'].unique())


# In[ ]:

##sorted(df5['Agency'].unique())


# In[ ]:

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


# In[ ]:

##result['Agency'] = result['Agency'].replace(['COMP-Communication Ser'], ['COMP-Communication Services'])


# In[ ]:

##result['Agency'] = result['Agency'].replace(["COMP-Comptroller's O"], ["COMP-Comptroller's Office"])


# In[ ]:

# result['Agency'].head(5)


# In[ ]:

# sorted(result['Agency'].unique())
# len(result['Agency'].unique())


# # In[ ]:

# sorted(df1['AnnualSalary'].unique())


# # In[ ]:

# sorted(df5['AnnualSalary'].unique())


# # In[ ]:

# len(result)


# In[ ]:

result = result.dropna()


# In[ ]:

# len(result)


# # In[ ]:

# result.head(10)


# In[ ]:

#result1 = result.copy()


# In[ ]:




# In[ ]:

# result1.dtypes


# In[ ]:

#result1.drop(['HireDate','JobTitle'],axis=1,inplace=True)


# In[ ]:

# result1.dtypes


# In[ ]:

#result1['JobTitle'].value_counts()


# In[ ]:

#X = result1.copy()

# result1 = result1[result1['Agency'] == 'HLTH-Health Department']
# X['JobTitle'] = X['JobTitle'].astype('category').cat.codes
# X['Agency'] = X['Agency'].astype('category').cat.codes
# X = pd.concat([pd.get_dummies(result1['Agency'], prefix = 'Agency_'), result1['Experience']], axis=1)
# X = pd.concat([result1['Agency'], result1['Experience']], axis=1)
# Y = result1['AnnualSalary']
##X = pd.get_dummies(X['Agency'], prefix = 'Agency_')
##X['Agency'] = X['Agency'].astype('category').cat.codes
# X.columns
# Y.head(1)
# len(X)
#result1.columns


# In[ ]:

result2 = result.copy()
X = result2
X.drop(['HireDate'],axis=1,inplace=True)
X['JobTitle'] = X['JobTitle'].astype('category').cat.codes
X['Agency'] = X['Agency'].astype('category').cat.codes
 
X = pd.concat([X['JobTitle'], X['Agency'], X['Experience']], axis=1)
Y = result2['AnnualSalary']


# In[ ]:

# result1.columns


# In[ ]:

#X = pd.concat([X, pd.get_dummies(['District'], prefix = 'District')], axis=1)


# In[ ]:

#X.drop(['AnnualSalary'],axis=1,inplace=True)


# In[ ]:

# X.columns


# # In[ ]:

# X.dtypes


# # In[ ]:

# #Y = result1['AnnualSalary']


# # In[ ]:

# X.columns


# In[ ]:

# Y.head(2)


# # In[ ]:

# len(X[X['Agency__HLTH-Health Department']==1])


# # In[ ]:

# Y.dtypes
#X_new = X[X['Agency__HLTH-Health Department']==1]
#X = X.sample(frac=1).reset_index(drop=True)

X_new = X[:10000]
Y_new = Y[:10000]
# X_new = X
# Y_new = Y
# len(X_new)
#X_new = pd.concat([X_new['Experience']], axis=1)
#X = pd.concat([pd.get_dummies(result1['Agency'], prefix = 'Agency_'), result1['Experience']], axis=1)
# X_new.columns


# In[ ]:
print ('about to start exec')

from sklearn import preprocessing
from sklearn import cross_validation
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X_new, Y_new)
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
svr_rbf.fit(X_new, Y_new)
# model.fit(X_new, Y_new)
print ('results: ' + str(svr_rbf.score(X_new, Y_new)))
plt.scatter(X_new, Y_new, color='darkorange', label='data')
plt.hold('on')
plt.plot(Xnew, y_rbf, color='navy', lw=lw, label='RBF model')
plt.show()

# In[ ]:

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



