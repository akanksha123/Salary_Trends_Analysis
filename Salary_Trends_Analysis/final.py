import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_2011.csv');
df = df.dropna();
df['HireDate'] = pd.to_datetime(df['HireDate']);
df['HireDate'] = df['HireDate'].dt.year;
df['Experience'] = df['HireDate'].sub(2011, axis=0);
df['Experience'] = df['Experience'].abs();
df = df[(df.HireDate >= 1972) & (df.Experience <= 2011)];
df.drop_duplicates(inplace = True);
df1 = df.copy();
df1 = df1.drop(['Name', 'AgencyID', 'GrossPay'], axis=1);
df1['AnnualSalary'] = df1['AnnualSalary'].dropna().apply(lambda time: (time[1:]));
df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2012.csv');
df = df.dropna();
df['HireDate'] = pd.to_datetime(df['HireDate']);
df['HireDate'] = df['HireDate'].dt.year;
df['Experience'] = df['HireDate'].sub(2012, axis=0);
df['Experience'] = df['Experience'].abs();
df = df[(df.Experience >= 0) & (df.Experience <= 40)];
df = df[(df.HireDate >= 1972) & (df.Experience <= 2012)];
df.drop_duplicates(inplace = True);
df.rename(columns={'name': 'Name'}, inplace=True);
df['AnnualSalary'] = df['AnnualSalary'].dropna().apply(lambda time: (time[1:]));
df2 = df.copy();
df2 = df2.drop(['Name', 'AgencyID', 'GrossPay'], axis=1);
df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2013.csv');
df = df.dropna();
df['HIRE_DT'] = pd.to_datetime(df['HIRE_DT']);
df['DESCR'] = df['DESCR'].replace(['COMP-Audits (001)', 'COMP-Audits (002)'], ['COMP-Audits', 'COMP-Audits']);
df['DESCR'].replace(regex=True,inplace=True,to_replace=r'\d',value=r'');
df['DESCR'] = df['DESCR'].str.replace(r'(',"");
df['DESCR'] = df['DESCR'].str.replace(r')',"");
df['HIRE_DT'] = df['HIRE_DT'].dt.year;
df['Experience'] = df['HIRE_DT'].sub(2013, axis=0);
df['Experience'] = df['Experience'].abs();
df = df[(df.Experience >= 0) & (df.Experience <= 40)];
df = df[(df.HIRE_DT >= 1972) & (df.Experience <= 2013)];
df.rename(columns={'NAME': 'Name','JOBTITLE': 'JobTitle','DESCR': 'Agency','HIRE_DT': 'HireDate','ANNUAL_RT': 'AnnualSalary'}, inplace=True);
df3 = df.copy();
df3 = df3.drop(['Name', 'DEPTID', 'Gross'], axis=1);
df3['AnnualSalary'] = df3['AnnualSalary'].dropna().apply(lambda time: (time[1:]));
df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2014.csv');
df = df.dropna();
df['HireDate'] = pd.to_datetime(df['HireDate']);
df['HireDate'] = df['HireDate'].dt.year;
df['Experience'] = df['HireDate'].sub(2014, axis=0);
df['Experience'] = df['Experience'].abs();
df = df[(df.Experience >= 0) & (df.Experience <= 40)];
df = df[(df.HireDate >= 1972) & (df.Experience <= 2014)];
df.rename(columns={' Name' :'Name'},inplace=True);
df4 = df.copy();
df4 = df4.drop(['Name', 'AgencyID', 'GrossPay'], axis=1);
df4['AnnualSalary'] = df4['AnnualSalary'].dropna().apply(lambda time: (time[1:]));
df = pd.read_csv('E:\\Desktop\\DS\\DS\\project\\final\\Baltimore_City_Employee_Salaries_FY2015.csv');
df = df.dropna();
df['HireDate'] = pd.to_datetime(df['HireDate']);
df['HireDate'] = df['HireDate'].dt.year;
df['Experience'] = df['HireDate'].sub(2015, axis=0);
df['Experience'] = df['Experience'].abs();
df = df[(df.Experience >= 0) & (df.Experience <= 40)];
df = df[(df.HireDate >= 1972) & (df.Experience <= 2015)];
df5 = df.copy();
df5 = df5.drop(['name', 'AgencyID', 'GrossPay'], axis=1);
df5['Agency'].replace(regex=True,inplace=True,to_replace=r'\d',value=r'');
df5['Agency'] = df5['Agency'].str.replace(r'(',"");
df5['Agency'] = df5['Agency'].str.replace(r')',"");
df5['AnnualSalary'] = df5['AnnualSalary'].dropna().apply(lambda time: (time[1:]));
frames = [df1,df2,df3,df4];
result = pd.concat(frames);
result['Agency'] = result['Agency'].map(lambda x: x.strip());
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
'Youth Cust']);
result = result.dropna();
result1 = result.copy();
result1.drop(['HireDate'],axis=1,inplace=True);
X = result1.copy();
X['JobTitle'] = X['JobTitle'].astype('category').cat.codes;
X['Agency'] = X['Agency'].astype('category').cat.codes;
X['Agency'] = X['Agency'].astype('category').cat.codes;
X.drop(['AnnualSalary'],axis=1,inplace=True);
Y = result1['AnnualSalary'];
model = LogisticRegression();
model.fit(X,Y);
print ("Model trained");