# -*- coding: utf-8 -*-
"""
Small analysis of Estonian kennelshows using Bernese mountain dogs data from kennelliit.ee and CatBoost algorithm
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import seaborn as sns

# load and join all data to a single frame
df_2019 = pd.read_csv('dogshows_bernese_est_2019.csv')
df_2018 = pd.read_csv('dogshows_bernese_est_2018.csv')
df_2017 = pd.read_csv('dogshows_bernese_est_2017.csv')
df_2016 = pd.read_csv('dogshows_bernese_est_2016.csv')
df_2015 = pd.read_csv('dogshows_bernese_est_2015.csv')
df_2014 = pd.read_csv('dogshows_bernese_est_2014.csv')
df_2013 = pd.read_csv('dogshows_bernese_est_2013.csv')
frames = [df_2019, df_2018, df_2017, df_2016, df_2015, df_2014, df_2013]
df = pd.concat(frames, join='inner').reset_index(drop=True)
del frames, df_2019, df_2018, df_2017, df_2016, df_2015, df_2014, df_2013

# general preprocessing
df['tulemused'] = df['tulemused'].fillna(value='0') # fill empty values
df["koer"].value_counts() # check dogs' number of records
df = df[df.groupby('koer').koer.transform(len) > 6] # leave out dogs that have low appearance
#df = df[df.groupby('kohtunik').kohtunik.transform(len) > 10] # leave out dogs that have low appearance
df.insert(loc=6, column='judge_country', value=df['kohtunik'], allow_duplicates=True) # copy column with new name to separate judge and country
df['kohtunik'] = df['kohtunik'].str.replace('Kohtunik: ','').str.split(", ").str[0] # replace strings in column and select first element
df['judge_country'] = df['judge_country'].str.replace('Eesti','Estonia').str.split(", ").str[1] # replace strings in column and select last element
df.insert(loc=3, column='gender', value=df['klass'], allow_duplicates=True) # copy column with new name to divide dog class and gender
df['klass'] = df['klass'].str.replace('E ','').str.replace('I ', '') # remove gender info from class
df['gender'] = df['gender'].str.split(" ").str[0] # replace strings in column and select first words as gender (E or I)
df['koer'] =  df['koer'].str.replace('-','').str.replace('/','').str.replace('.','').str.replace("'","").str.replace('LŠVK ', 'LSVK').str.replace('LŠVK', 'LSVK').str.replace('Ž', 'Z') # replace some chars
df['dogcode'] =  df['koer'].str.split(' ', 1).str[0] # separate dog code to a new column
df['koer'] = df['koer'].str.title().str.split(' ', 1).str[1] # separate dog name and leave the doge code out

# delete inferior classes (repeating show results Kasv, Paar and Järg) and babies and puppies results
df = df[df.klass != 'Kasv'] # delete rows with value 'Kasv'
df = df[df.klass != 'Paar'] # delete rows with value 'Paar'
df = df[df.klass != 'Järg'] # delete rows with value 'Järg'
df = df[df.klass != 'Beebi'] # delete rows with value 'Beebi'
df = df[df.klass != 'Kuts'] # delete rows with value 'Kuts'

# remove some spaces from results and transform string to a list of results
df['tulemused'] = df['tulemused'].str.replace('SP 1','SP1').str.replace('SP 2','SP2').str.replace('SP 3','SP3').str.replace('SP 4','SP4').str.replace('VL 1','VL1').str.replace('VL 2','VL2').str.replace('VL 3','VL3').str.replace('1 EAH','1EAH').str.replace('2 EAH','2EAH').str.replace('3 EAH','3EAH').str.replace('4 EAH','4EAH').str.replace('Jun SERT','JunSERT')
df["tulemused"] = df["tulemused"].str.split(' ')

# set data types to category
df['naitus'] = df['naitus'].astype('category')
df['koer'] = df['koer'].astype('category')
df['klass'] = df['klass'].astype('category')
df['gender'] = df['gender'].astype('category')
df['kohtunik'] = df['kohtunik'].astype('category')

# count amounts of participants per class and add column with that information to the class data frame
df_dogs_per_show = df.groupby('naitus').size().reset_index(name='dogs_per_show')
df = pd.merge(df, df_dogs_per_show, on=['naitus'], how='inner')
del df_dogs_per_show

"""Hierarcy of titles by dog classes
All dogs (open class - dogs of Ava,Ch, Noo, Jun, Vet classes):
14 - TP
13 - VSP
12 - PI2+SK, PE2+SK
11 - PI3+SK, PE3+SK
10 - PI4+SK, PE4+SK
9 - SK
8 - SP1
7 - SP2
6 - SP3
5 - SP4
4 - VH
3 - H
2 - R
1 - EVH
0 - 0
"""
# formatted show results as numeric and categorical variables - uncomment the one that will be used
gradelist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # numeric grades
#gradelist = ['muhvigi', 'ei_voi_hinnata', 'rahuldav', 'hea', 'vaga_hea', 'suureparane', 'suureparane_4', 'suureparane_3', 'suureparane_2', 'suureparane_1', 'sert_kandidaat', 'parim_4', 'parim_3', 'parim_2', 'vastassugupoole_parim', 'tou_parim'] # categorical grades
    
# Remap dog result as one grade
def calculate_grade(row):

    grade = gradelist[0]
    sk = False
    
    try:
        if 'SK' in row["tulemused"]: sk = True
        for result in row["tulemused"]:
            if result == 'TP': grade = gradelist[15]
            elif result == 'VSP': grade = gradelist[14]
            elif (result == 'PI2' and sk == True) or (result == 'PE2' and sk == True): grade = gradelist[13]
            elif (result == 'PI3' and sk == True) or (result == 'PE3' and sk == True): grade = gradelist[12]
            elif (result == 'PI4' and sk == True) or (result == 'PE4' and sk == True): grade = gradelist[11]
            elif result == 'SK': grade = gradelist[10]
            elif (result == 'SP1' and sk == False): grade = gradelist[9]
            elif (result == 'SP2' and sk == False): grade = gradelist[8]
            elif (result == 'SP3' and sk == False): grade = gradelist[7]
            elif (result == 'SP4' and sk == False): grade = gradelist[6]
            elif (result == 'SP' and sk == False): grade = gradelist[5]
            elif (result == 'VH' and sk == False): grade = gradelist[4]
            elif (result == 'H' and sk == False): grade = gradelist[3]
            elif (result == 'R' and sk == False): grade = gradelist[2]
            elif (result == 'EVH' and sk == False): grade = gradelist[1]
            elif (result == '0' and sk == False): grade = gradelist[0]
    except:
        return grade

    return grade

# make new column with zero points to all dogs and then run calculations
df["grade"] = gradelist[0]
df['grade'] = df.apply(calculate_grade, axis=1)

""" Small analysis of results """
# sum dog results and sort values by grade
res_open = df.groupby(['koer', 'gender'])['grade'].sum().reset_index()
res_open = res_open.sort_values('grade', ascending=False).reset_index(drop=True)

# information about judges, where does the judges come from 
df_judges = df.drop_duplicates('naitus') # as there is judge per breed and show, we can leave one row per show
df_judges = df_judges.groupby('judge_country')['kohtunik'].count().reset_index(name='nr_of_judges') # get sum of judges per country
#df_judges.to_csv('judge_countries.csv')

""" End of preprocessing, start of CatBoost implementation"""
# Finding the missing values
missingvalues = df.isnull().sum()
df.grade = df.grade.astype(float).fillna(0.0) # convert data type

# Creating a training set for modeling and validation set to check model performance
X = df.drop(['naitus', 'klass', 'gender', 'koeralink', 'tulemused', 'judge_country', 'dogcode', 'dogs_per_show', 'grade'], axis=1)
y = df.grade
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
X_test = df.drop(['naitus', 'klass', 'gender', 'koeralink', 'tulemused', 'judge_country', 'dogcode', 'dogs_per_show', 'grade'], axis=1)

# Look at the data type of variables 
X.dtypes
categorical_features_indices = np.where(X.dtypes != np.float)[0]
X_train.info()

# regression model
#model=CatBoostRegressor(iterations=32, random_seed=63, learning_rate=0.5, depth=12, eval_metric='RMSE')
#model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_validation, y_validation), use_best_model=True, plot=True)
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_validation, model.predict(X_validation))))
#reg_pred = model.predict(X_test) # predict test results

# classification model
model = CatBoostClassifier(iterations=9, learning_rate=0.5, loss_function='MultiClass', custom_loss=['Accuracy']) #custom_loss='AUC'
model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_validation,y_validation), plot=True)
pool = Pool(X, y, cat_features=categorical_features_indices)
scores = cv(pool, model.get_params(), fold_count=5, plot=True)
best_value = np.min(scores['test-MultiClass-mean'])
best_iter = np.argmin(scores['test-MultiClass-mean'])

# print out best scores
print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    np.max(scores['test-Accuracy-mean']),
    scores['test-Accuracy-std'][np.argmax(scores['test-Accuracy-mean'])],
    np.argmax(scores['test-Accuracy-mean'])
))
print('Precise validation accuracy score: {}'.format(np.max(scores['test-Accuracy-mean'])))

# print out feature importance of classification model
feature_importances = model.get_feature_importance(pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))

""" dataframe with test results"""
results = pd.DataFrame()
results['koer'] = X_test['koer']
results['kohtunik'] = X_test['kohtunik']
results['grade_pred'] = model.predict(X_test).astype(np.int) # predicted results of classification model
#results['grade_pred'] = np.rint(model.predict(X_test)) # predicted results of regression model, rounded to integer
results = results.sort_values('kohtunik', ascending=False).reset_index(drop=True)
df_judges_by_grade = results.groupby(['kohtunik', 'koer'])['grade_pred'].sum().reset_index().groupby('kohtunik').mean()

""" dataframe containing all judge and dog combinations and predicted results"""
uniq_dogs = df['koer'].copy()
uniq_judges = df['kohtunik'].copy()
uniq_dogs.drop_duplicates(inplace=True)
uniq_judges.drop_duplicates(inplace=True)
n_rows = uniq_dogs.count() * uniq_judges.count()
df_pred = pd.DataFrame(index=np.arange(n_rows), columns =['koer','kohtunik']) # make dataframe containing all dog and judges combination
df_pred['koer'] = pd.concat([uniq_dogs]*uniq_judges.count(), ignore_index=True)# populate dogs to dataframe
df_pred = df_pred.sort_values('koer', ascending=True).reset_index(drop=True) # sort by dog name
df_pred['kohtunik'] = pd.concat([uniq_judges]*uniq_dogs.count(), ignore_index=True) # now add different judge name to dog
df_pred['grade'] = np.rint(model.predict(df_pred)) # add column with predicted values
df_pred['predicted'] = 1 # 1 - predicted result
df['predicted'] = 0 # 0 - real (not predicted) result

""" dataframe containing real and predicted results"""
df_all = df.drop(['naitus', 'klass', 'gender', 'koeralink', 'tulemused', 'judge_country', 'dogcode', 'dogs_per_show'], axis=1)
df_all = df_pred.append(df_all).reset_index(drop=True)
dup_series = df_all.duplicated(subset=None, keep='last') # keep real value
df_all = df_all[~dup_series]
#df_all.to_csv("bern.csv", index=False) # save some data for javascript vizualisation


"""Vizualisation"""
# Categorical data vizualisation with seaborn
"""df_v = df[df.groupby('kohtunik').kohtunik.transform(len) > 30] # select only most frequent judges for plot
sns.set(style="ticks", color_codes=True)
sns.catplot(x='kohtunik', y='grade', data=df_v, jitter=True);

# crosstab for frequent results
risttabel = pd.crosstab(df_v.koer, df_v.kohtunik, values=df_v.grade, aggfunc='mean').round(0)

# matplotlib vizualisation
colors = np.random.rand(91,91,4).reshape(-1,4)[0:len(values),:]
fig, ax = plt.subplots()
ax.plot(names, values, label="dogs")
ax.scatter(names, values, c=colors)
ax.legend()
plt.ylabel('Tulemus')
plt.xlabel('Kohtunik')
plt.xticks(rotation=90)
plt.show()

# export for javascript vizualisation
df_exp = pd.DataFrame()
df_exp['koer'] = df['koer']
df_exp['grade'] = df['grade']
df_exp.to_csv("data.csv", index=False)"""

""" Below is version for KModes clustering

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

# label encode categorical data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_ml["kohtunikukood"] = le.fit_transform(df_ml["kohtunik"])
df_ml["koerakood"] = le.fit_transform(df_ml["koer"])

# make new dataframe for clustering
df_ml = df.loc[:, ['koer', 'kohtunik', 'grade']]
#df_ml = pd.get_dummies(data=df_ml, columns=['koer'])
df_ml["koer"] = df_ml["koer"].astype('category')
df_ml["kohtunik"] = df_ml["kohtunik"].astype('category')
df_ml.describe()

from kmodes.kmodes import KModes
km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1) # init='Cao'
clusters = km.fit_predict(df_ml)

# add claculated clusters todataset
df_ml['clusters'] = clusters
print(km.cluster_centroids_)
print(clusters)
plt.plot(clusters)
plt.scatter(x=df_ml['clusters'], y=df_ml['points'])
"""
