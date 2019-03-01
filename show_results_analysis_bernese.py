# -*- coding: utf-8 -*-
"""
Small analysis of Estonian kennelshows using Bernese mountain dogs data from kennelliit.ee and CatBoost algorithm
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
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
np.where(pd.isnull(df)) # check for missing values
df['tulemused'] = df['tulemused'].fillna(value='0') # fill empty values
df["koer"].value_counts() # check dogs' number of records
df = df[df.groupby('koer').koer.transform(len) > 1] # leave out dogs that have only one appearance
df.insert(loc=6, column='judge_country', value=df['kohtunik'], allow_duplicates=True) # copy column with new name to separate judge and country
df['kohtunik'] = df['kohtunik'].str.replace('Kohtunik: ','').str.split(", ").str[0] # replace strings in column and select first element
df['judge_country'] = df['judge_country'].str.replace('Eesti','Estonia').str.split(", ").str[1] # replace strings in column and select last element
df.insert(loc=3, column='gender', value=df['klass'], allow_duplicates=True) # copy column with new name to divide dog class and gender
df['klass'] = df['klass'].str.replace('E ','').str.replace('I ', '') # remove gender info from class
df['gender'] = df['gender'].str.split(" ").str[0] # replace strings in column and select first words as gender (E or I)
df['dogcode'] =  df['koer'].str.split(' ', 1).str[0] # separate dog code to a new column
df['koer'] = df['koer'].str.split(' ', 1).str[1] # separate dog name and leave the doge code out 

# delete inferior classes (repeating show results Kasv, Paar and Järg) and babies and puppies results
df = df[df.klass != 'Kasv'] # delete rows with value 'Kasv'
df = df[df.klass != 'Paar'] # delete rows with value 'Paar'
df = df[df.klass != 'Järg'] # delete rows with value 'Järg'
df = df[df.klass != 'Beebi'] # delete rows with value 'Beebi'
df = df[df.klass != 'Kuts'] # delete rows with value 'Kuts'

# remove some spaces from results and transform string to a list of results
df['tulemused'] = df['tulemused'].str.replace('SP 1','SP1').str.replace('SP 2','SP2').str.replace('SP 3','SP3').str.replace('SP 4','SP4').str.replace('VL 1','VL1').str.replace('VL 2','VL2').str.replace('VL 3','VL3').str.replace('1 EAH','1EAH').str.replace('2 EAH','2EAH').str.replace('3 EAH','3EAH').str.replace('4 EAH','4EAH').str.replace('Jun SERT','JunSERT')
df["tulemused"] = df["tulemused"].str.split(' ')

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
# formatted show results as numeric and categorical variables
num_gradelist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # numeric grades
cat_gradelist = ['Zero', 'EivõiHinnata', 'Rahuldav', 'Hea', 'VägaHea', 'SuurePärane', 'SuurePärane4', 'SuurePärane3', 'SuurePärane2', 'SuurePärane1', 'SertKandidaat', 'Parim4', 'Parim3', 'Parim2', 'VastasSugupooleParim', 'TõuParim'] # categorical grades
gradelist = num_gradelist # decide which to use, numeric or categorical
    
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

# sum dog results and sort values by grade
res_open = df.groupby(['koer', 'gender'])['grade'].sum().reset_index()
res_open = res_open.sort_values('grade', ascending=False).reset_index(drop=True)

# information about judges, where does the judges come from 
df_judges = df.drop_duplicates('naitus') # as there is judge per breed and show, we can leave one row per show
df_judges = df_judges.groupby('judge_country')['kohtunik'].count().reset_index(name='nr_of_judges') # get sum of judges per country
#df_judges.to_csv('judge_countries.csv')

# Finding the missing values
missingvalues = df.isnull().sum()

# convert data to
#df.koer = df.koer.astype(float).fillna(0.0)

""" End of preprocessing, start of CatBoost implementation"""
# make new dataframe with needed only data
df_cb = df.filter(['koer', 'gender', 'kohtunik', 'dogs_per_show', 'grade'], axis=1)

# make training and test set
train, test = train_test_split(df_cb, train_size=0.8)

# Creating a training set for modeling and validation set to check model performance
X = train.drop(['grade'], axis=1)
y = train.grade
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)

y_valid_pred = 0*y
y_test_pred = 0

# Look at the data type of variables 
X.dtypes
categorical_features_indices = np.where(X.dtypes != np.float)[0]

# building model
model=CatBoostRegressor(iterations=53, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)

results = pd.DataFrame()
results['koer'] = test['koer']
results['kohtunik'] = test['kohtunik']
results['grade'] = model.predict(test)
results['grade'] = results['grade'].round() # round values to integers
#results.to_csv("predicted_results.csv")
results = results.sort_values('kohtunik', ascending=False).reset_index(drop=True)
kohtunikuheldus = results.groupby(['kohtunik', 'koer'])['grade'].sum().reset_index().groupby('kohtunik').mean()

""" Cross validation"""
"""from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(model, df_cb, y, cv=6) # change cv for different number of folds
print ('Cross-validated scores:', scores)
predictions = cross_val_predict(model, df_cb, y, cv=6) # Make cross validated predictions
accuracy = metrics.r2_score(y, predictions)
print ('Cross-Predicted Accuracy:', accuracy)"""


from sklearn.model_selection import KFold
# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)

# Run CV

for i, (train_index, test_index) in enumerate(kf.split(train)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
    print( "\nFold ", i)
    
    # Run model for this fold
    fit_model = model.fit( X_train, y_train )
        
    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  Gini = ", eval_gini(y_valid, pred) )
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    y_test_pred += fit_model.predict_proba(X_test)[:,1]
    
y_test_pred /= K  # Average test set predictions
print( "\nGini for full training set:" )
#eval_gini(y, y_valid_pred)


"""Vizualisation"""
# Categorical data vizualisation with seaborn
df_v = df[df.groupby('kohtunik').kohtunik.transform(len) > 30] # select only most frequent judges for plot
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
df_exp.to_csv("data.csv", index=False)

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
df_ml['koer'] = df_ml['koer'].str.replace('LŠVK ', 'LŠVK').str.replace('-','').str.replace('/','').str.replace('.','').str.split(' ').str[0]
#df_ml = pd.get_dummies(data=df_ml, columns=['koer'])
df_ml["koer"] = df_ml["koer"].astype('category')
df_ml["kohtunik"] = df_ml["kohtunik"].astype('category')
df_ml.describe()

from kmodes.kmodes import KModes
# km = KModes(n_clusters=4, init='Cao', n_init=5, verbose=1)
km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(df_ml)

# add claculated clusters todataset
df_ml['clusters'] = clusters
print(km.cluster_centroids_)
print(clusters)
plt.plot(clusters)
plt.scatter(x=df_ml['clusters'], y=df_ml['points'])
"""
