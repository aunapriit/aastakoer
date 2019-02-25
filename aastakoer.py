# -*- coding: utf-8 -*-
"""
Calculate points to each dog and make ranking lists by dog classes as given at https://esakt.eu/images/Aparim15a.pdf.
"""
import pandas as pd

# Importing the dataset
df = pd.read_csv('dogshows_bernese_est_2018.csv')

# make copy of date column and extract dates from it
df.insert(loc=1, column='date', value=[str[-10:] for str in df['naitus']], allow_duplicates=True)
try:
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
except ValueError:
    print('unable to convert to dates')
    pass

# processing judge information - separation of judge name and country
df.insert(loc=7, column='judge_Country', value=df['kohtunik'], allow_duplicates=True) # copy column with new name
df['kohtunik'] = df['kohtunik'].str.replace('Kohtunik: ','').str.split(", ").str[0] # replace strings in column and select first element
df['judge_Country'] = df['judge_Country'].str.replace('Eesti','Estonia').str.split(", ").str[1] # replace strings in column and select last element
df['judge_Country'] = df['judge_Country'].astype('category')

# divide dog class and gender to a separate columns
df.insert(loc=3, column='gender', value=df['klass'], allow_duplicates=True) # copy column with new name
df['klass'] = df['klass'].str.replace('E ','').str.replace('I ', '') # remove gender info from class
df['gender'] = df['gender'].str.split(" ").str[0] # replace strings in column and select last element

# delete inferior classes (repeating show results)
df = df[df.klass != 'Kasv'] # delete rows with value 'Kasv'
df = df[df.klass != 'Paar'] # delete rows with value 'Paar'

# delete unofficial show results
df = df[~df.naitus.str.contains("EKL Jõulushow")].reset_index(drop=True)

# remove some spaces from results and transform string to a list of results
df['tulemused'] = df['tulemused'].str.replace('SP 1','SP1').str.replace('SP 2','SP2').str.replace('SP 3','SP3').str.replace('SP 4','SP4').str.replace('VL 1','VL1').str.replace('VL 2','VL2').str.replace('VL 3','VL3').str.replace('1 EAH','1EAH').str.replace('2 EAH','2EAH').str.replace('3 EAH','3EAH').str.replace('4 EAH','4EAH').str.replace('Jun SERT','JunSERT')
df["tulemused"] = df["tulemused"].str.split(' ')

# separate dog classes to 4 levels
df_babies = df[(df['klass'] == 'Beebi')]
df_puppies = df[(df['klass'] == 'Kuts')]
df_juniors = df[(df['klass'] == 'Jun')]
df_open = df[(df['klass'] == 'Ava') | (df['klass'] == 'Ch') | (df['klass'] == 'Noo') | (df['klass'] == 'Jun') | (df['klass'] == 'Vet')] # Jun ,ust be included

# count amounts of participants per class and add column with that information to the class data frame
df_babies_per_show = df_babies.groupby('naitus').size().reset_index(name='babies_per_show')
df_babies = pd.merge(df_babies, df_babies_per_show, on=['naitus'], how='inner')
df_puppies_per_show = df_puppies.groupby('naitus').size().reset_index(name='puppies_per_show')
df_puppies = pd.merge(df_puppies, df_puppies_per_show, on=['naitus'], how='inner')
df_juniors_per_show = df_juniors.groupby('naitus').size().reset_index(name='juniors_per_show')
df_juniors = pd.merge(df_juniors, df_juniors_per_show, on=['naitus'], how='inner')
df_open_per_show = df_open.groupby('naitus').size().reset_index(name='dogs_per_show')
df_open = pd.merge(df_open, df_open_per_show, on=['naitus'], how='inner')

"""Hierarcy of titles by dog classes

All dogs (open class - dogs of Ava,Ch, Noo, Jun, Vet classes):
TP
VSP
PI2 + SK, PE2 + SK
PI3 + SK, PE3 + SK
PI4 + SK, PE4 + SK

Junior dogs(dogs of Jun class):
TPJ
VSPJ
SP2 + SK
SP3 + SK
SP4 + SK

Puppies class (dogs of Kuts, Beebi classes):
TPK või TPB
VSPK või VSPB
2EAH
3EAH
4EAH"""

# Calculate points to give
def calculate_points(row):

    bonus = 0
    points = 0
    sk = False # decide if dog is sertificate candidate
    
    # look for a dog count per show
    if hasattr(row, 'dogs_per_show'):
        dogscount = row['dogs_per_show']
        klass = 1
    elif hasattr(row, 'juniors_per_show'):
        dogscount = row['juniors_per_show']
        klass = 2
    elif hasattr(row, 'puppies_per_show'):
        dogscount = row['puppies_per_show']
        klass = 3
    elif hasattr(row, 'babies_per_show'):
        dogscount = row['babies_per_show']
        klass = 4
            
    # choose bonus points depending on amount of dogs attended
    if dogscount >= 1 and dogscount <= 5: bonus = 0
    elif dogscount >= 6 and dogscount <= 10: bonus = 1
    elif dogscount >= 11 and dogscount <= 15: bonus = 2
    elif dogscount >= 16 and dogscount <= 25: bonus = 3
    elif dogscount >= 26 and dogscount <= 50: bonus = 4
    elif dogscount >= 51 and dogscount <= 75: bonus = 5
    elif dogscount >= 76 and dogscount <= 100: bonus = 6
    elif dogscount >= 101 and dogscount <= 150: bonus = 7
    elif dogscount > 150: bonus = 8
    
    try:
        if 'SK' in row["tulemused"]: sk = True
        
        for result in row["tulemused"]:
            if (result == 'TP' and klass == 1) or (result == 'TPJ' and klass == 2) or (result == 'TPK' and klass == 3) or (result == 'TPB' and klass == 4):
                points = 10 + bonus
            elif (result == 'VSP' and klass == 1) or (result == 'VSPJ' and klass == 2) or (result == 'VSPK' and klass == 3) or (result == 'VSPB' and klass == 4):
                points = 9 + bonus
            elif (result == 'PI2' and sk == True and klass == 1) or (result == 'PE2' and sk == True and klass == 1) or (result == '2EAH' and (klass == 3 or klass == 4)) or (result == 'SP2' and sk == True and klass == 2):
                points = 7 + bonus
            elif (result == 'PI3' and sk == True and klass == 1) or (result == 'PE3' and sk == True and klass == 1) or (result == '3EAH' and (klass == 3 or klass == 4)) or (result == 'SP3' and sk == True and klass == 2):
                points = 6 + bonus
            elif (result == 'PI4' and sk == True and klass == 1) or (result == 'PE4' and sk == True and klass == 1) or (result == '4EAH' and (klass == 3 or klass == 4)) or (result == 'SP4' and sk == True and klass == 2):
                points = 5 + bonus
    except:
        return points

    return points

# make new column with zero points to all dogs and then run calculations
df_open["points"] = 0
df_open['points'] = df_open.apply(calculate_points, axis=1)
df_juniors["points"] = 0
df_juniors['points'] = df_juniors.apply(calculate_points, axis=1)
df_puppies["points"] = 0
df_puppies['points'] = df_puppies.apply(calculate_points, axis=1)
df_babies["points"] = 0
df_babies['points'] = df_babies.apply(calculate_points, axis=1)

# filter out only 5 biggest results per dog
df_open = df_open.sort_values(['koer','points'],ascending=False).groupby('koer').head()
df_juniors = df_juniors.sort_values(['koer','points'],ascending=False).groupby('koer').head()
df_puppies = df_puppies.sort_values(['koer','points'],ascending=False).groupby('koer').head()
df_babies = df_babies.sort_values(['koer','points'],ascending=False).groupby('koer').head()

# sum dog results
res_open = df_open.groupby(['koer', 'gender'])['points'].sum().reset_index()
res_juniors = df_juniors.groupby(['koer', 'gender'])['points'].sum().reset_index()
res_puppies = df_puppies.groupby(['koer', 'gender'])['points'].sum().reset_index()
res_babies = df_babies.groupby(['koer', 'gender'])['points'].sum().reset_index()

# sort values by points
res_open = res_open.sort_values('points', ascending=False).reset_index(drop=True)
res_juniors = res_juniors.sort_values('points', ascending=False).reset_index(drop=True)
res_puppies = res_puppies.sort_values('points', ascending=False).reset_index(drop=True)
res_babies = res_babies.sort_values('points', ascending=False).reset_index(drop=True)

# filter out only estonian dogs, i.e. select only rows where column 'koer' has prefix 'EST'
res_open_EST = res_open[res_open.koer.str.match("EST-")].reset_index(drop=True)
res_juniors_EST = res_juniors[res_juniors.koer.str.match("EST-")].reset_index(drop=True)
res_puppies_EST = res_puppies[res_puppies.koer.str.match("EST-")].reset_index(drop=True)
res_babies_EST = res_babies[res_babies.koer.str.match("EST-")].reset_index(drop=True)

# join puppies and babies
frames = [res_puppies_EST, res_babies_EST]
res_puppies_EST = pd.concat(frames)
res_puppies_EST = res_puppies_EST.groupby(['koer', 'gender'])['points'].sum().reset_index()
res_puppies_EST = res_puppies_EST.sort_values('points', ascending=False).reset_index(drop=True)

# write dog of the year results data to a files
res_open_EST.to_excel('Tulemused_ava_EST_2018.xlsx', sheet_name='Tulemused avaklass')
res_juniors_EST.to_excel('Tulemused_juunior_EST_2018.xlsx', sheet_name='Tulemused juuniorklass')
res_puppies_EST.to_excel('Tulemused_kutsikad_EST_2018.xlsx', sheet_name='Tulemused kutsika klass')

df_open.to_excel('ava_2018.xlsx', sheet_name='avaklass')
df_juniors.to_excel('juunior_2018.xlsx', sheet_name='juuniorklass')
df_puppies.to_excel('kutsikad_2018.xlsx', sheet_name='kutsika klass')
