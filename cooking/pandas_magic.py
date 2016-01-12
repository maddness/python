# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 22:11:41 2015
@author: maddness
"""

import json
import pandas as pd

from pandas.io.json import json_normalize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def underscope(line):
    return '_'.join(line.split())
   

# Read data 

with open('data/train_small.json') as input:
#with open('data/train.json') as input:    
    data = json.load(input)

df = json_normalize(data, 'ingredients', ['id', 'cuisine'])
df.rename(columns = {0: 'ingredient'}, inplace=True)


# Add cuisine id
cuisine_set = set(df.groupby('cuisine').count().index.values)
cuisine_id_map = {}

for (i, value) in enumerate(cuisine_set):
    cuisine_id_map[value] = i

df['cuisine_id'] = df.cuisine.apply(lambda x: cuisine_id_map[x])


# Remove singles
group_by_ing = df.groupby('ingredient').count()
single_ings = group_by_ing[group_by_ing.id == 1].index.values
without_singles = df[~df.ingredient.isin(single_ings)]


# Replace spaces with underscopes
without_singles.loc[:,'ingredient'] = without_singles.ingredient.apply(underscope)
df = without_singles


# Add vectorized column from words
vectorizer = CountVectorizer(analyzer="word", max_features=5500)
vectors = vectorizer.fit_transform(df.ingredient).toarray()

df = df.reset_index(drop=True)
df['vec'] = pd.Series(list(vectors))



# Train model
forest = RandomForestClassifier(n_estimators=10)
forest.fit(df.vec.values, df.cuisine_id)

















