# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np

from pandas.io.json import json_normalize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

def underscope(line):
    return '_'.join(line.split())
  
  

#with open('data/train_small.json') as input:
with open('data/train.json') as input:    
    data = json.load(input)

data = json_normalize(data, 'ingredients', ['id', 'cuisine'])
data.rename(columns = {0: 'ingredient'}, inplace=True)


# split train set
df, test = train_test_split(data, test_size = 0.5)


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
df = without_singles.reset_index(drop=True)


# Add vectorized column from words
vectorizer = CountVectorizer(analyzer="word", max_features=5500)
vectors = vectorizer.fit_transform(df.ingredient)


# Train model
forest = RandomForestClassifier(n_estimators=10)
#forest.fit(vectors, df.cuisine_id)

















