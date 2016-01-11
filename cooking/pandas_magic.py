# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 22:11:41 2015
@author: maddness
"""

import json

from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import CountVectorizer


def underscope(line):
    return '_'.join(line.split())
    

# Read data 

#with open('train_small.json') as input:
with open('train.json') as input:    
    data = json.load(input)

df = json_normalize(data, 'ingredients', ['id', 'cuisine'])
df.rename(columns = { 0: 'ingredient'}, inplace=True)


# Remove singles
group_by_ing = df.groupby('ingredient').count()
single_ings = group_by_ing[group_by_ing.id == 1].index.values
without_singles = df[~df.ingredient.isin(single_ings)]

# Replace spaces with underscopes
without_singles.loc[:,'ingredient'] = without_singles.ingredient.apply(underscope)
df = without_singles

# Vectors from words
vectorizer = CountVectorizer(analyzer="word", max_features=5500)
vectors = vectorizer.fit_transform(df.ingredient).toarray()

print df.columns

