# -*- coding: utf-8 -*-


#%%
import json
import numpy as np
import pandas as pd

from pandas.io.json import json_normalize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


with open('data/train.json') as input:    
    data = json.load(input)

#data = data[:1000]
train_raw, test_raw = train_test_split(data, test_size = 0.2)

# make a flat dataframe
flat_data = json_normalize(train_raw, 'ingredients', ['id', 'cuisine'])
flat_data.rename(columns = {0: 'ingredient'}, inplace=True)

# replace spaces with underscopes in ingridients
def underscope_lower(ing_list):
    return '_'.join(ing_list.split()).lower()

flat_data['ingredient'] = flat_data.ingredient.map(underscope_lower)


# Remove singles
#group_by_ing = flat_data.groupby('ingredient').count()
#single_ings = group_by_ing[group_by_ing.id == 1].index.values
#data_no_singles = flat_data[~flat_data.ingredient.isin(single_ings)].reset_index(drop=True)


# group it back up
#group_by_id = data_no_singles.groupby(['id', 'cuisine'])
group_by_id = flat_data.groupby(['id', 'cuisine'])

train = group_by_id['ingredient'].apply(lambda x:  ' '.join(x.tolist())).reset_index()


# Add cuisine id
cuisines = list(enumerate(np.unique(train.cuisine)))
cuisines_dict = { val: key for (key, val) in cuisines}
train['cuisine_id'] = train.cuisine.map(cuisines_dict)


#%% Add vectorized column from words
vectorizer = CountVectorizer(analyzer="word", max_features=5500)
vectors = vectorizer.fit_transform(train.ingredient).toarray()


# Train model
forest = RandomForestClassifier(n_estimators=10)
forest.fit(vectors, train.cuisine_id)

#%% prepare test data
test = pd.DataFrame(test_raw)

def to_line_with_ingrids(ing_list):
    ings_lower_formatted = map(underscope_lower, ing_list)
    return ' '.join(ings_lower_formatted)

test['ingredients'] = test['ingredients'].apply(to_line_with_ingrids)
test['cuisine_id_expected'] = test.cuisine.map(cuisines_dict)

test_vectors = vectorizer.transform(test.ingredients).toarray()

# make predictions
result = forest.predict(test_vectors)

print accuracy_score(test.cuisine_id_expected, result)








