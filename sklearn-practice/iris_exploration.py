# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:43:24 2016

@author: aostrikov
"""

import pandas as pd
import seaborn as sns;

from sklearn.datasets import load_iris

data = load_iris()

data['feature_names']
data['data']
data['target']
data['target_names']

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

df = pd.DataFrame(data['data'], columns=columns)




