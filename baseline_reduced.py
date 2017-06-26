# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:09:19 2017

@author: alex
"""

# --- Import Libraries --- #
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score

# --- Read Data --- #
# Adapted from Paul-Antoine Nguyen (GBM.py)
os.chdir('C:/Users/alex/Documents/Instacart_Data/')
IDIR = 'C:/Users/alex/Documents/Instacart_Data/data/'

print('loading prior')
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading train')
train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print('loading products')
products = pd.read_csv(IDIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])

print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))

# Reshape train 
train.head(4)

# Merge product information and train/priors by product_id 
# Merge order infromation and train/priors by order_id
#A = train.merge(products, on = 'product_id').sort_values(['order_id', 'add_to_cart_order']).reset_index()
#B = priors.merge(products, on = 'product_id').sort_values(['order_id', 'add_to_cart_order']).reset_index()

train_merge = train.merge(products, on = 'product_id').merge(orders, on = 'order_id').sort_values(['order_id', 'add_to_cart_order']).reset_index()
priors_merge = priors.merge(products, on = 'product_id').merge(orders, on = 'order_id').sort_values(['order_id', 'add_to_cart_order']).reset_index()
print('Merged train {}: {}'.format(train_merge.shape, ', '.join(train_merge.columns)))
train_merge.head(5)

print('Merged priors {}: {}'.format(priors_merge.shape, ', '.join(priors_merge.columns)))
priors_merge.head(5)

# Find n unique users
priors_merge.user_id.unique().shape
train_merge.user_id.unique().shape

# Any users in common?
len(set(priors_merge.user_id.unique()).intersection(train_merge.user_id.unique()))


# Trying to undersatnd the structure of the problem
orders[orders['eval_set'] == 'test']




