# -*- coding: utf-8 -*-
EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
POS_EMBEDDING_DIM = 2

# train.py
model_type='TGIN'
data_path='dataset'
dataset='electronics'  
tri_data='wnd3_alpha_01_theta_09_tri_num_10' 
n_tri=10
batch_size=128
maxlen=10
test_iter=100
save_iter=100

# model.py
n_neg = 5
use_dice = True
use_negsampling = True

# tgin.py
single_tri_agg_flag = 'reduce_mean'
multi_tri_agg_flag = 'weighted_sum'

