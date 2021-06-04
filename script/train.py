# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy
import random
import pickle

import tensorflow as tf
from pre import prepare_data
from settings import *
from model import *
from utils import *
from data_iterator import DataIterator


best_auc = 0.0
def eval(sess, test_data, model, model_path):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    for src, tri0_src, tri1_src, tgt in test_data:
        nums += 1
        origin_inp, tri0_inp, tri1_inp = prepare_data(src,
                                            tri0_src,
                                            tri1_src,
                                            tgt,
                                            n_tri=n_tri,
                                            return_neg=True)
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = origin_inp
        mids_tri0, cats_tri0, wi_tri0, mid0_his, cat0_his, wi0_his, mid0_tri_mask, mid0_his_mask = tri0_inp
        mids_tri1, cats_tri1, wi_tri1, mid1_his, cat1_his, wi1_his, mid1_tri_mask, mid1_his_mask = tri1_inp
    
        prob, loss, acc, aux_loss  = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, 
                                                     noclk_mids, noclk_cats,
                                                     mids_tri0, cats_tri0, wi_tri0, mid0_his, cat0_his, wi0_his,
                                                     mids_tri1, cats_tri1, wi_tri1, mid1_his, cat1_his, wi1_his,
                                                     mid0_tri_mask, mid1_tri_mask,
                                                     mid0_his_mask, mid1_his_mask,])

        
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum


def train(seed=1234):
    train_file = os.path.join(data_path, dataset, "local_train_splitByUser") 
    test_file = os.path.join(data_path, dataset, "local_test_splitByUser")
    uid_voc = os.path.join(data_path, dataset, "uid_voc.pkl")
    mid_voc = os.path.join(data_path, dataset, "mid_voc.pkl")
    cat_voc = os.path.join(data_path, dataset, "cat_voc.pkl")
    item_info = os.path.join(data_path, dataset, "item-info")
    reviews_info = os.path.join(data_path, dataset, "reviews-info")
    
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        t1 = time.time()
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                  dataset, tri_data, batch_size, maxlen, shuffle_each_epoch=False)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                 dataset, tri_data, batch_size, maxlen)
        print('# Load data time (s):', round(time.time()-t1, 2))
        
        n_uid, n_mid, n_cat = train_data.get_n()
        print(n_uid, n_mid, n_cat)
        t1 = time.time()
        model = TGIN(n_uid, n_mid, n_cat, n_tri, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        print('# Contruct model time (s):', round(time.time()-t1, 2))
        
        t1 = time.time()
        print('test_auc: %.4f -- test_loss: %.4f -- test_accuracy: %.4f -- test_aux_loss: %.4f' % eval(
                sess, test_data, model, best_model_path))
        print('# Eval model time (s):', round(time.time()-t1, 2))
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        lr = 0.001
        for itr in xrange(2):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            t1 = time.time()
            for src, tri0_src, tri1_src, tgt in train_data:
                origin_inp, tri0_inp, tri1_inp = prepare_data(src,
                                                    tri0_src,
                                                    tri1_src,
                                                    tgt,
                                                    n_tri=n_tri,
                                                    return_neg=True)
             
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = origin_inp
                mids_tri0, cats_tri0, wi_tri0, mid0_his, cat0_his, wi0_his, mid0_tri_mask, mid0_his_mask = tri0_inp
                mids_tri1, cats_tri1, wi_tri1, mid1_his, cat1_his, wi1_his, mid1_tri_mask, mid1_his_mask = tri1_inp
           
                loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr,
                                                          noclk_mids, noclk_cats,
                                                          mids_tri0, cats_tri0, wi_tri0, mid0_his, cat0_his, wi0_his,
                                                          mids_tri1, cats_tri1, wi_tri1, mid1_his, cat1_his, wi1_his,
                                                          mid0_tri_mask, mid1_tri_mask,
                                                          mid0_his_mask, mid1_his_mask,])
             
                
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1
        
                # Print & Save
                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    print('[Time] ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print('Best_auc:', best_auc)
                    print('iter: %d --> train_loss: %.4f -- train_accuracy: %.4f -- train_aux_loss: %.4f' % \
                          (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                    print('test_auc: %.4f -- test_loss: %.4f -- test_accuracy: %.4f -- test_aux_loss: %.4f' % eval(
                            sess, test_data, model, best_model_path))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % save_iter) == 0:
                    print('save model iter: %d' % (iter))
                    model.save(sess, model_path + "--" + str(iter))
            lr *= 0.5
        #----------------------------------------------- One Epoch------------------------------------------------#
        
        

def test(seed=1234):
    train_file = os.path.join(data_path, dataset, "local_train_splitByUser") 
    test_file = os.path.join(data_path, dataset, "local_test_splitByUser")
    uid_voc = os.path.join(data_path, dataset, "uid_voc.pkl")
    mid_voc = os.path.join(data_path, dataset, "mid_voc.pkl")
    cat_voc = os.path.join(data_path, dataset, "cat_voc.pkl")
    item_info = os.path.join(data_path, dataset, "item-info")
    reviews_info = os.path.join(data_path, dataset, "reviews-info")
    
    model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        t1 = time.time()
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                  dataset, tri_data, batch_size, maxlen, shuffle_each_epoch=False)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                 dataset, tri_data, batch_size, maxlen)
        print('# Load data time (s):', round(time.time()-t1, 2))
        n_uid, n_mid, n_cat = train_data.get_n()
        
        model = TGIN(n_uid, n_mid, n_cat, n_tri, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        model.restore(sess, model_path)
        print('test_auc: %.4f -- test_loss: %.4f -- test_accuracy: %.4f -- test_aux_loss: %.4f' % eval(
            sess, test_data, model, model_path))

        
    
if __name__ == '__main__':
    if len(sys.argv) == 4:
        SEED = int(sys.argv[3])
    else:
        SEED = 1234        
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    
    if sys.argv[1] == 'train':
        train(seed=SEED)
    elif sys.argv[1] == 'test':
        test(seed=SEED)
    else:
        print('do nothing...')
    print('Finish!!!')
        
    
