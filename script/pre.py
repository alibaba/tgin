import os
import sys
import time
import numpy
import random
#import pickle
import cPickle as pickle
from settings import *

def prepare_data(input, tri0_input, tri1_input, target, n_tri, return_neg=False):
    def fun(tri0_input):
        mid0_triangle = [tri0_inp[0] for tri0_inp in tri0_input] #[B, TRIANGLE_NUM, 3]
        cat0_triangle = [tri0_inp[1] for tri0_inp in tri0_input]
        mid0_weight = [tri0_inp[2] for tri0_inp in tri0_input]
        mid0_tri_list = [tri0_inp[3] for tri0_inp in tri0_input] 
        cat0_tri_list = [tri0_inp[4] for tri0_inp in tri0_input] 
        wi0_tri_list = [tri0_inp[5] for tri0_inp in tri0_input]
     
        return mid0_triangle, cat0_triangle, mid0_weight,\
            mid0_tri_list, cat0_tri_list, wi0_tri_list

    # x: a list of sentences
    t1 = time.time()
    lengths_x = [len(s[4]) for s in input]  # Length of history list
    seqs_mid = [inp[3] for inp in input]    # Items of history list
    seqs_cat = [inp[4] for inp in input]    # Item cates of history list 
    noclk_seqs_mid = [inp[5] for inp in input]  # Negetive item list of each items [B, max_len, n_neg]
    noclk_seqs_cat = [inp[6] for inp in input]  # Negetive cate list of each items [B, max_len, n_neg]
    
    mid0_tri_cand, cat0_tri_cand, wi0_tri_cand,\
    mid0_tri_list, cat0_tri_list, wi0_tri_list = fun(tri0_input)
    mid1_tri_cand, cat1_tri_cand, wi1_tri_cand,\
    mid1_tri_list, cat1_tri_list, wi1_tri_list= fun(tri1_input)
    
    mid0_cand_lengths = [len(i) for i in mid0_tri_cand]
    mid1_cand_lengths = [len(i) for i in mid1_tri_cand]
    
    #------------------------------ Negative list ------------------------------#  
    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        # 0-hop
        new_mid0_tri_list = []
        new_cat0_tri_list = []
        new_wi0_tri_list= []
        # 1-hop
        new_mid1_tri_list = []
        new_cat1_tri_list = []
        new_wi1_tri_list= []
        
        for l_x, inp, tri0_inp, tri1_inp in zip(lengths_x, input, tri0_input, tri1_input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
                # 0-hop
                new_mid0_tri_list.append(tri0_inp[3][l_x - maxlen:])
                new_cat0_tri_list.append(tri0_inp[4][l_x - maxlen:])
                new_wi0_tri_list.append(tri0_inp[5][l_x - maxlen:])
                # 1-hop
                new_mid1_tri_list.append(tri1_inp[3][l_x - maxlen:])
                new_cat1_tri_list.append(tri1_inp[4][l_x - maxlen:])
                new_wi1_tri_list.append(tri1_inp[5][l_x - maxlen:])
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
                # 0-hop                  
                new_mid0_tri_list.append(tri0_inp[3])
                new_cat0_tri_list.append(tri0_inp[4])
                new_wi0_tri_list.append(tri0_inp[5])
                # 1-hop 
                new_mid1_tri_list.append(tri1_inp[3])
                new_cat1_tri_list.append(tri1_inp[4])
                new_wi1_tri_list.append(tri1_inp[5])                          

        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat
        # 0-hop
        mid0_tri_list = new_mid0_tri_list
        cat0_tri_list = new_cat0_tri_list
        wi0_tri_list = new_wi0_tri_list
        # 1-hop
        mid1_tri_list = new_mid1_tri_list
        cat1_tri_list = new_cat1_tri_list
        wi1_tri_list = new_wi1_tri_list
        
        if len(lengths_x) < 1:
            return None, None, None, None
    
    
    #------------------------------ History list ------------------------------# 
    maxlen_x = maxlen
    n_samples = len(seqs_mid)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('int64')  # mask-tag is 0
    mid0_tri_mask = numpy.zeros((n_samples, n_tri*3)).astype('int64') 
    mid1_tri_mask = numpy.zeros((n_samples, n_tri*3)).astype('int64')
    
    # 0-hop
    mids_tri0 = numpy.zeros((n_samples, n_tri*3)).astype('int64')
    cats_tri0 = numpy.zeros((n_samples, n_tri*3)).astype('int64')
    wi_tri0 = numpy.zeros((n_samples, n_tri)).astype('float64')
    
    mid0_his = numpy.zeros((n_samples, maxlen_x, n_tri*3)).astype('int64')
    cat0_his = numpy.zeros((n_samples, maxlen_x, n_tri*3)).astype('int64')
    wi0_his = numpy.zeros((n_samples, maxlen_x, n_tri*1)).astype('float64')
    mid0_his_tri_mask = numpy.zeros((n_samples, maxlen_x, n_tri*3)).astype('int64')
    # 1-hop
    mids_tri1 = numpy.zeros((n_samples, n_tri*3)).astype('int64')
    cats_tri1 = numpy.zeros((n_samples, n_tri*3)).astype('int64')
    wi_tri1 = numpy.zeros((n_samples, n_tri)).astype('float64')
    
    mid1_his = numpy.zeros((n_samples, maxlen_x, n_tri*3)).astype('int64')
    cat1_his = numpy.zeros((n_samples, maxlen_x, n_tri*3)).astype('int64')
    wi1_his = numpy.zeros((n_samples, maxlen_x, n_tri*1)).astype('float64')
    mid1_his_tri_mask = numpy.zeros((n_samples, maxlen_x, n_tri*3)).astype('int64')
        
    for idx, [s_mid0_cand, s_cat0_cand, s_wi0_cand,
              s_mid1_cand, s_cat1_cand, s_wi1_cand,
              s_x, s_y, no_sx, no_sy, 
              s_mid0, s_cat0, s_wi0,
              s_mid1, s_cat1, s_wi1] in enumerate(zip(mid0_tri_cand, cat0_tri_cand, wi0_tri_cand,
                                                      mid1_tri_cand, cat1_tri_cand, wi1_tri_cand,
                                                      seqs_mid, seqs_cat, 
                                                      noclk_seqs_mid, noclk_seqs_cat,
                                                      mid0_tri_list, cat0_tri_list, wi0_tri_list,
                                                      mid1_tri_list, cat1_tri_list, wi1_tri_list)):
        # condidates
        mid0_tri_mask[idx, :mid0_cand_lengths[idx]] = 1
        mids_tri0[idx, :mid0_cand_lengths[idx]] = s_mid0_cand
        cats_tri0[idx, :mid0_cand_lengths[idx]] = s_cat0_cand
        wi_tri0[idx, :mid0_cand_lengths[idx]//3] = s_wi0_cand
        
        mid1_tri_mask[idx, :mid1_cand_lengths[idx]] = 1
        mids_tri1[idx, :mid1_cand_lengths[idx]] = s_mid1_cand
        cats_tri1[idx, :mid1_cand_lengths[idx]] = s_cat1_cand
        wi_tri1[idx, :mid1_cand_lengths[idx]//3] = s_wi1_cand
        
        # history records
        mid_mask[idx, :lengths_x[idx]] = 1
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy
    
        # input each triangle 
        s_mid0_lengths = [len(tri_list) for tri_list in s_mid0]
        s_mid1_lengths = [len(tri_list) for tri_list in s_mid1]
        for t_idx, [s_mid0_tri, s_cat0_tri, s_wi0_tri,
                    s_mid1_tri, s_cat1_tri, s_wi1_tri] in enumerate(zip(s_mid0, s_cat0, s_wi0,
                                                                        s_mid1, s_cat1, s_wi1)):
            mid0_his_tri_mask[idx, :lengths_x[idx], :s_mid0_lengths[t_idx]] = 1
            mid0_his[idx, :lengths_x[idx], :s_mid0_lengths[t_idx]] = s_mid0_tri
            cat0_his[idx, :lengths_x[idx], :s_mid0_lengths[t_idx]] = s_cat0_tri
            wi0_his[idx, :lengths_x[idx], :s_mid0_lengths[t_idx]//3] = s_wi0_tri
            
            mid1_his_tri_mask[idx, :lengths_x[idx], :s_mid1_lengths[t_idx]] = 1
            mid1_his[idx, :lengths_x[idx], :s_mid1_lengths[t_idx]] = s_mid1_tri
            cat1_his[idx, :lengths_x[idx], :s_mid1_lengths[t_idx]] = s_cat1_tri
            wi1_his[idx, :lengths_x[idx], :s_mid1_lengths[t_idx]//3] = s_wi1_tri

    
    #------------------------------ Record batch ------------------------------#  
    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])
     
    if return_neg:
        origin_inp = [uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), 
                      noclk_mid_his, noclk_cat_his]
        tri0_inp = [mids_tri0, cats_tri0, wi_tri0, mid0_his, cat0_his, wi0_his, mid0_tri_mask, mid0_his_tri_mask] 
        tri1_inp = [mids_tri1, cats_tri1, wi_tri1, mid1_his, cat1_his, wi1_his, mid1_tri_mask, mid1_his_tri_mask] 
        return origin_inp, tri0_inp, tri1_inp
    
    else:
        origin_inp = [uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)]
        tri0_inp = [mids_tri0, cats_tri0, wi_tri0, mid0_his, cat0_his, wi0_his, mid0_tri_mask, mid0_his_tri_mask]
        tri1_inp = [mids_tri1, cats_tri1, wi_tri1, mid1_his, cat1_his, wi1_his, mid1_tri_mask, mid1_his_tri_mask] 
        return origin_inp, tri0_inp, tri1_inp