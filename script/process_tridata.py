# -*- coding: utf-8 -*-
import os
import time
# import pickle as pkl
import cPickle as pkl
from tqdm import tqdm

from settings import *
from data_iterator import load_dict, fopen



def process_meta_tri(tri_line, length):
    if '' in tri_line:
        return None, None, None

    nodes = [tri_line[i].split("\x1d")[0] for i in range(3)]
    nodes = [int(n.split('_')[-1]) for n in nodes]
    cates = [meta_id_map[n] for n in nodes]
    weight = float(tri_line[3])
    return nodes, cates, weight


#--------------------------- Load data --------------------------#
def load_meta_id_map(data_path, dataset):
    mid_voc = os.path.join(data_path, dataset, "mid_voc.pkl")
    cat_voc = os.path.join(data_path, dataset, "cat_voc.pkl")
    item_info = os.path.join(data_path, dataset, "item-info")
    
    source_dicts = []
    for source_dict in [mid_voc, cat_voc]:
        source_dicts.append(load_dict(source_dict))

    # Mapping Dict: {item_id:category_id}
    f_meta = open(item_info, "r")
    meta_id_map = {}
    meta_map = {}
    for line in f_meta:
        arr = line.strip().split("\t")
        if arr[0] not in meta_map:
            meta_map[arr[0]] = arr[1]

    for key in meta_map:
        val = meta_map[key]
        if key in source_dicts[0]:
            mid_idx = source_dicts[0][key]
        else:
            mid_idx = 0
        if val in source_dicts[1]:
            cat_idx = source_dicts[1][val]
        else:
            cat_idx = 0
        meta_id_map[mid_idx] = cat_idx
    
    # Padding
    return meta_id_map

def process_triangles(tri_source, meta_id_map):
    mid0_tri_dict, mid0_cat_dict, mid0_wi_dict = {}, {}, {}
    mid1_tri_dict, mid1_cat_dict, mid1_wi_dict = {}, {}, {}
    mid0_length = 4
    mid1_length = 5
    mid2_length = 5
        
    tri_source = fopen(tri_source, 'r')
    tri_source.readline()
    for i in tqdm(range(NUM_ITEMS+1)):
        line = tri_source.readline()
        if len(line.strip())==0: 
            break
            
        #---------------------------- preprocess each center node ----------------------------
        node_data = line.strip("\n").split("\t")    
        center_node = int(node_data[0])
        if center_node not in mid0_tri_dict: mid0_tri_dict[center_node] = []
        if center_node not in mid0_cat_dict: mid0_cat_dict[center_node] = []
        if center_node not in mid0_wi_dict: mid0_wi_dict[center_node] = []    
        if center_node not in mid1_tri_dict: mid1_tri_dict[center_node] = []
        if center_node not in mid1_cat_dict: mid1_cat_dict[center_node] = []
        if center_node not in mid1_wi_dict: mid1_wi_dict[center_node] = []

        mid0_tri_data_len = FULL_TRIANGLE_NUM*mid0_length
        mid1_tri_data_len = FULL_TRIANGLE_NUM*mid1_length
        mid2_tri_data_len = FULL_TRIANGLE_NUM*mid2_length

        mid0_tri_data = node_data[1:mid0_tri_data_len+1]
        mid1_tri_data = node_data[mid0_tri_data_len+1:mid0_tri_data_len+mid1_tri_data_len+1]
        mid2_tri_data = node_data[mid0_tri_data_len+mid1_tri_data_len+1:]

        #---------------------------- 0-hop ----------------------------#
        for i in range(0, len(mid0_tri_data), mid0_length):
            tri_line = mid0_tri_data[i: i+mid0_length]   
            triangle, cates, weight = process_meta_tri(tri_line, mid0_length)
            if triangle == None:
                break
            mid0_tri_dict[center_node] += triangle
            mid0_cat_dict[center_node] += cates
            mid0_wi_dict[center_node].append(weight)
        #----------------------------1-hop ----------------------------#
        for i in range(0, len(mid1_tri_data), mid1_length):
            tri_line = mid1_tri_data[i: i+mid1_length]
            triangle, cates, weight = process_meta_tri(tri_line, mid1_length)
            if triangle == None:
                break
            mid1_tri_dict[center_node] += triangle
            mid1_cat_dict[center_node] += cates
            mid1_wi_dict[center_node].append(weight)
            
    #---------------------------- Padding Isolate Items ---------------------------#
    padding_node_list = []
    all_nodes = list(meta_id_map.keys())
    exsistng_nodes = list(mid0_tri_dict.keys())
    padding_node_list = list(set(all_nodes)-set(exsistng_nodes))
  
    for center_node in padding_node_list:     
        if center_node not in mid0_tri_dict: mid0_tri_dict[center_node] = []
        if center_node not in mid0_cat_dict: mid0_cat_dict[center_node] = []
        if center_node not in mid0_wi_dict: mid0_wi_dict[center_node] = []    
        if center_node not in mid1_tri_dict: mid1_tri_dict[center_node] = []
        if center_node not in mid1_cat_dict: mid1_cat_dict[center_node] = []
        if center_node not in mid1_wi_dict: mid1_wi_dict[center_node] = []

    # Output
    mid0_tri_dict = {k:v[:TRIANGLE_NUM*3] for k,v in mid0_tri_dict.items()}
    mid0_cat_dict = {k:v[:TRIANGLE_NUM*3] for k,v in mid0_cat_dict.items()}
    mid0_wi_dict = {k:v[:TRIANGLE_NUM] for k,v in mid0_wi_dict.items()}

    mid1_tri_dict = {k:v[:TRIANGLE_NUM*3] for k,v in mid1_tri_dict.items()}
    mid1_cat_dict = {k:v[:TRIANGLE_NUM*3] for k,v in mid1_cat_dict.items()}
    mid1_wi_dict = {k:v[:TRIANGLE_NUM] for k,v in mid1_wi_dict.items()}

    return mid0_tri_dict, mid0_cat_dict, mid0_wi_dict, mid1_tri_dict, mid1_cat_dict, mid1_wi_dict
    

def save():
    print('-- Saving...')
    save_dir = os.path.join(data_path, dataset, '{}_tri_num_{}'.format(tri_file[:tri_file.index('.')], TRIANGLE_NUM))
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    mid0_tri_path = os.path.join(save_dir, "mid0_tri_voc.pkl")
    mid0_cat_path = os.path.join(save_dir, "mid0_cat_voc.pkl")
    mid0_wi_path = os.path.join(save_dir, "mid0_wi_voc.pkl")
    mid1_tri_path = os.path.join(save_dir, "mid1_tri_voc.pkl")
    mid1_cat_path = os.path.join(save_dir, "mid1_cat_voc.pkl")
    mid1_wi_path = os.path.join(save_dir, "mid1_wi_voc.pkl")

    pkl.dump(mid0_tri_dict, open(mid0_tri_path, 'wb'))
    pkl.dump(mid0_cat_dict, open(mid0_cat_path, 'wb'))
    pkl.dump(mid0_wi_dict, open(mid0_wi_path, 'wb'))
    print('-- mid0 Finished!')

    pkl.dump(mid1_tri_dict, open(mid1_tri_path, 'wb'))
    pkl.dump(mid1_cat_dict, open(mid1_cat_path, 'wb'))
    pkl.dump(mid1_wi_dict, open(mid1_wi_path, 'wb'))
    print('-- mid1 Finished!')


#--------------------------- Settings --------------------------#
FULL_TRIANGLE_NUM = 10
TRIANGLE_NUM = 10
NUM_ITEMS = 62899
dataset = 'electronics'
tri_file = 'wnd3_alpha_01_theta_09.csv'
tri_source = 'triangle_data/{}_triangle/{}'.format(dataset, tri_file)

meta_id_map = load_meta_id_map('dataset', dataset)
mid0_tri_dict, mid0_cat_dict, mid0_wi_dict,\
mid1_tri_dict, mid1_cat_dict, mid1_wi_dict = process_triangles(tri_source, meta_id_map)
save()