# -*- coding: utf-8 -*-
import os
import numpy
import json
#import pickle as pkl
import cPickle as pkl
import random
import gzip
import shuffle


def unicode_to_utf8(d):
    return dict((key, value) for (key, value) in d.items())
    # return dict((key.encode("UTF-8"), value) for (key, value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(pkl.load(f))

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class DataIterator:
    def __init__(self, source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 item_info,
                 reviews_info,
                 dataset,
                 tri_data,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None):
        # shuffle the input file
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
            
        # user&item&category mapping dict 
        mid0_tri_voc = 'dataset/{}/{}/mid0_tri_voc.pkl'.format(dataset, tri_data)
        mid0_cat_voc = 'dataset/{}/{}/mid0_cat_voc.pkl'.format(dataset, tri_data)
        mid0_wi_voc = 'dataset/{}/{}/mid0_wi_voc.pkl'.format(dataset, tri_data)
        mid1_tri_voc = 'dataset/{}/{}/mid1_tri_voc.pkl'.format(dataset, tri_data)
        mid1_cat_voc = 'dataset/{}/{}/mid1_cat_voc.pkl'.format(dataset, tri_data)
        mid1_wi_voc = 'dataset/{}/{}/mid1_wi_voc.pkl'.format(dataset, tri_data)
        
        self.source_dicts = [] 
        for source_dict in [uid_voc, mid_voc, cat_voc, 
                            mid0_tri_voc, mid0_cat_voc, mid0_wi_voc,
                            mid1_tri_voc, mid1_cat_voc, mid1_wi_voc]:
            self.source_dicts.append(load_dict(source_dict))

        # Mapping Dict: {item:category}
        f_meta = open(item_info, "r")
        meta_map = {}
        for line in f_meta:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1]
        self.meta_id_map = {}
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx
        
        # Get all the interacted items 
        f_review = open(reviews_info, "r") #[user, item, rating, timestamp]
        self.mid_list_for_random = []
        for line in f_review:
            arr = line.strip().split("\t")
            tmp_idx = 0
            if arr[1] in self.source_dicts[1]: # if the item exsist, 
                tmp_idx = self.source_dicts[1][arr[1]] # get item's ID
            self.mid_list_for_random.append(tmp_idx) # list of all the interacted items

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1]) 
        self.n_cat = len(self.source_dicts[2])

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        tri0_source = []
        tri1_source = []
        target = []
        
        # Buffer: ss is one line of local_train_splitByUser/local_test_splitByUser
        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

            # sort by history behavior length
            if self.sort_by_length:
                his_length = numpy.array([len(s[4].split("")) for s in self.source_buffer])
                tidx = his_length.argsort()
                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            '''
            each ss, [id, user, item, category, [item list], [item cate list]]
            [['0',
              'AZPJ9LUT0FEPY',
              'B00AMNNTIA',
              'Literature & Fiction',
              '0307744434\x020062248391\x020470530707\x020978924622\x021590516400',
              'Books\x02Books\x02Books\x02Books\x02Books']]
            '''
            while True:
                #---------- Mapping the train/test data to unique index, mask-tag is 0 ----------#
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0
                # Triangles
                mid0_triangle = self.source_dicts[3][mid] if mid in self.source_dicts[3] else 0
                cat0_triangle = self.source_dicts[4][mid] if mid in self.source_dicts[4] else 0
                mid0_weight = self.source_dicts[5][mid] if mid in self.source_dicts[5] else 0  
                mid1_triangle = self.source_dicts[6][mid] if mid in self.source_dicts[6] else 0
                cat1_triangle = self.source_dicts[7][mid] if mid in self.source_dicts[7] else 0
                mid1_weight = self.source_dicts[8][mid] if mid in self.source_dicts[8] else 0   
    
                #------------------------------ History list ------------------------------#
                tmp = []
                tri0_tmp = []
                weight0_tmp = []
                cate0_tmp = []
                tri1_tmp = []
                weight1_tmp = []
                cate1_tmp = []
                for fea in ss[4].split(""):
                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tri0 = self.source_dicts[3][m]
                    tri0_cat = self.source_dicts[4][m]
                    wi0 = self.source_dicts[5][m]
                    tri1 = self.source_dicts[6][m]
                    tri1_cat = self.source_dicts[7][m]
                    wi1 = self.source_dicts[8][m]
                    
                    tmp.append(m)
                    tri0_tmp.append(tri0)
                    cate0_tmp.append(tri0_cat)
                    weight0_tmp.append(wi0)
                    tri1_tmp.append(tri1)
                    cate1_tmp.append(tri1_cat)
                    weight1_tmp.append(wi1)

                mid_list = tmp
                mid0_tri_list = tri0_tmp
                mid0_weight_list = weight0_tmp
                cate0_list = cate0_tmp     
                mid1_tri_list = tri1_tmp
                mid1_weight_list = weight1_tmp
                cate1_list = cate1_tmp
                      
                tmp1 = []
                for fea in ss[5].split(""):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp1.append(c)
                cat_list = tmp1
             
                # if len(mid_list) > self.maxlen:
                #    continue
                if self.minlen != None:
                    if len(mid_list) <= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue
              
                #-------------------------------- Negative sample -------------------------------#
                noclk_mid_list = []
                noclk_cat_list = []
                for pos_mid in mid_list:
                    noclk_tmp_mid = []
                    noclk_tmp_cat = []
                    noclk_index = 0
                    while True:
                        # Random sample negative item for (history records) mid_list
                        # Including item+category
                        noclk_mid_indx = random.randint(0, len(self.mid_list_for_random) - 1)
                        noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                        if noclk_mid == pos_mid:
                            continue
                        noclk_tmp_mid.append(noclk_mid)
                        noclk_m = self.meta_id_map[noclk_mid]
                        noclk_tmp_cat.append(noclk_m)
                        '''
                        tri0 = self.source_dicts[3][noclk_m]
                        noclk_tri0_cat = self.source_dicts[4][noclk_m]
                        wi0 = self.source_dicts[5][noclk_m]
                        tri1 = self.source_dicts[3][noclk_m]
                        noclk_tri1_cat = self.source_dicts[4][noclk_m]
                        wi1 = self.source_dicts[5][noclk_m]
                        '''
                        noclk_index += 1
                        if noclk_index >= 5:
                            break
                    noclk_mid_list.append(noclk_tmp_mid)
                    noclk_cat_list.append(noclk_tmp_cat)
                    
                #--------------------------------------------------------------------------------#
                source.append([uid, mid, cat, mid_list, cat_list, noclk_mid_list, noclk_cat_list])
                tri0_source.append([mid0_triangle, cat0_triangle, mid0_weight, 
                                    mid0_tri_list, cate0_list, mid0_weight_list])
                tri1_source.append([mid1_triangle, cat1_triangle, mid1_weight,
                                    mid1_tri_list, cate1_list, mid1_weight_list])
                target.append([float(ss[0]), 1 - float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in max-batch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, tri0_source, tri1_source, target = self.next()
            
        return source, tri0_source, tri1_source, target