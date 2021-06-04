# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend 
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
# from tensorflow.python.ops.rnn import dynamic_rnn
from rnn import dynamic_rnn
from Dice import dice
from settings import *
from utils import *
from tgin import *


class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, n_tri, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None,], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None,], name='mid_batch_ph')
            self.cat_batch_ph = tf.placeholder(tf.int32, [None,], name='cat_batch_ph')
            self.mask = tf.placeholder(tf.int32, [None, None], name='mask')
            self.mid0_tri_mask = tf.placeholder(tf.int32, [None, None], name='mid0_tri_mask')
            self.mid1_tri_mask = tf.placeholder(tf.int32, [None, None], name='mid1_tri_mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.lr = tf.placeholder(tf.float32, [])
            self.use_negsampling = use_negsampling
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None],
                                                         name='noclk_mid_batch_ph')  # generate 3 item IDs negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], 
                                                         name='noclk_cat_batch_ph')
                
            #------------------------------ Triangle placeholder ------------------------------#
            # 0-hop
            self.mids_tri0_batch_ph = tf.placeholder(tf.int32, [None, n_tri*3], name='mids_tri0_batch_ph')
            self.cats_tri0_batch_ph = tf.placeholder(tf.int32, [None, n_tri*3], name='cats_tri0_batch_ph')
            self.wi_tri0_batch_ph = tf.placeholder(tf.float32, [None, n_tri*1], name='wi_tri0_batch_ph')
            self.mid0_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen, n_tri*3], name='mid0_his_batch_ph')
            self.cat0_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen, n_tri*3], name='cat0_his_batch_ph')
            self.wi0_his_batch_ph = tf.placeholder(tf.float32, [None, maxlen, n_tri*1], name='wi0_his_batch_ph')  
            self.mid0_his_mask = tf.placeholder(tf.int32, [None, maxlen, n_tri*3], name='mid0_his_mask')
            
            '''
            self.noclk_mid0_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen, n_neg, n_tri*3], 
                                                          name='noclk_mid0_his_batch_ph')
            self.noclk_cat0_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen, n_neg, n_tri*3], 
                                                          name='noclk_cat0_his_batch_ph')
            self.noclk_wi0_his_batch_ph = tf.placeholder(tf.float32, [None, maxlen, n_neg, n_tri*1], 
                                                         name='noclk_wi0_his_batch_ph')
            '''
            # 1-hop
            self.mids_tri1_batch_ph = tf.placeholder(tf.int32, [None, n_tri*3], name='mids_tri1_batch_ph')
            self.cats_tri1_batch_ph = tf.placeholder(tf.int32, [None, n_tri*3], name='cats_tri1_batch_ph')
            self.wi_tri1_batch_ph = tf.placeholder(tf.float32, [None, n_tri*1], name='wi_tri1_batch_ph')
            self.mid1_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen, n_tri*3], name='mid1_his_batch_ph')
            self.cat1_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen, n_tri*3], name='cat1_his_batch_ph')
            self.wi1_his_batch_ph = tf.placeholder(tf.float32, [None, maxlen, n_tri*1], name='wi1_his_batch_ph')
            self.mid1_his_mask = tf.placeholder(tf.int32, [None, maxlen, n_tri*3], name='mid1_his_mask')
            '''
            self.noclk_mid1_his_batch_ph = tf.placeholder(tf.int32, [None, None, n_neg, n_tri*3], 
                                                          name='noclk_mid1_his_batch_ph')
            self.noclk_cat1_his_batch_ph = tf.placeholder(tf.int32, [None, None, n_neg, n_tri*3], 
                                                          name='noclk_cat1_his_batch_ph')
            self.noclk_wi1_his_batch_ph = tf.placeholder(tf.int32, [None, None, n_neg, n_tri*1], 
                                                         name='noclk_wi1_his_batch_ph')
            '''
            

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph) # tri:user_profile
            
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph) 
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                           self.noclk_mid_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                                           self.noclk_cat_batch_ph)
                
            self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
            self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
            self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

            POS_EMBEDDING_DIM = 2
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, POS_EMBEDDING_DIM])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.item_his_eb)[0], 1])  # B*T,E
            self.pos_batch_embedded = tf.reshape(self.position_his_eb, [tf.shape(self.item_his_eb)[0], -1, 
                                                                        self.position_his_eb.get_shape().as_list()[1]])  

        
  
        '''
        Input fot triangle
        ub0_triangle_node, ub0_triangle_score, cand0_triangle_node, cand0_triangle_score,
        ub1_triangle_node, ub1_triangle_score, cand1_triangle_node, cand1_triangle_score,
        pos_aware_tp_fea, user_profile,
        '''
        # 0-hop
        self.mid0_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid0_his_batch_ph)
        self.cat0_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat0_his_batch_ph)
        self.ub0_triangle_node = tf.concat([self.mid0_his_batch_embedded, self.cat0_his_batch_embedded], 3)
        self.ub0_triangle_score = tf.expand_dims(self.wi0_his_batch_ph, 3) 
        
        self.mid0_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mids_tri0_batch_ph)
        self.cat0_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cats_tri0_batch_ph)
        self.cand0_triangle_node = tf.concat([self.mid0_batch_embedded, self.cat0_batch_embedded], 2) 
        self.cand0_triangle_score = tf.expand_dims(self.wi_tri0_batch_ph, 2) 
        # 1-hop
        self.mid1_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid1_his_batch_ph)
        self.cat1_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat1_his_batch_ph)
        self.ub1_triangle_node = tf.concat([self.mid1_his_batch_embedded, self.cat1_his_batch_embedded], 3)
        self.ub1_triangle_score = tf.expand_dims(self.wi1_his_batch_ph, 3) 
            
        self.mid1_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mids_tri1_batch_ph)
        self.cat1_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cats_tri1_batch_ph)
        self.cand1_triangle_node = tf.concat([self.mid1_batch_embedded, self.cat1_batch_embedded], 2) 
        self.cand1_triangle_score = tf.expand_dims(self.wi_tri1_batch_ph, 2) 
        
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]],
                -1)  # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1],
                                                 36])  # cat embedding 18 concate item embedding 18.

            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

            
    def build_fcn_net(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
          
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()
        

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    
    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    
    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer],
                                                   feed_dict={
                                                       self.uid_batch_ph: inps[0],
                                                       self.mid_batch_ph: inps[1],
                                                       self.cat_batch_ph: inps[2],
                                                       self.mid_his_batch_ph: inps[3],
                                                       self.cat_his_batch_ph: inps[4],
                                                       self.mask: inps[5],
                                                       self.target_ph: inps[6],
                                                       self.seq_len_ph: inps[7], 
                                                       self.lr: inps[8],
                                                       self.noclk_mid_batch_ph: inps[9],
                                                       self.noclk_cat_batch_ph: inps[10],                   
                                                       # Triangles 0-hop
                                                       self.mids_tri0_batch_ph: inps[11],
                                                       self.cats_tri0_batch_ph: inps[12],
                                                       self.wi_tri0_batch_ph: inps[13],
                                                       self.mid0_his_batch_ph: inps[14],
                                                       self.cat0_his_batch_ph: inps[15],
                                                       self.wi0_his_batch_ph: inps[16],
                                                       # Triangles 1-hop
                                                       self.mids_tri1_batch_ph: inps[17],
                                                       self.cats_tri1_batch_ph: inps[18],
                                                       self.wi_tri1_batch_ph: inps[19],
                                                       self.mid1_his_batch_ph: inps[20],
                                                       self.cat1_his_batch_ph: inps[21],
                                                       self.wi1_his_batch_ph: inps[22],
                                                       self.mid0_tri_mask:inps[23],
                                                       self.mid1_tri_mask:inps[24],
                                                       self.mid0_his_mask:inps[25],
                                                       self.mid1_his_mask:inps[26]
                                                   })
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _  = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
                # Triangles 0-hop
                self.mids_tri0_batch_ph: inps[11],
                self.cats_tri0_batch_ph: inps[12],
                self.wi_tri0_batch_ph: inps[13],
                self.mid0_his_batch_ph: inps[14],
                self.cat0_his_batch_ph: inps[15],
                self.wi0_his_batch_ph: inps[16],
                # Triangles 1-hop
                self.mids_tri1_batch_ph: inps[17],
                self.cats_tri1_batch_ph: inps[18],
                self.wi_tri1_batch_ph: inps[19],
                self.mid1_his_batch_ph: inps[20],
                self.cat1_his_batch_ph: inps[21],
                self.wi1_his_batch_ph: inps[22],
                self.mid0_tri_mask:inps[23],
                self.mid1_tri_mask:inps[24],
                self.mid0_his_mask:inps[25],
                self.mid1_his_mask:inps[26]
            })
            return loss, accuracy, 0

        
    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss],
                                                       feed_dict={
                                                           self.uid_batch_ph: inps[0],
                                                           self.mid_batch_ph: inps[1],
                                                           self.cat_batch_ph: inps[2],
                                                           self.mid_his_batch_ph: inps[3],
                                                           self.cat_his_batch_ph: inps[4],
                                                           self.mask: inps[5],
                                                           self.target_ph: inps[6],
                                                           self.seq_len_ph: inps[7],
                                                           self.noclk_mid_batch_ph: inps[8],
                                                           self.noclk_cat_batch_ph: inps[9],
                                                           # Triangles 0-hop
                                                           self.mids_tri0_batch_ph: inps[10],
                                                           self.cats_tri0_batch_ph: inps[11],
                                                           self.wi_tri0_batch_ph: inps[12],
                                                           self.mid0_his_batch_ph: inps[13],
                                                           self.cat0_his_batch_ph: inps[14],
                                                           self.wi0_his_batch_ph: inps[15],
                                                           # Triangles 1-hop
                                                           self.mids_tri1_batch_ph: inps[16],
                                                           self.cats_tri1_batch_ph: inps[17],
                                                           self.wi_tri1_batch_ph: inps[18],
                                                           self.mid1_his_batch_ph: inps[19],
                                                           self.cat1_his_batch_ph: inps[20],
                                                           self.wi1_his_batch_ph: inps[21],
                                                           self.mid0_tri_mask:inps[22],
                                                           self.mid1_tri_mask:inps[23],
                                                           self.mid0_his_mask:inps[24],
                                                           self.mid1_his_mask:inps[25]
                                                       })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                # Triangles 0-hop
                self.mids_tri0_batch_ph: inps[10],
                self.cats_tri0_batch_ph: inps[11],
                self.wi_tri0_batch_ph: inps[12],
                self.mid0_his_batch_ph: inps[13],
                self.cat0_his_batch_ph: inps[14],
                self.wi0_his_batch_ph: inps[15],
                # Triangles 1-hop
                self.mids_tri1_batch_ph: inps[16],
                self.cats_tri1_batch_ph: inps[17],
                self.wi_tri1_batch_ph: inps[18],
                self.mid1_his_batch_ph: inps[19],
                self.cat1_his_batch_ph: inps[20],
                self.wi1_his_batch_ph: inps[21],
                self.mid0_tri_mask:inps[22],
                self.mid1_tri_mask:inps[23],                                                          
                self.mid0_his_mask:inps[24],
                self.mid1_his_mask:inps[25]
            })
            return probs, loss, accuracy, 0
    
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


        
class TGIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, n_tri, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling):
        super(TGIN, self).__init__(n_uid, n_mid, n_cat, n_tri, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling) 
        #-------------------- mask keys_missing_value before input into the model --------------------#
        # Step-1, mask the history records 
        his_shape = self.ub1_triangle_node.get_shape().as_list() # [B, SEQ_LENGTH, TRIANGLE_NUM*3, H]
        his_paddings = tf.zeros_like(self.ub1_triangle_node) 
        
        key_masks = tf.equal(self.mask, tf.ones_like(self.mask)) # [B, SEQ_LENGTH]
        key_masks = tf.expand_dims(key_masks, 2) 
        key_masks = tf.expand_dims(key_masks, 3) 
        key_masks = tf.tile(key_masks, [1, 1, his_shape[-2], his_shape[-1]])
        self.ub0_triangle_node = tf.where(key_masks, self.ub0_triangle_node, his_paddings) 
        self.ub1_triangle_node = tf.where(key_masks, self.ub1_triangle_node, his_paddings) 

        # Step-2, mask the null triangles in history records 
        key_mid0_his_mask = tf.equal(self.mid0_his_mask, tf.ones_like(self.mid0_his_mask))  # [B, SEQ_LENGTH, TRIANGLE_NUM*3]
        key_mid0_his_mask = tf.expand_dims(key_mid0_his_mask, 3) 
        key_mid0_his_mask = tf.tile(key_mid0_his_mask, [1, 1, 1, his_shape[-1]])
        self.ub0_triangle_node = tf.where(key_mid0_his_mask, self.ub0_triangle_node, his_paddings)
        
        key_mid1_his_mask = tf.equal(self.mid1_his_mask, tf.ones_like(self.mid1_his_mask)) 
        key_mid1_his_mask = tf.expand_dims(key_mid1_his_mask, 3) 
        key_mid1_his_mask = tf.tile(key_mid1_his_mask, [1, 1, 1, his_shape[-1]])
        self.ub1_triangle_node = tf.where(key_mid1_his_mask, self.ub1_triangle_node, his_paddings)
        
        # Step-3, mask the null triangles in candidate items 
        cand_shape = self.cand0_triangle_node.get_shape().as_list()  # [B, TRIANGLE_NUM*3]
        cand_paddings = tf.zeros_like(self.cand0_triangle_node)
    
        key_mid0_tri_mask = tf.equal(self.mid0_tri_mask, tf.ones_like(self.mid0_tri_mask)) 
        key_mid0_tri_mask = tf.expand_dims(key_mid0_tri_mask, 2)  
        key_mid0_tri_mask = tf.tile(key_mid0_tri_mask, [1, 1, cand_shape[-1]])
        self.cand0_triangle_node = tf.where(key_mid0_tri_mask, self.cand0_triangle_node, cand_paddings)

        key_mid1_tri_mask = tf.equal(self.mid1_tri_mask, tf.ones_like(self.mid1_tri_mask)) 
        key_mid1_tri_mask = tf.expand_dims(key_mid1_tri_mask, 2)  
        key_mid1_tri_mask = tf.tile(key_mid1_tri_mask, [1, 1, cand_shape[-1]])
        self.cand1_triangle_node = tf.where(key_mid1_tri_mask, self.cand1_triangle_node, cand_paddings)
        #---------------------------------------------------------------------------------------------#        

        if use_negsampling:
            inp, ub0_triangle_agg, ub1_triangle_agg =\
                                        tgin(self.ub0_triangle_node, self.ub0_triangle_score, 
                                        self.cand0_triangle_node, self.cand0_triangle_score,
                                        self.ub1_triangle_node, self.ub1_triangle_score, 
                                        self.cand1_triangle_node, self.cand1_triangle_score,
                                        self.pos_batch_embedded, self.uid_batch_embedded, 'Triangle_layer')
            aux_loss_0 = self.auxiliary_loss(ub0_triangle_agg[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.noclk_item_his_eb[:, 1:, :],
                                             self.mask[:, 1:], stag="gru00")
            aux_loss_1 = self.auxiliary_loss(ub1_triangle_agg[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.noclk_item_his_eb[:, 1:, :],
                                             self.mask[:, 1:], stag="gru01")
            self.aux_loss = aux_loss_0+aux_loss_1
        else:
            inp = tgin(self.ub0_triangle_node, self.ub0_triangle_score, 
                                      self.cand0_triangle_node, self.cand0_triangle_score,
                        self.ub1_triangle_node, self.ub1_triangle_score, self.cand1_triangle_node, self.cand1_triangle_score,
                        self.pos_batch_embedded, self.uid_batch_embedded, 'Triangle_layer')

        inp = tf.concat([self.uid_batch_embedded,
                         self.item_eb, 
                         self.item_his_eb_sum, 
                         self.item_eb * self.item_his_eb_sum, 
                         inp], -1)

        # Fully connected layer
        self.build_fcn_net(inp, use_dice)