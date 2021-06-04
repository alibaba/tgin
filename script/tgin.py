# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from settings import *
from tensorflow.python.framework import ops

def tgin(ub0_triangle_node, ub0_triangle_score, cand0_triangle_node, cand0_triangle_score,
         ub1_triangle_node, ub1_triangle_score, cand1_triangle_node, cand1_triangle_score,
         pos_aware_tp_fea, user_profile, var_scp, reuse=tf.AUTO_REUSE):
    """
    Args:
        ub0_triangle_node: [B, SEQ_LENGTH, TRIANGLE_NUM*3, H]
        ub_triangle_score: [B, SEQ_LENGTH, TRIANGLE_NUM, 1]
        cand_triangle_node: [B, TRIANGLE_NUM*3, H]
        cand_triangle_score: [B, TRIANGLE_NUM, 1] 
        pos_aware_tp_fea: [B, T, H] 
    Output:
        output: [B, H] 
    """
    with tf.variable_scope(name_or_scope=var_scp, reuse=reuse):
        weight_collects = [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES]
        hop0_out, ub0_triangle_agg = triangle_net(ub0_triangle_node, ub0_triangle_score, 
                                cand0_triangle_node, cand0_triangle_score, pos_aware_tp_fea, var_scp="tn0")
        hop1_out, ub1_triangle_agg = triangle_net(ub1_triangle_node, ub1_triangle_score, 
                                cand1_triangle_node, cand1_triangle_score, pos_aware_tp_fea, var_scp="tn1")
        multi_seqs = tf.stack([hop0_out, hop1_out], axis=1)
        output = fusion_unit(multi_seqs, user_profile, var_scp="fu")
    if use_negsampling:
        return output, ub0_triangle_agg, ub1_triangle_agg
    else:
        return output


def triangle_net(ub_triangle_node, ub_triangle_score, cand_triangle_node, cand_triangle_score, 
                 pos_aware_tp_fea, var_scp, reuse=tf.AUTO_REUSE):
    """
    Args:
        ub_triangle_node: [B, SEQ_LENGTH, TRIANGLE_NUM*3, H]
        ub_triangle_score: [B, SEQ_LENGTH, TRIANGLE_NUM, 1]
        cand_triangle_node: [B, TRIANGLE_NUM*3, H]
        cand_triangle_score: [B, TRIANGLE_NUM, 1] 
        pos_aware_tp_fea: [B, T, H] 
    Output:
        output: [B, H] 
    """
    with tf.variable_scope(name_or_scope=var_scp, reuse=reuse):
        #------------------------------- History triangle aggregation -------------------------------#
        ub_tri_shape = ub_triangle_node.get_shape().as_list()
        ub_triangle_node_reshape = tf.reshape(ub_triangle_node, [-1, ub_tri_shape[1], ub_tri_shape[2] / 3, 3, ub_tri_shape[3]])
        ub_triangle_score_reshape = tf.reshape(ub_triangle_score, [-1, ub_tri_shape[1], ub_tri_shape[2] / 3, 1])
        ub_triangle_agg = triangle_aggregation(ub_triangle_node_reshape, ub_triangle_score_reshape, 
                                               "agg", single_tri_agg_flag, multi_tri_agg_flag, hidden_units=HIDDEN_SIZE)  
        
        #----------------------------- Candidate triangle aggregation -----------------------------#
        cand_tri_shape = cand_triangle_node.get_shape().as_list()
        cand_triangle_node_reshape = tf.reshape(cand_triangle_node, [-1, 1, cand_tri_shape[1] / 3, 3, cand_tri_shape[2]])
        cand_triangle_score_reshape = tf.reshape(cand_triangle_score, [-1, 1, cand_tri_shape[1] / 3, 1])
        cand_triangle_agg = triangle_aggregation(cand_triangle_node_reshape, cand_triangle_score_reshape, 
                                                 "agg", single_tri_agg_flag, multi_tri_agg_flag, hidden_units=HIDDEN_SIZE)
        cand_triangle_agg = tf.reshape(cand_triangle_agg, [-1, cand_triangle_agg.get_shape().as_list()[-1]]) 

        # print("var_scp: " + var_scp + ", ub_triangle_agg shape: " + str(ub_triangle_agg.get_shape().as_list()))
        # print("var_scp: " + var_scp + ", cand_triangle_agg shape: " + str(cand_triangle_agg.get_shape().as_list()))

        #-------------------------------- aux_loss self-attention --------------------------------# 
        if use_negsampling:
            ub_triangle_agg = aux_loss_mhsa(ub_triangle_agg, num_units=EMBEDDING_DIM*2, 
                                            num_heads=4, dropout_rate=0., is_training=True)
    
        #------------------------------- position aware attention -------------------------------#  
        pos_attn_output = pos_aware_attention(cand_triangle_agg, ub_triangle_agg, pos_aware_tp_fea, var_scp + "_pos_attn")
        output = tf.concat([pos_attn_output, cand_triangle_agg], -1)
    return output, ub_triangle_agg


def triangle_aggregation(triangle_node, triangle_weight, var_scp, single_tri_agg="reduce_mean", multi_tri_agg="weighted_sum", 
                         hidden_units=HIDDEN_SIZE, reuse=tf.AUTO_REUSE):
    """
    Args:
        triangle_node: [B, SEQ_LENGTH, TRIANGLE_NUM, 3, H]
        triangle_weight: [B, SEQ_LENGTH, TRIANGLE_NUM, 1] 
        # node-pooling >>> triangle-pooling
        single_tri_agg: reduce_mean| mhsa
        multi_tri_agg: reduce_mean | weighted_sum | mhsa
    Output:
        output:  [B, SEQ_LENGTH, H]
    """
    with tf.variable_scope(name_or_scope=var_scp, reuse=reuse):        
        weight_collects = [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES]
        #  Intra-triangle Aggregation Layer.
        if single_tri_agg == "reduce_mean":
            triangle_vect = tf.reduce_mean(triangle_node, axis=3) 
        else:
            print('other aggregation strategies...')
            # triangle_node_shape = triangle_node.get_shape().as_list()
            # triangle_node_stack = tf.reshape(triangle_node, [-1, triangle_node_shape[3], triangle_node_shape[4]])
            # triangle_vect = average_mhsa(triangle_node_stack, var_scp="single_mhsa") 
            triangle_vect = tf.reduce_mean(triangle_vect, axis=3) 
           
        w = tf.get_variable('w', [triangle_vect.get_shape().as_list()[-1], hidden_units],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32), 
                            collections=weight_collects)
        b = tf.get_variable('b', [hidden_units], initializer=tf.constant_initializer(0.0), collections=weight_collects)
        triangle_output = tf.einsum("ijkl,lm->ijkm", triangle_vect, w) + b
    
        # Inter-triangle Aggregation Layer.
        if multi_tri_agg == "mhsa":
            triangle_ouput_shape = triangle_output.get_shape().as_list()
            triangle_stack = tf.reshape(triangle_output, [-1, triangle_ouput_shape[2], triangle_ouput_shape[3]])
            output = mhsa_inter_triangle_aggregation(triangle_stack, 
                                                          num_units=EMBEDDING_DIM*2, 
                                                          num_heads=4, 
                                                          dropout_rate=0., 
                                                          is_training=True)
            output = tf.reshape(output, [-1,
                                         triangle_ouput_shape[1],
                                         triangle_ouput_shape[2],
                                         output.get_shape().as_list()[-1]])
            output = tf.reduce_mean(output, axis=2) 
        elif multi_tri_agg == "weighted_sum":
            output = tf.reduce_sum(triangle_output * triangle_weight, axis=2) / (tf.reduce_sum(triangle_weight, axis=2) + 1e-9)
        else:
            output = tf.reduce_mean(triangle_output, axis=2) 
    return output


def layer_norm(inputs, name, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable(name+'gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable(name+'beta', params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs


def mhsa_inter_triangle_aggregation(inputs, num_units, num_heads, dropout_rate, name="", 
                         is_training=True, is_layer_norm=True, keys_missing_value=0):
    """
    Args:
        inputs: [B*T, TRIANGLE_NUM, H]
    Output:
        outputs: [B*T, TRIANGLE_NUM, H]
    """
    x_shape = inputs.get_shape().as_list() 
    keys_empty = tf.reduce_sum(inputs, axis=2)  # [B*T, TRIANGLE_NUM] 
    keys_empty_cond = tf.equal(keys_empty, keys_missing_value) 
    keys_empty_cond = tf.expand_dims(keys_empty_cond, 2)  
    keys_empty_cond = tf.tile(keys_empty_cond, [num_heads, 1, x_shape[1]])   # [B*T, TRIANGLE_NUM, TRIANGLE_NUM]
    
    Q_K_V = tf.layers.dense(inputs, 3 * num_units) 
    Q, K, V = tf.split(Q_K_V, 3, -1)
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # [B*T*num_heads, TRIANGLE_NUM, num_units]
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # [B*T*num_heads, TRIANGLE_NUM, num_units]
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # [B*T*num_heads, TRIANGLE_NUM, num_units]  
    
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [B*T*num_heads, TRIANGLE_NUM, TRIANGLE_NUM] 
    align= outputs / (36 ** 0.5)
    # Diag mask 
    diag_val = tf.ones_like(align[0, :, :])  # [TRIANGLE_NUM, TRIANGLE_NUM]
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [TRIANGLE_NUM, TRIANGLE_NUM]
    key_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1]) # [B*T*num_heads, TRIANGLE_NUM, TRIANGLE_NUM]
    paddings = tf.ones_like(key_masks) * (-2 ** 32 + 1)
    
    outputs = tf.where(tf.equal(key_masks, 0), paddings, align)
    outputs = tf.where(keys_empty_cond, paddings, outputs)
    outputs = tf.nn.softmax(outputs)   # [B*T, TRIANGLE_NUM, TRIANGLE_NUM]
    # Attention Matmul
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    outputs = tf.matmul(outputs, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [B*T, TRIANGLE_NUM, num_units]
    
    outputs = tf.layers.dense(outputs, num_units) # [B*T, TRIANGLE_NUM, num_units]
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    # Residual connection
    outputs += inputs 
    # Normalize
    if is_layer_norm:
        outputs = layer_norm(outputs,name=name) 
        
    outputs1 = tf.layers.dense(outputs, EMBEDDING_DIM*4, activation=tf.nn.relu)
    outputs1 = tf.layers.dense(outputs1, EMBEDDING_DIM*2)
    outputs = outputs1+outputs
    return outputs
    

def pos_aware_attention(queries, keys, keys_tp, var_scp, hidden_units=HIDDEN_SIZE, keys_missing_value=0, reuse=tf.AUTO_REUSE):
    """
    # pos aware attention H = V^T * relu(w_h * h_i + w_tp * h_tp_i + b)
    Args:
        queries: cand_triangle_agg  [B, H] 
        keys:    ub_triangle_agg    [B, T, H]
        keys_tp: pos_aware_tp_fea   [B, T, H_tp]
    Output:
        output: [B, H]
    """
    with tf.variable_scope(name_or_scope=var_scp, reuse=reuse):
        weight_collects = [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES]
        queries_hidden_units = queries.get_shape().as_list()[-1]
        queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # [B, T*H]
        queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units]) # [B, T, H]

        #-----------------------------------------------------------------------------------------------#
        w_q = tf.get_variable('w_q', [queries.get_shape().as_list()[-1], hidden_units],
                              initializer=tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32),
                              collections=weight_collects)
        w_h = tf.get_variable('w_h', [keys.get_shape().as_list()[-1], hidden_units],
                              initializer=tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32), 
                              collections=weight_collects)
        w_tp = tf.get_variable('w_tp', [keys_tp.get_shape().as_list()[-1], hidden_units],
                               initializer=tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32),
                               collections=weight_collects)
        b = tf.get_variable('b', [hidden_units], initializer=tf.constant_initializer(0.0), collections=weight_collects)
        # print("var_scp: " + var_scp + ", keys shape: " + str(keys.get_shape().as_list()))
        # print("var_scp: " + var_scp + ", keys_tp shape: " + str(keys_tp.get_shape().as_list()))
        
        #-----------------------------------------------------------------------------------------------#
        din_hidden_fea = tf.nn.leaky_relu(tf.einsum("ijk,kl->ijl", keys, w_h) + tf.einsum("ijk,kl->ijl", 
                                                                                          keys_tp, w_tp) + b) # [B, T, h]
        # print("var_scp: " + var_scp + ", din_hidden_fea shape: " + str(din_hidden_fea.get_shape().as_list()))
        # print("var_scp: " + var_scp + ", queries shape: " + str(queries.get_shape().as_list()))
        
        outputs = tf.reduce_mean(tf.multiply(tf.einsum("ijk,kl->ijl", queries, w_q), din_hidden_fea), axis=2)
        outputs = tf.expand_dims(outputs, axis=1) 
        
        keys_empty = tf.reduce_sum(keys, axis=2)  # [B, T]
        keys_empty_cond = tf.equal(keys_empty, keys_missing_value) # [B, T]
        keys_empty_cond = tf.reshape(keys_empty_cond, [-1, 1, keys_empty_cond.get_shape().as_list()[-1]]) # [bs, 1, max_len]

        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(keys_empty_cond, paddings, outputs)
        weighted = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        weighted_sum = tf.matmul(weighted, keys) # [B, 1, H]
        weighted_sum = tf.reshape(weighted_sum, [-1, keys.get_shape().as_list()[-1]])

    return weighted_sum


def fusion_unit(multi_seqs, user_profile, var_scp, hidden_units=HIDDEN_SIZE, keys_missing_value=0, reuse=tf.AUTO_REUSE):
    """
    Args:
        multi_seqs: [B, X, H]
        user_profile: [B, H_up]
    Output:
        output: [B, H]
    """
    with tf.variable_scope(name_or_scope=var_scp, reuse=reuse):
        #--------------------------------------------------------------------------------------------------------------------#
        weight_collects = [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES]

        w_up = tf.get_variable('w_up', [user_profile.get_shape().as_list()[-1], hidden_units],
                               initializer=tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32), 
                               collections=weight_collects)
        b_up = tf.get_variable('b_up', [hidden_units], initializer=tf.constant_initializer(0.0), collections=weight_collects)
        user_profile_output = tf.matmul(user_profile, w_up) + b_up  # [B, hidden_units]

        w_ms = tf.get_variable('w_ms', [multi_seqs.get_shape().as_list()[-1], hidden_units],
                               initializer=tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32), 
                               collections=weight_collects)
        b_ms = tf.get_variable('b_ms', [hidden_units], initializer=tf.constant_initializer(0.0), collections=weight_collects)
        multi_seqs_output = tf.einsum("ijk,kl->ijl", multi_seqs, w_ms) + b_ms  # [B, X, hidden_units]
        #--------------------------------------------------------------------------------------------------------------------#
        
        user_profile_output = tf.tile(user_profile_output, [1, tf.shape(multi_seqs_output)[1]])  # [B, X*H]
        user_profile_output = tf.reshape(user_profile_output, [-1, tf.shape(multi_seqs_output)[1], 
                                                               multi_seqs_output.get_shape().as_list()[-1]])  # [B, X, hidden_units]

        keys_empty = tf.reduce_sum(multi_seqs, axis=2)  # [B, X]
        keys_empty_cond = tf.equal(keys_empty, keys_missing_value)  # [B, X]
        keys_empty_cond = tf.reshape(keys_empty_cond, [-1, 1, keys_empty_cond.get_shape().as_list()[-1]])  # [B, 1, X]
        outputs = tf.reduce_mean(tf.multiply(multi_seqs_output, user_profile_output), axis=2)
        outputs = tf.reshape(outputs, [-1, 1, outputs.get_shape().as_list()[-1]])  # [B, 1, X]

        # Scale
        outputs = outputs / (multi_seqs.get_shape().as_list()[-1] ** 0.5)
        # Activation
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(keys_empty_cond, paddings, outputs)
        weighted = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        weighted_sum = tf.matmul(weighted, multi_seqs)  # [B, 1, H]
        weighted_sum = tf.reshape(weighted_sum, [-1, multi_seqs.get_shape().as_list()[-1]])

    return weighted_sum


def aux_loss_mhsa(inputs, num_units, num_heads, dropout_rate, name="", is_training=True, is_layer_norm=True):
    """
    Args:
        inputs: [B*T, TRIANGLE_NUM, H]
    Output:
        outputs:[B*T, TRIANGLE_NUM, H]
    """ 
    Q_K_V = tf.layers.dense(inputs, 3 * num_units)
    Q, K, V = tf.split(Q_K_V, 3, -1)
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) 
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) 
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) 
    
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) 
    align= outputs / (36 ** 0.5)
    # Diag mask 
    diag_val = tf.ones_like(align[0, :, :]) 
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense() 
    
    key_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1]) 
    padding = tf.ones_like(key_masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), padding, align)
    outputs = tf.nn.softmax(outputs) 
    
    # Attention Matmul
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    outputs = tf.matmul(outputs,V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
    outputs = tf.layers.dense(outputs, num_units)
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    
    # Residual connection
    outputs += inputs
    # Normalize
    if is_layer_norm:
        outputs = layer_norm(outputs,name=name)
    return outputs
