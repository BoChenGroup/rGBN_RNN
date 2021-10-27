#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:21:44 2018

@author: cb413
"""

import numpy as np
import tensorflow as tf
import pickle

class DPGDS_Model():
    def __init__(self, config, V_tm, V_lm):

#######################################DPGDS_PARAM###############################################

        self.V_dim = V_tm  ## DPGDS vocabulary lenth
        self.H_dim = config.H     ## DPGDS encoder dimension
        self.K_dim = config.K     ## DPGDS topic dimension
        self.Batch_Size = config.Batch_Size
        self.L = config.DPGDS_Layer   ## DPGDS layer
        self.real_min = config.real_min

        self.seed = config.seed
        self.W_1, self.W_2, self.W_3, self.W_4, self.W_5, self.W_6, self.W_1_k, self.b_1_k, \
        self.W_1_l, self.b_1_l, self.W_3_k, self.b_3_k, self.W_3_l, self.b_3_l, self.W_5_k, \
        self.b_5_k, self.W_5_l, self.b_5_l = self.initialize_weight()

        self.Doc_input = tf.placeholder("float32", shape=[self.Batch_Size, config.sent_J, self.V_dim])  # N*J*V
        self.phi_1 = tf.placeholder(tf.float32, shape=[self.V_dim, self.K_dim[0]])  # N*V
        self.phi_2 = tf.placeholder(tf.float32, shape=[self.K_dim[0], self.K_dim[1]])  # N*V
        self.phi_3 = tf.placeholder(tf.float32, shape=[self.K_dim[1], self.K_dim[2]])  # N*V
        self.pi_1 = tf.placeholder(tf.float32, shape=[self.K_dim[0], self.K_dim[0]])  # N*V
        self.pi_2 = tf.placeholder(tf.float32, shape=[self.K_dim[1], self.K_dim[1]])  # N*V
        self.pi_3 = tf.placeholder(tf.float32, shape=[self.K_dim[2], self.K_dim[2]])  # N*V

        self.state1 = tf.zeros([self.Batch_Size, self.H_dim[0]], dtype=tf.float32)
        self.state2 = tf.zeros([self.Batch_Size, self.H_dim[1]], dtype=tf.float32)
        self.state3 = tf.zeros([self.Batch_Size, self.H_dim[2]], dtype=tf.float32)

        self.theta_1C_HT_each = [];  self.theta_2C_HT_each = [];  self.theta_3C_HT_each = []
        self.h1 = [];    self.h2 = [];    self.h3 = []
        self.LB = 0;  self.L1 = 0

#######################################LSTM_PARAM###############################################


        self.droprate = tf.placeholder(tf.float32,[])

        for j in range(config.sent_J):
            input_X = self.Doc_input[:,j,:]  ### N*V
            ##################  DPGDS layer1  ########################
            # self.state1 = tf.sigmoid(tf.matmul(input_X, self.W_1))
            self.state1 = tf.sigmoid(tf.matmul(input_X, self.W_1)+ tf.matmul(self.state1, self.W_2))
            self.k_1, self.l_1 = self.Encoder_Weilbull(self.state1, 0, self.W_1_k, self.b_1_k, self.W_1_l, self.b_1_l)  # K*N,  K*N
            theta_1, theta_1c = self.reparameterization(self.k_1, self.l_1, 0, self.Batch_Size)  # K * N batch_size = 20
            ##################  DPGDS layer2  ########################
            # self.state2 = tf.sigmoid(tf.matmul(self.state1, self.W_3) )
            self.state2 = tf.sigmoid(tf.matmul(self.state1, self.W_3) + tf.matmul(self.state2, self.W_4))
            self.k_2, self.l_2 = self.Encoder_Weilbull(self.state2, 1, self.W_3_k, self.b_3_k, self.W_3_l, self.b_3_l)
            theta_2, theta_2c = self.reparameterization(self.k_2, self.l_2, 1, self.Batch_Size)
            ##################  DPGDS layer3  ########################
            # self.state3 = tf.sigmoid(tf.matmul(self.state2, self.W_5))
            self.state3 = tf.sigmoid(tf.matmul(self.state2, self.W_5) + tf.matmul(self.state3, self.W_6))
            self.k_3, self.l_3 = self.Encoder_Weilbull(self.state3, 2, self.W_5_k, self.b_5_k, self.W_5_l, self.b_5_l)
            theta_3, theta_3c = self.reparameterization(self.k_3, self.l_3, 2, self.Batch_Size)


            if j == 0:
                alpha_1_t = tf.matmul(self.phi_2, theta_2)
                alpha_2_t = tf.matmul(self.phi_3, theta_3)
                alpha_3_t = tf.ones([self.K_dim[2], self.Batch_Size], dtype='float32')  # K * 1
            else:
                alpha_1_t = tf.matmul(self.phi_2, theta_2) + tf.matmul(self.pi_1, theta_left_1)
                alpha_2_t = tf.matmul(self.phi_3, theta_3) + tf.matmul(self.pi_2, theta_left_2)
                alpha_3_t = tf.matmul(self.pi_3, theta_left_3)

            L1_1_t = (tf.transpose(input_X)) * self.log_max_tf(tf.matmul(self.phi_1, theta_1)) - tf.matmul(self.phi_1, theta_1)  # - tf.lgamma( X_VN_t + 1)
            theta1_KL = tf.reduce_sum(self.KL_GamWei(alpha_1_t, np.float32(1.0), self.k_1, self.l_1))
            theta2_KL = tf.reduce_sum(self.KL_GamWei(alpha_2_t, np.float32(1.0), self.k_2, self.l_2))
            theta3_KL = tf.reduce_sum(self.KL_GamWei(alpha_3_t, np.float32(1.0), self.k_3, self.l_3))

            self.LB = self.LB + (1 * tf.reduce_sum(L1_1_t) + 0.1 * theta1_KL + 0.01 * theta2_KL + 0.001 * theta3_KL)/self.Batch_Size
            self.L1 = self.L1 + tf.reduce_sum(L1_1_t)/self.Batch_Size

            theta_left_1 = theta_1
            theta_left_2 = theta_2
            theta_left_3 = theta_3

            self.theta_1C_HT_each.append(theta_1c)
            self.theta_2C_HT_each.append(theta_2c)
            self.theta_3C_HT_each.append(theta_3c)

        self.tm_Loss = - self.LB

        Optimizer = tf.train.AdamOptimizer(config.learning_rate)
        threshold = 1
        grads_vars = Optimizer.compute_gradients(self.tm_Loss)
        capped_gvs = []
        for grad, var in grads_vars:
            if grad is not None:
                grad = tf.where(tf.is_nan(grad), threshold * tf.ones_like(grad), grad)
                grad = tf.where(tf.is_inf(grad), threshold * tf.ones_like(grad), grad)
                capped_gvs.append((tf.clip_by_value(grad, -threshold, threshold), var))
        self.tm_train_step = Optimizer.apply_gradients(capped_gvs)

        self.theta_1C_HT = tf.transpose(self.theta_1C_HT_each, [1, 2, 0])
        self.theta_2C_HT = tf.transpose(self.theta_2C_HT_each, [1, 2, 0])
        self.theta_3C_HT = tf.transpose(self.theta_3C_HT_each, [1, 2, 0])


    def weight_variable(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32))


    def bias_variable(self,shape):
        return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32))


    def reparameterization(self, Wei_shape, Wei_scale, l, Batch_Size):
        eps = tf.random_uniform(shape=[np.int32(self.K_dim[l]), Batch_Size], dtype=tf.float32)  # K_dim[i] * none
        # eps = tf.ones(shape=[np.int32(self.K_dim[l]), Batch_Size], dtype=tf.float32) /2 # K_dim[i] * none
        theta = Wei_scale * tf.pow(-self.log_max_tf(1 - eps), 1 / Wei_shape)
        theta_c = tf.transpose(theta)
        return theta, theta_c  # K*N    N*K

    def Encoder_Weilbull(self,input_x, l, W_k, b_k, W_l, b_l):  # i = 0:T-1 , input_x N*V
        # feedforward
        k_tmp = tf.nn.softplus(tf.matmul(input_x, W_k) + b_k)  # none * 1
        k_tmp = tf.tile(k_tmp,
                        [1, self.K_dim[l]])  # reshpe   ????                                             # none * K_dim[i]
        k = tf.maximum(k_tmp, self.real_min)
        lam = tf.nn.softplus(tf.matmul(input_x, W_l) + b_l)  # none * K_dim[i]
        return tf.transpose(k), tf.transpose(lam)


    def log_max_tf(self,input_x):
        return tf.log(tf.maximum(input_x, self.real_min))


    def KL_GamWei(self,Gam_shape, Gam_scale, Wei_shape, Wei_scale):  # K_dim[i] * none
        eulergamma = 0.5772
        KL_Part1 = eulergamma * (1 - 1 / Wei_shape) + self.log_max_tf(Wei_scale / Wei_shape) + 1 + Gam_shape * self.log_max_tf(
            Gam_scale)
        KL_Part2 = -tf.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max_tf(Wei_scale) - eulergamma / Wei_shape)
        KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * tf.exp(tf.lgamma(1 + 1 / Wei_shape))
        return KL

    def initialize_weight(self):
        V_dim = self.V_dim
        H_dim = self.H_dim
        K_dim = self.K_dim
        W_1 = self.weight_variable(shape=[V_dim, H_dim[0]])
        W_2 = tf.Variable(tf.eye(H_dim[0], dtype=tf.float32))
        W_3 = self.weight_variable(shape=[H_dim[0], H_dim[1]])
        W_4 = tf.Variable(tf.eye(H_dim[1], dtype=tf.float32))
        W_5 = self.weight_variable(shape=[H_dim[1], H_dim[2]])
        W_6 = tf.Variable(tf.eye(H_dim[2], dtype=tf.float32))
        W_1_k = self.weight_variable(shape=[H_dim[0], 1])
        b_1_k = self.bias_variable(shape=[1])
        W_1_l = self.weight_variable(shape=[H_dim[0], K_dim[0]])
        b_1_l = self.bias_variable(shape=[K_dim[0]])
        W_3_k = self.weight_variable(shape=[H_dim[1], 1])
        b_3_k = self.bias_variable(shape=[1])
        W_3_l = self.weight_variable(shape=[H_dim[1], K_dim[1]])
        b_3_l = self.bias_variable(shape=[K_dim[1]])
        W_5_k = self.weight_variable(shape=[H_dim[2], 1])
        b_5_k = self.bias_variable(shape=[1])
        W_5_l = self.weight_variable(shape=[H_dim[2], K_dim[2]])
        b_5_l = self.bias_variable(shape=[K_dim[2]])
        return W_1, W_2, W_3, W_4, W_5, W_6, W_1_k, b_1_k, W_1_l, b_1_l, W_3_k, b_3_k, W_3_l, b_3_l, W_5_k, b_5_k, W_5_l, b_5_l





