#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:21:44 2018

@author: cb413
"""

import numpy as np
import tensorflow as tf
import pickle

class DPGDS_LSTM_Model():
    def __init__(self, config, V_tm, V_lm):

#######################################DPGDS_PARAM###############################################
        # self.ForgetRate = np.power((config.Setting['tao0FR'] + np.linspace(1, config.Setting['Iterall'], config.Setting['Iterall'])),
        #                       -config.Setting['kappa0FR'])
        # epsit = np.power((config.Setting['tao0'] + np.linspace(1, config.Setting['Iterall'], config.Setting['Iterall'])), -config.Setting['kappa0'])
        # self.epsit = config.Setting['epsi0'] * epsit / epsit[0]

        # params
        self.V_dim = V_tm  ## DPGDS vocabulary lenth
        self.H_dim = config.H     ## DPGDS encoder dimension
        self.K_dim = config.K     ## DPGDS topic dimension
        self.Batch_Size = config.Batch_Size
        self.L = config.DPGDS_Layer   ## DPGDS layer
        self.real_min = config.real_min

        self.seed = config.seed
        self.W_1, self.W_2, self.W_1_k, self.b_1_k, self.W_1_l, self.b_1_l = self.initialize_weight()

        self.Doc_input = tf.placeholder("float32", shape=[self.Batch_Size, config.sent_J, self.V_dim])  # N*J*V
        self.phi_1 = tf.placeholder(tf.float32, shape=[self.V_dim, self.K_dim[0]])  # N*V
        self.pi_1 = tf.placeholder(tf.float32, shape=[self.K_dim[0], self.K_dim[0]])  # N*V

        self.state1 = tf.zeros([self.Batch_Size, self.H_dim[0]], dtype=tf.float32)

        self.theta_1C_HT = []
        self.h1 = []
        self.LB = 0
        self.L1 = 0

#######################################LSTM_PARAM###############################################

        self.Sent_input = tf.placeholder(tf.int32, [self.Batch_Size, config.sent_J, config.lm_sent_len])
        self.lm_mask = tf.placeholder(tf.float32,  [self.Batch_Size, config.sent_J, config.lm_sent_len])
        self.Sent_output = tf.placeholder(tf.int32,[self.Batch_Size, config.sent_J, config.lm_sent_len])

        self.is_training = tf.placeholder(tf.bool,[])
        self.droprate = tf.placeholder(tf.float32,[])

        self.vocab_size = V_lm
        self.LSTM_word_embedding = tf.get_variable("lstm_embedding", [self.vocab_size, config.word_embedding_size],
                                              trainable=config.word_embedding_update,
                                              initializer=tf.random_uniform_initializer(-0.5 / config.word_embedding_size,
                                                                                        0.5 / config.word_embedding_size))
        self.lstm_inputs = tf.nn.embedding_lookup(self.LSTM_word_embedding, self.Sent_input)
        # if self.is_training is True and config.lm_keep_prob < 1.0:
        #     self.lstm_inputs = tf.nn.dropout(self.lstm_inputs, config.lm_keep_prob, seed=self.seed)
        self.lstm_inputs = tf.nn.dropout(self.lstm_inputs, self.droprate, seed=self.seed)

        self.hidden1 = []
        self.cell_1 = tf.nn.rnn_cell.BasicLSTMCell(config.rnn_hidden_size1)
        self.Uz1, self.Ur1, self.Uh1, self.Wz1, self.Wr1, self.Wh1, self.bz1, self.br1, self.bh1 = self.GRU_params(config.theta_size1,config.rnn_hidden_size1,config.rnn_bias)

        for j in range(config.sent_J):
            input_X = self.Doc_input[:,j,:]  ### N*V
            ##################  DPGDS layer1  ########################
            if self.is_training is True:
                self.state1 = tf.sigmoid(tf.matmul(input_X, self.W_1) + tf.matmul(self.state1, self.W_2))
            else:
                self.state1 = tf.sigmoid(tf.matmul(input_X, self.W_1))
                # self.state1 = tf.zeros([self.Batch_Size, self.H_dim[0]], dtype=tf.float32)
            # self.state1 = tf.sigmoid(tf.matmul(input_X, self.W_1) + tf.matmul(self.state1, self.W_2))
            self.k_1, self.l_1 = self.Encoder_Weilbull(self.state1, 0, self.W_1_k, self.b_1_k, self.W_1_l, self.b_1_l)  # K*N,  K*N
            theta_1, theta_1c = self.reparameterization(self.k_1, self.l_1, 0, self.Batch_Size)  # K * N batch_size = 20

            self.h1.append(self.state1)
            if j == 0:
                alpha_1_t = tf.ones([self.K_dim[0], self.Batch_Size], dtype='float32')
            else:
                if self.is_training is not True:
                    alpha_1_t = tf.ones([self.K_dim[0], self.Batch_Size], dtype='float32')
                else:
                    alpha_1_t = tf.matmul(self.pi_1, theta_left_1)

            L1_1_t = (tf.transpose(input_X)) * self.log_max_tf(tf.matmul(self.phi_1, theta_1)) - tf.matmul(self.phi_1, theta_1)  # - tf.lgamma( X_VN_t + 1)
            theta1_KL = tf.reduce_sum(self.KL_GamWei(alpha_1_t, np.float32(1.0), self.k_1, self.l_1))

            self.LB = self.LB + (1 * tf.reduce_sum(L1_1_t) + 0.1 * theta1_KL)
            self.L1 = self.L1 + tf.reduce_sum(L1_1_t)

            theta_left_1 = theta_1
            self.theta_1c = theta_1c/tf.maximum(tf.tile(tf.reshape(tf.reduce_max(theta_1c,axis=1),[self.Batch_Size,1]),[1,self.K_dim[0]]), config.real_min)
            self.theta_1C_HT.append(self.theta_1c)

            self.initial_state_t_1 = self.cell_1.zero_state(config.Batch_Size, tf.float32)

            for t in range(config.lm_sent_len):
                x_t = self.lstm_inputs[:, j, t, :]
                ##################  LSTM layer1  ########################
                if t==0:
                    self.H_t_1,self.state_t_1 = self.cell_1(x_t,self.initial_state_t_1)
                else:
                    self.H_t_1, self.state_t_1 = self.cell_1(x_t,self.state_t_1)
                self.h_t_1_theta = self.GRU_theta_hidden(self.theta_1c, self.H_t_1, self.Uz1, self.Ur1, self.Uh1, self.Wz1, self.Wr1, self.Wh1, self.bz1, self.br1, self.bh1)
                # if self.is_training is True and config.lm_keep_prob < 1.0:
                #     h_t_1_drop = tf.nn.dropout(self.h_t_1_theta, config.lm_keep_prob, seed=self.seed )
                #     print('dropout used:',self.lm_keep_prob)
                # else:
                #     h_t_1_drop = self.h_t_1_theta
                #     print('dropout not used')
                h_t_1_drop = tf.nn.dropout(self.h_t_1_theta, self.droprate, seed=self.seed )
                self.hidden1.append(h_t_1_drop)

        self.theta_1C_HT = tf.transpose(self.theta_1C_HT,[1,2,0])

        #### transpose hidden[8*30,8,600] to hidden_t[8,8*30,600]
        hidden_all = self.hidden1
        self.hidden = tf.transpose(hidden_all,perm=[1, 0, 2])
        hidden_size = config.rnn_hidden_size1
        self.lm_softmax_w = tf.get_variable("lm_softmax_w", [hidden_size, self.vocab_size])
        self.lm_softmax_b = tf.get_variable("lm_softmax_b", [self.vocab_size], initializer=tf.constant_initializer())
        self.lm_logits = tf.matmul(tf.reshape(self.hidden, [-1, hidden_size]), self.lm_softmax_w) + self.lm_softmax_b

        self.tm_Loss = - self.LB
        self.tm_train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(self.tm_Loss)

        lm_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.Sent_output, [-1]), logits=self.lm_logits)
        lm_crossent_m = lm_crossent * tf.reshape(self.lm_mask, [-1])
        self.lm_Loss = tf.reduce_sum(lm_crossent_m) / self.Batch_Size
        lm_tvars = tf.trainable_variables()
        lm_grads, _ = tf.clip_by_global_norm(tf.gradients(self.lm_Loss, lm_tvars), config.max_grad_norm)
        self.lm_train_step= tf.train.AdamOptimizer(config.learning_rate).apply_gradients(zip(lm_grads, lm_tvars))
        # self.lm_train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(self.lm_Loss)
        # self.tm_Loss +
        self.joint_Loss = self.lm_Loss

        jm_tvars = tf.trainable_variables()
        jm_grads, _ = tf.clip_by_global_norm(tf.gradients(self.joint_Loss, jm_tvars), config.max_grad_norm)
        self.joint_train_step = tf.train.AdamOptimizer(config.learning_rate).apply_gradients(zip(jm_grads, jm_tvars))
        # self.joint_train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(self.joint_Loss)



    def GRU_params(self,input_size,output_size,rnn_bias):
        Uz = self.rnn_weight_variable([input_size, output_size])
        Ur = self.rnn_weight_variable([input_size, output_size])
        Uh = self.rnn_weight_variable([input_size, output_size])
        Wz = self.rnn_weight_variable([output_size, output_size])
        Wr = self.rnn_weight_variable([output_size, output_size])
        Wh = self.rnn_weight_variable([output_size, output_size])
        bz = self.rnn_bias_variable(rnn_bias, [output_size, ])
        br = self.rnn_bias_variable(rnn_bias, [output_size, ])
        bh = self.rnn_bias_variable(rnn_bias, [output_size, ])
        return Uz,Ur,Uh,Wz,Wr,Wh,bz,br,bh
    def GRU_theta_hidden(self,t_t,h_t_1,Uz,Ur,Uh,Wz,Wr,Wh,bz,br,bh):
        z = tf.nn.sigmoid(tf.matmul(t_t, Uz) + tf.matmul(h_t_1, Wz) + bz)
        r = tf.nn.sigmoid(tf.matmul(t_t, Ur) + tf.matmul(h_t_1, Wr) + br)
        h = tf.nn.tanh(tf.matmul(t_t, Uh) + tf.matmul(tf.multiply(h_t_1, r), Wh)  + bh)
        s_t = tf.multiply(1 - z, h) + tf.multiply(z, h_t_1)
        return s_t


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
        W_1_k = self.weight_variable(shape=[H_dim[0], 1])
        b_1_k = self.bias_variable(shape=[1])
        W_1_l = self.weight_variable(shape=[H_dim[0], K_dim[0]])
        b_1_l = self.bias_variable(shape=[K_dim[0]])


        return W_1, W_2, W_1_k, b_1_k, W_1_l, b_1_l



    def rnn_weight_variable(self,shape):
        return tf.Variable(tf.random_uniform(shape, minval=-0.05, maxval=0.05, seed = self.seed, dtype=tf.float32),
                           trainable=True)


    def rnn_bias_variable(self, bias, shape):
        return tf.Variable(tf.constant(bias, shape=shape, dtype=tf.float32), trainable=True)

    def softmax(self,x):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, keepdims=True)
        s = x_exp / x_sum
        return s

    def sample(self, probs, temperature):
        temperature=0.75
        if temperature == 0:
            return np.argmax(probs)
        probs = probs.astype(np.float64) #convert to float64 for higher precision
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        probs = np.reshape(probs,[probs.shape[1],])
        return np.argmax(np.random.multinomial(1, probs, 1))


    #generate a sentence given conv_hidden
    def generate(self, sess,config, Theta, start_word_id, temperature, max_length, stop_word_id):
        state_t_1 = sess.run(self.cell_1.zero_state(1, tf.float32))
        # state_t_1 = (np.ones(1, config.rnn_hidden_size1),np.ones(1, config.rnn_hidden_size1))
        x = [[start_word_id]]
        sent = [start_word_id]
        theta1 = np.reshape(Theta[0],[1,config.K[0]])
        for _ in range(max_length):
            lm_logits, state_t_1= sess.run([self.lm_logits, self.state_t_1],
                    {self.Sent_input: np.reshape(x,[1,1,1]), self.initial_state_t_1: state_t_1,
                     self.theta_1c: theta1})

            lm_logits_exp = np.exp(lm_logits)
            lm_logits_sum = np.sum(lm_logits_exp, keepdims=True)
            probs = lm_logits_exp / lm_logits_sum
            sent.append(self.sample(probs, temperature))
            if sent[-1] == stop_word_id:
                break
            x = [[ sent[-1] ]]
        return sent

    #generate a sequence of words, given a topic
    def generate_on_topic(self, sess, config, topic_id, start_word_id, temperature=0.75, max_length=30, stop_word_id=None):
        index = topic_id
        theta = []
        for layer in range(1):
            thet = np.zeros(config.K[layer])
            thet[index] = 1
            theta.append(thet)
        Theta = [theta[0]]
        # num = 1
        # theta1 = np.zeros([config.K[0]])
        # theta1[topic_id[0]] = num
        # theta2 = np.zeros([config.K[1]])
        # theta2[topic_id[1]] = num
        # theta3 = np.zeros([config.K[2]])
        # theta3[topic_id[2]] = num
        # Theta = [theta1, theta2, theta3]

        return self.generate(sess,config, Theta, start_word_id, temperature, max_length, stop_word_id)

    #generate a sequence of words, given a document
    def generate_on_doc(self, sess, Theta, start_word_id, temperature=1.0, max_length=30, stop_word_id=None):
        return self.generate(sess, Theta, start_word_id, temperature, max_length, stop_word_id)
