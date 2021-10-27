# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Oct  7 15:48:23 2017
# rengong zhineng hexin daima , jaizhi yigeyi
# """
# # from sample import Sampler
# # GSL_sampler = Sampler()
#
# import scipy.io as sio
# # import matplotlib.pyplot as plt
# # from collections import defaultdict
# # import PGBN_sampler
import random
import time
import os
import config as cf
import config_generate as  cf_generate
import numpy as np
import pickle
import tensorflow as tf
from DPGDS_sampler import MyThread
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim.models as g
from data_process_all import gen_vocab, gen_data, Get_TM_vocab, Bag_of_words, get_batches, \
    init_embedding, get_batches_tm_lm,get_batches_lm,get_sent_num,get_batch_tm_lm_all,get_doc_sent
import PGBN_sampler
from DPGDS_LSTM_test import DPGDS_LSTM_Model


####
# main#
###

# set the seeds
random.seed(cf.seed)
np.random.seed(cf.seed)

# globals
vocabxid = {}
idxvocab = []

# constants
pad_symbol = "<pad>"
start_symbol = "<go>"
end_symbol = "<eos>"
unk_symbol = "<unk>"
dummy_symbols = [pad_symbol, start_symbol, end_symbol, unk_symbol]

# # first pass to collect vocabulary information
# print("First pass on train corpus to collect vocabulary stats...")
# idxvocab, vocabxid, tm_ignore = gen_vocab(dummy_symbols, cf.train_corpus, cf.stopwords, cf.vocab_minfreq,
#                                           cf.vocab_maxfreq, cf.verbose)
# TM_vocab = Get_TM_vocab(idxvocab, tm_ignore)
# vocab_save = open('D:/PGBN_LSTM/Data/apnews/apnews_Vocab_data1.pckl','wb')
# pickle.dump([idxvocab, vocabxid, tm_ignore, TM_vocab], vocab_save)
# vocab_save.close()
#
# # second pass to collect train/valid data for topic and language model
# print("Processing train corpus to collect sentence and document data...")
# train_sents, train_docs, train_docids, train_stats = gen_data(vocabxid, dummy_symbols, tm_ignore, cf.train_corpus,
#                                                               cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)
# train_data_save = open('D:/PGBN_LSTM/Data/apnews/apnews_train_data1.pckl','wb')
# pickle.dump([train_sents, train_docs, train_docids, train_stats], train_data_save)
# train_data_save.close()
#
# print("Processing valid corpus to collect sentence and document data...")
# valid_sents, valid_docs, valid_docids, valid_stats = gen_data(vocabxid, dummy_symbols, tm_ignore, cf.valid_corpus,
#                                                               cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)
# valid_data_save = open('D:/PGBN_LSTM/Data/apnews/apnews_valid_data1.pckl','wb')
# pickle.dump([valid_sents, valid_docs, valid_docids, valid_stats], valid_data_save)
# valid_data_save.close()
#
# print("Processing test corpus to collect sentence and document data...")
# test_sents, test_docs, test_docids, test_stats = gen_data(vocabxid, dummy_symbols, tm_ignore, cf.test_corpus,
#                                                               cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)
# test_data_save = open('D:/PGBN_LSTM/Data/apnews/apnews_test_data1.pckl','wb')
# pickle.dump([test_sents, test_docs, test_docids, test_stats], test_data_save)
# test_data_save.close()

vocab_save = open('D:/PGBN_LSTM/Data/apnews/apnews_Vocab_data1.pckl','rb')
[idxvocab, vocabxid, tm_ignore, TM_vocab] = pickle.load(vocab_save)
vocab_save.close()

train_data_save = open('D:/PGBN_LSTM/Data/apnews/apnews_train_data1.pckl','rb')
[train_sents, train_docs, train_docids, train_stats] = pickle.load(train_data_save)
train_data_save.close()

# valid_data_save = open('D:/PGBN_LSTM/Data/apnews/apnews_valid_data1.pckl','rb')
# [valid_sents, valid_docs, valid_docids, valid_stats] = pickle.load(valid_data_save)
# valid_data_save.close()

test_data_save = open('D:/PGBN_LSTM/Data/apnews/apnews_test_data1.pckl','rb')
[test_sents, test_docs, test_docids, test_stats] = pickle.load(test_data_save)
test_data_save.close()



if cf.dataname == 'apnews' or cf.dataname == 'bnc':
    TM_train_doc, train_doc_bow = Bag_of_words(train_docs[0], idxvocab, tm_ignore)
    # TM_validation_doc, validation_doc_bow = Bag_of_words(valid_docs[0], idxvocab, tm_ignore)
    TM_test_doc, test_doc_bow = Bag_of_words(test_docs[0], idxvocab, tm_ignore)
    # sio.savemat("data/"+ cf.dataname + '/' + cf.dataname + '_TM_data.mat', {'TM_vocab': TM_vocab, 'train_data': train_data, 'validation_data': validation_data,'test_data': test_data})
# elif cf.dataname == 'imdb':  # imdb traindata is too big to save
#     TM_train_doc1, train_data1 = Bag_of_words(train_docs[0][:50000], idxvocab, tm_ignore)
#     TM_train_doc2, train_data2 = Bag_of_words(train_docs[0][50000:], idxvocab, tm_ignore)
#     TM_validation_doc, validation_data = Bag_of_words(valid_docs[0], idxvocab, tm_ignore)
#     TM_test_doc, test_data = Bag_of_words(test_docs[0], idxvocab, tm_ignore)
#     # sio.savemat("data/"+ cf.dataname + '/' + cf.dataname + '_TM_data.mat', {'TM_vocab': TM_vocab, 'train_data_1': train_data1, 'train_data_2': train_data2, 'validation_data': validation_data,'test_data': test_data})
# else:
#     print("There is another dataset ! ")

print('-----------------------------data ----------load---------------------------------------- ')

################################################################### data prepare ##############################################################################

test_sent_num = get_sent_num(test_sents[1],len(test_doc_bow[0]))
test_Doc = get_doc_sent(test_sents[1], test_doc_bow, test_sent_num, cf.sent_J)

doc_num_batches_test = int(np.floor(float(len(test_Doc)) / cf.Batch_Size))
batch_ids_test = [item for item in range(doc_num_batches_test)]

graph = tf.get_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) # tf.Graph().as_default()
e = 0
batch_id = 5000

V_tm = len(idxvocab) - len(tm_ignore)
PHI_save = open(os.path.join(cf.output_dir, cf.output_path, cf.dataname + str(e) + '_Phi_Pi_' + str(batch_id) + '.pckl'), 'rb')
[Phi,Pi,Hidden] = pickle.load( PHI_save )
PHI_save.close()

print(Phi[0][0][10])


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    tf.set_random_seed(cf.seed)
    gener = DPGDS_LSTM_Model(cf_generate, V_tm, len(vocabxid))

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)
    saver.restore(sess, os.path.join(cf.output_dir, cf.output_prefix, cf.dataname + str(e) + "_model_" + str(batch_id) + ".ckpt"))

########## generate sentence from topics ######################
    test_lm_costs = 0
    test_lm_words = 0
    test_theta = []

    # sentence = gener.generate_on_topic( sess, config=cf_generate, Phi=Phi, topic_id = 9, start_word_id = 1, temperature=0.75, max_length=30, stop_word_id=2)
    sentence = gener.generate_on_co_topic( sess, config=cf_generate, topic_id = [4,5,6], start_word_id = 1, temperature=0.75, max_length=30, stop_word_id=2)
    sents = ''
    for id in sentence:
        sents += idxvocab[id]
        sents += ' '
    print(sentence)
    print(sents)

  # for batch_id_t in batch_ids_test:
    #     Doc_test = test_Doc[(batch_id_t * cf.Batch_Size):((batch_id_t + 1) * cf.Batch_Size)]
    #     X_test_batch, y_test_batch, d_test_batch, m_test_batch = get_batch_tm_lm_all(Doc_test, test_doc_bow,
    #                                                                                  len(idxvocab),
    #                                                                                  tm_ignore,
    #                                                                                  cf.Batch_Size)
    #     curr_cost_test, tm_cost_test, lm_cost_test, Theta_test = sess.run(
    #         [gener.joint_Loss, gener.tm_Loss, gener.lm_Loss,
    #          [gener.theta_1C_HT, gener.theta_2C_HT, gener.theta_3C_HT]],
    #         feed_dict={gener.Doc_input: d_test_batch, gener.phi_1: Phi[0], gener.phi_2: Phi[1], gener.phi_3: Phi[2],
    #                    gener.pi_1: Pi[0], gener.pi_2: Pi[1], gener.pi_3: Pi[2],
    #                    gener.Sent_input: X_test_batch, gener.Sent_output: y_test_batch,
    #                    gener.lm_mask: m_test_batch, gener.is_training: False})
    #
    #
    #     test_lm_costs += lm_cost_test * cf.Batch_Size
    #     test_lm_words += np.sum(m_test_batch)
    #     test_theta.append(Theta_test)  ####### Theta: 3*   N*K*J
    # print("test language model perplexity = %.3f" % (np.exp(test_lm_costs / test_lm_words)))

