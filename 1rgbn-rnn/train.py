# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Oct  7 15:48:23 2017
# """

# from sample import Sampler
# GSL_sampler = Sampler()
# import scipy.io as sio
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import PGBN_sampler
# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim.models as g
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import time
import os
import config as cf
import numpy as np
import pickle
import tensorflow as tf
from DPGDS_sampler import MyThread
from data_process_all import gen_vocab, gen_data, Get_TM_vocab, Bag_of_words, get_batches, \
    init_embedding, get_batches_tm_lm,get_batches_lm,get_sent_num,get_batch_tm_lm_all,get_doc_sent
import PGBN_sampler
from model_layer3_different_stack import DPGDS_LSTM_Model

def initialize_Phi_Pi(V_tm):
    Phi = [0] * cf.DPGDS_Layer
    Pi = [0] * cf.DPGDS_Layer
    NDot_Phi = [0] * cf.DPGDS_Layer

    NDot_Pi = [0] * cf.DPGDS_Layer
    for l in range(cf.DPGDS_Layer):
        if l == 0:
            Phi[l] = np.random.rand(V_tm, cf.K[l])
        else:
            Phi[l] = np.random.rand(cf.K[l - 1], cf.K[l])

        Phi[l] = Phi[l] / np.sum(Phi[l], axis=0)
        Pi[l] = np.eye(cf.K[l])
    return Phi, Pi, NDot_Phi, NDot_Pi


def update_Pi_Phi(miniBatch, Phi,Pi, Theta, MBratio, MBObserved,NDot_Phi,NDot_Pi):

    ForgetRate = np.power((cf.Setting['tao0FR'] + np.linspace(1, cf.Setting['Iterall'], cf.Setting['Iterall'])),
                          -cf.Setting['kappa0FR'])
    epsit = np.power((cf.Setting['tao0'] + np.linspace(1, cf.Setting['Iterall'], cf.Setting['Iterall'])), -cf.Setting['kappa0'])
    epsit = cf.Setting['epsi0'] * epsit / epsit[0]

    L = cf.DPGDS_Layer
    A_VK = [0]* L
    L_KK = [0]* L
    Piprior = [0]* L
    EWSZS_Phi = [0]* L
    EWSZS_Pi = [0]* L

    Xi = []
    Vk = []
    for l in range(L):
        Xi.append(1)
        Vk.append(np.ones((cf.K[l], 1)))

    threads = []

    # 循环minibatch更新L_KK,A_VK
    for i in range(cf.Batch_Size):
        Theta1 = Theta[0][ i, :, :]
        Theta2 = Theta[1][ i, :, :]
        Theta3 = Theta[2][ i, :, :]

        t = MyThread(i, np.transpose(miniBatch[i, :, :]), Phi, Theta1, Theta2, Theta3, L, cf.K , cf.sent_J, Pi)
        threads.append(t)
    for t in threads:
        t.setDaemon(True)
        t.start()
    for t in threads:
        t.join()
    for t in threads:
        AA, BB, CC = t.get_result()
        for l in range(L):
            A_VK[l] = A_VK[l] + BB[l]
            L_KK[l] = L_KK[l] + CC[l]

    for l in range(len(Phi)):
        EWSZS_Phi[l] = MBratio * A_VK[l]
        EWSZS_Pi[l] = MBratio * L_KK[l]

        if (MBObserved == 0):
            NDot_Phi[l] = EWSZS_Phi[l].sum(0)
            NDot_Pi[l] = EWSZS_Pi[l].sum(0)
        else:
            NDot_Phi[l] = (1 - ForgetRate[MBObserved]) * NDot_Phi[l] + ForgetRate[MBObserved] * EWSZS_Phi[l].sum(
                0)  # 1*K
            NDot_Pi[l] = (1 - ForgetRate[MBObserved]) * NDot_Pi[l] + ForgetRate[MBObserved] * EWSZS_Pi[l].sum(0)  # 1*K
        # Sample Phi
        tmp = EWSZS_Phi[l] + cf.eta0  # V*K
        tmp = (1 / np.maximum(NDot_Phi[l], cf.real_min)) * (tmp - tmp.sum(0) * Phi[l])  # V*K
        tmp1 = (2 / np.maximum(NDot_Phi[l], cf.real_min)) * Phi[l]
        tmp = Phi[l] + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(Phi[l].shape[0],Phi[l].shape[1])
        Phi[l] = PGBN_sampler.ProjSimplexSpecial(tmp, Phi[l], 0)

        # Sample Pi
        Piprior[l] = np.dot(Vk[l], np.transpose(Vk[l]))
        Piprior[l][np.arange(Piprior[l].shape[0]), np.arange(Piprior[l].shape[1])] = 0
        Piprior[l] = Piprior[l] + np.diag(np.reshape(Xi[l] * Vk[l], Vk[l].shape[0], 1))

        tmp = EWSZS_Pi[l] + Piprior[l]  # V*K
        tmp = (1 / np.maximum(NDot_Pi[l], cf.real_min)) * (tmp - tmp.sum(0) * Pi[l])  # V*K
        tmp1 = (2 / np.maximum(NDot_Pi[l], cf.real_min)) * Pi[l]
        tmp = Pi[l] + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(Pi[l].shape[0],
                                                                                                    Pi[l].shape[1])
        Pi[l] = PGBN_sampler.ProjSimplexSpecial(tmp, Pi[l], 0)

    return Phi, Pi, NDot_Phi, NDot_Pi

def log_max(input_x):
    return tf.log(tf.maximum(input_x, cf.real_min))


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
#
# # second pass to collect train/valid data for topic and language model
# print("Processing train corpus to collect sentence and document data...")
# train_sents, train_docs, train_docids, train_stats = gen_data(vocabxid, dummy_symbols, tm_ignore, cf.train_corpus,
#                                                               cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)

# print("Processing valid corpus to collect sentence and document data...")
# valid_sents, valid_docs, valid_docids, valid_stats = gen_data(vocabxid, dummy_symbols, tm_ignore, cf.valid_corpus,
#                                                               cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)
#
# print("Processing test corpus to collect sentence and document data...")
# test_sents, test_docs, test_docids, test_stats = gen_data(vocabxid, dummy_symbols, tm_ignore, cf.test_corpus,
#                                                               cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)

# TM_vocab = Get_TM_vocab(idxvocab, tm_ignore)


# vocab_save = open(cf.data_path + cf.dataname + '/' + cf.dataname + '_Vocab_data.pckl', 'wb')
# pickle.dump([idxvocab, vocabxid, tm_ignore, TM_vocab], vocab_save)
# vocab_save.close()
#
# train_data_save = open(cf.data_path + cf.dataname + '/' + cf.dataname + '_train_data.pckl','wb')
# pickle.dump([train_sents, train_docs, train_docids, train_stats], train_data_save)
# train_data_save.close()
#
# valid_data_save = open(cf.data_path + cf.dataname + '/' + cf.dataname + '_valid_data.pckl','wb')
# pickle.dump([valid_sents, valid_docs, valid_docids, valid_stats], valid_data_save)
# valid_data_save.close()
#
# test_data_save = open(cf.data_path + cf.dataname + '/' + cf.dataname + '_test_data.pckl','wb')
# pickle.dump([test_sents, test_docs, test_docids, test_stats], test_data_save)
# test_data_save.close()

vocab_save = open(cf.data_path + cf.dataname + '/' + cf.dataname + '_Vocab_data.pckl','rb')
[idxvocab, vocabxid, tm_ignore, TM_vocab] = pickle.load(vocab_save)
vocab_save.close()

train_data_save = open(cf.data_path + cf.dataname + '/' + cf.dataname + '_train_data.pckl','rb')
[train_sents, train_docs, train_docids, train_stats] = pickle.load(train_data_save)
train_data_save.close()

# valid_data_save = open(cf.data_path + cf.dataname + '/' + cf.dataname + '_valid_data.pckl','rb')
# [valid_sents, valid_docs, valid_docids, valid_stats] = pickle.load(valid_data_save)
# valid_data_save.close()

test_data_save = open(cf.data_path + cf.dataname + '/' + cf.dataname + '_test_data.pckl','rb')
[test_sents, test_docs, test_docids, test_stats] = pickle.load(test_data_save)
test_data_save.close()





if cf.dataname == 'apnews' or cf.dataname == 'bnc':
    TM_train_doc, train_doc_bow = Bag_of_words(train_docs[0], idxvocab, tm_ignore)
    # TM_validation_doc, validation_doc_bow = Bag_of_words(valid_docs[0], idxvocab, tm_ignore)
    TM_test_doc, test_doc_bow = Bag_of_words(test_docs[0], idxvocab, tm_ignore)
    # sio.savemat("data/"+ cf.dataname + '/' + cf.dataname + '_TM_data.mat', {'TM_vocab': TM_vocab, 'train_data': train_data, 'validation_data': validation_data,'test_data': test_data})
elif cf.dataname == 'imdb':  # imdb traindata is too big to save
    # TM_train_doc1, train_doc_bow1 = Bag_of_words(train_docs[0][:50000], idxvocab, tm_ignore)
    # TM_train_doc2, train_doc_bow2 = Bag_of_words(train_docs[0][50000:], idxvocab, tm_ignore)
    TM_train_doc, train_doc_bow = Bag_of_words(train_docs[0], idxvocab, tm_ignore)
    # TM_validation_doc, validation_data = Bag_of_words(valid_docs[0], idxvocab, tm_ignore)
    TM_test_doc, test_doc_bow = Bag_of_words(test_docs[0], idxvocab, tm_ignore)
    # sio.savemat("data/"+ cf.dataname + '/' + cf.dataname + '_TM_data.mat', {'TM_vocab': TM_vocab, 'train_data_1': train_data1, 'train_data_2': train_data2, 'validation_data': validation_data,'test_data': test_data})
else:
    print("There is another dataset ! ")
# train_doc_bow = train_doc_bow1

print('-----------------------------data ----------load---------------------------------------- ')

################################################################### data prepare ##############################################################################
train_sent_num = get_sent_num(train_sents[1],len(train_doc_bow[0]))
train_Doc = get_doc_sent(train_sents[1], train_doc_bow, train_sent_num, cf.sent_J)

test_sent_num = get_sent_num(test_sents[1],len(test_doc_bow[0]))
test_Doc = get_doc_sent(test_sents[1], test_doc_bow, test_sent_num, cf.sent_J)

doc_num_batches = int(np.floor(float(len(train_Doc)) / cf.Batch_Size))
batch_ids = [item for item in range(doc_num_batches)]

doc_num_batches_test = int(np.floor(float(len(test_Doc)) / cf.Batch_Size))
batch_ids_test = [item for item in range(doc_num_batches_test)]

V_tm = len(idxvocab) - len(tm_ignore)
Phi, Pi, NDot_Phi, NDot_Pi = initialize_Phi_Pi(V_tm)


cf.Setting['Iterall'] = cf.epoch_size * doc_num_batches



if cf.save_model:
    if not os.path.exists(os.path.join(cf.output_dir, cf.output_prefix)):
        os.makedirs(os.path.join(cf.output_dir, cf.output_prefix))
    if not os.path.exists(os.path.join(cf.output_dir, cf.output_path)):
        os.makedirs(os.path.join(cf.output_dir, cf.output_path))
# saver = tf.train.Saver(max_to_keep=0)
graph = tf.get_default_graph()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) # tf.Graph().as_default()
# config = tf.ConfigProto(allow_soft_placement=True, gpu_options = gpu_options)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with graph.as_default(), tf.Session(config = config) as sess:
    tf.set_random_seed(cf.seed)
    DL = DPGDS_LSTM_Model(cf, V_tm, len(vocabxid))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)






    # initialise word embedding
    if cf.word_embedding_model:
        print("Loading word embedding model...")
        # mword = g.Word2Vec.load(cf.word_embedding_model)
        mword = g.KeyedVectors.load_word2vec_format(cf.word_embedding_model, binary=True)
        cf.word_embedding_size = mword.vector_size
        word_emb = init_embedding(mword, idxvocab)  # mword is a word embedding from gensim.wordembedding
        sess.run(DL.LSTM_word_embedding.assign(word_emb))

    train_tm_like = []
    train_lm_perplexity = []
    test_lm_perplexity = []

    # train model
    for e in range(cf.epoch_size):
        train_theta = []
        print("\nEpoch =", e)
        time_start = time.time()
        # random.shuffle(batch_ids)


        for batch_id in batch_ids:
            MBObserved = int(e * doc_num_batches + batch_id)
            Doc = train_Doc[(batch_id * cf.Batch_Size):((batch_id + 1) * cf.Batch_Size)]
            X_train_batch, y_train_batch,d_train_batch, m_train_batch = get_batch_tm_lm_all(Doc,train_doc_bow,len(idxvocab),tm_ignore,cf.Batch_Size)

            _, tm_like,lm_cost, Theta = sess.run([DL.joint_train_step, DL.L1,DL.lm_Loss,[DL.theta_1C_HT,DL.theta_2C_HT,DL.theta_3C_HT]],
                                    feed_dict={DL.Doc_input:d_train_batch, DL.phi_1:Phi[0], DL.phi_2:Phi[1],DL.phi_3:Phi[2],
                                               DL.pi_1:Pi[0], DL.pi_2:Pi[1] , DL.pi_3:Pi[2],
                                               DL.Sent_input: X_train_batch, DL.Sent_output: y_train_batch, DL.lm_mask: m_train_batch, DL.droprate:cf.lm_keep_prob,DL.batch_num:MBObserved})
            Phi, Pi, NDot_Phi, NDot_Pi = update_Pi_Phi(d_train_batch, Phi, Pi, Theta, doc_num_batches, MBObserved, NDot_Phi, NDot_Pi)

            train_theta.append(Theta)  ####### Theta: 3*   N*K*J

            train_tm_like.append(tm_like)
            train_lm_perplexity.append(np.exp(lm_cost* cf.Batch_Size/np.sum(m_train_batch)))

            # train_lm_perplexity.append(np.exp(lm_cost* cf.Batch_Size/np.sum(m_train_batch)))
            if batch_id%100 == 0:
                print("\nMinibatch =", batch_id)
                print("\ntopic model likelihood:", tm_like)
                print("\nlanguage model Cost:", lm_cost* cf.Batch_Size/np.sum(m_train_batch))
            if cf.save_model and batch_id%2000==0:


                end = time.time()
                print(end - time_start)


                print("-----------------------------------testing-------------------------------------")
                # random.shuffle(batch_ids_test)
                test_lm_costs = 0
                test_lm_like = 0
                test_lm_words = 0
                test_theta = []
                for batch_id_t in batch_ids_test:
                    Doc_test = test_Doc[(batch_id_t * cf.Batch_Size):((batch_id_t + 1) * cf.Batch_Size)]
                    X_test_batch, y_test_batch, d_test_batch, m_test_batch = get_batch_tm_lm_all(Doc_test, test_doc_bow,
                                                                                                     len(idxvocab),
                                                                                                     tm_ignore,
                                                                                                     cf.Batch_Size)
                    curr_cost_test, tm_like_test, lm_cost_test, Theta_test = sess.run(
                        [DL.joint_Loss, DL.L1, DL.lm_Loss,
                         [DL.theta_1C_HT, DL.theta_2C_HT,DL.theta_3C_HT]],
                        feed_dict={DL.Doc_input: d_test_batch, DL.phi_1: Phi[0], DL.phi_2: Phi[1], DL.phi_3: Phi[2],
                                   DL.pi_1: Pi[0], DL.pi_2: Pi[1], DL.pi_3: Pi[2],
                                   DL.Sent_input: X_test_batch, DL.Sent_output: y_test_batch,
                                   DL.lm_mask: m_test_batch, DL.droprate:1})

                    test_lm_costs += lm_cost_test * cf.Batch_Size
                    test_lm_words += np.sum(m_test_batch)
                    test_theta.append(Theta_test)  ####### Theta: 2*   N*K*J




                test_theta_save = open(os.path.join(cf.output_dir, cf.output_path,
                                    cf.dataname + str(e) + '_test_theta_' + str(batch_id) + '.pckl'),'wb')
                pickle.dump(test_theta, test_theta_save)
                test_theta_save.close()

                print("test topic likelihood = %.3f" % (tm_like_test / doc_num_batches_test))

                print("test language model perplexity = %.3f" % (np.exp(test_lm_costs / test_lm_words)))
                test_lm_perplexity.append(np.exp(test_lm_costs / test_lm_words))

                if np.exp(test_lm_costs / test_lm_words)<=min(test_lm_perplexity):
                    print('----------------------------------- saving model -------------------------------------')

                    saver.save(sess, os.path.join(cf.output_dir, cf.output_prefix,
                                                  cf.dataname + str(e) + "_model_" + str(batch_id) + ".ckpt"))
                    Phi_Pi_save = open(os.path.join(cf.output_dir, cf.output_path,
                                                    cf.dataname + str(e) + '_Phi_Pi_hidden_Doc_' + str(batch_id) + '.pckl'), 'wb')
                    pickle.dump([Phi, Pi], Phi_Pi_save)
                    Phi_Pi_save.close()


        state_save = open(os.path.join(cf.output_dir, cf.output_path, cf.dataname + '-result.pckl'), 'wb')
        pickle.dump([train_tm_like,train_lm_perplexity,test_lm_perplexity], state_save)
        state_save.close()
        print(test_lm_perplexity)
