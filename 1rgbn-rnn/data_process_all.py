import codecs
import sys
import operator
import math
import re
import numpy as np
from collections import defaultdict
import config as cf
import random

def update_vocab(symbol, idxvocab, vocabxid):
    idxvocab.append(symbol)
    vocabxid[symbol] = len(idxvocab) - 1

def gen_vocab(dummy_symbols, corpus, stopwords, vocab_minfreq, vocab_maxfreq, verbose):
    # corpus = corpus.lower()
    idxvocab = []
    vocabxid = defaultdict(int)
    vocab_freq = defaultdict(int)
    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        for word in line.strip().split():
            vocab_freq[word] += 1
        if line_id % 1000 == 0 and verbose:
            sys.stdout.write(str(line_id) + " processed\r")
            sys.stdout.flush()

    #add in dummy symbols into vocab
    for s in dummy_symbols:
        update_vocab(s, idxvocab, vocabxid)

    #remove low fequency words

    for w, f in sorted(vocab_freq.items(), key=operator.itemgetter(1), reverse=True):
        # if f > vocab_minfreq:
        #     update_vocab(w, idxvocab, vocabxid)
        if f < vocab_minfreq:
            break
        else:
            update_vocab(w, idxvocab, vocabxid)

    # # remove words appears less than 100 docs
    # lessdocwords = []
    # vocab_doc_freq = np.zeros([len(idxvocab), 2])
    # for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
    #     # docw = [item for sublst in line for item in sublst]
    #     vocab_doc_freq[:, 1] = 0
    #     for w in line.strip().split():
    #     # for w in docw:
    #         if vocab_doc_freq[vocabxid[w]][1] == 0:
    #             vocab_doc_freq[vocabxid[w]][0] += 1
    #             vocab_doc_freq[vocabxid[w]][1] += 1
    #         else:
    #             vocab_doc_freq[vocabxid[w]][1] += 1
    #
    # for d in range(len(idxvocab)):
    #     if vocab_doc_freq[d][0] < 100:
    #         lessdocwords.append(idxvocab[d])

    #ignore stopwords, frequent words and symbols for the document input for topic model
    stopwords = set([item.strip().lower() for item in open(stopwords)])
    freqwords = set([item[0] for item in sorted(vocab_freq.items(), key=operator.itemgetter(1), \
        reverse=True)[:int(float(len(vocab_freq))*vocab_maxfreq)]]) #ignore top N% most frequent words for topic model
    alpha_check = re.compile("[a-zA-Z]")
    symbols = set([ w for w in vocabxid.keys() if ((alpha_check.search(w) == None) or w.startswith("'")) ])
    ignore = stopwords | freqwords | symbols | set(dummy_symbols) | set(["n't"])    #   | set(lessdocwords)
    # ignore = stopwords | freqwords | set(lessdocwords)
    ignore = set([vocabxid[w] for w in ignore if w in vocabxid])

    return idxvocab, vocabxid, ignore

def gen_data(vocabxid, dummy_symbols, ignore, corpus, tm_sent_len, lm_sent_len, verbose, remove_short):
    sents = ([], []) #tuple of (tm_sents, lm_sents); each element is [(doc_id, seq_id, seq)]
    docs = ([], []) #tuple of (tm_docs, lm_docs); each element is [ [[doc1seq1], [doc2seq2]], [doc2_seqs] ]
    sent_lens = [] #original sentence lengths
    doc_lens = [] #original document lengths
    docids = [] #original document IDs
    start_symbol = dummy_symbols[1]
    end_symbol = dummy_symbols[2]
    unk_symbol = dummy_symbols[3]

    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        tm_sents = [vocabxid[start_symbol]] #sentences for tm
        lm_sents = [] #sentences for lm
        for s in line.strip().split("\t"):
            sent = [vocabxid[start_symbol]]
            for w in s.strip().split():
                if w in vocabxid:
                    sent.append(vocabxid[w])
                    if (vocabxid[w] not in ignore):
                        tm_sents.append(vocabxid[w])
                else:
                    sent.append(vocabxid[unk_symbol])
            sent.append(vocabxid[end_symbol])
            lm_sents.append(sent)

        #ignore documents with no words
        if not remove_short or (len(tm_sents) > 1):
            docids.append(line_id)
            sent_lens.extend([len(item)-1 for item in lm_sents])
            doc_lens.append(len(tm_sents))

            #chop tm_sents into sequences of length tm_sent_len
            seq_id = 0
            doc_seqs = []
            # for si in range(int(math.ceil(len(tm_sents) * 1.0 / tm_sent_len))):
            for si in range(int(math.ceil(len(tm_sents) * 1.0 / tm_sent_len))):
                seq = tm_sents[si*tm_sent_len:((si+1)*tm_sent_len+1)]
                if len(seq) > 1:
                    sents[0].append((len(docs[0]), seq_id, seq))
                    doc_seqs.append(seq[1:])
                    seq_id += 1
            docs[0].append(doc_seqs)

            #chop each sentence exceeding lm_sent_len into multiple sequences for the language model
            seq_id = 0
            doc_seqs = []
            for s in lm_sents:
                for si in range(int(math.ceil(len(s) * 1.0 / lm_sent_len))):    # int(math.ceil(len(s) * 1.0 / lm_sent_len))
                    seq = s[si*lm_sent_len:((si+1)*lm_sent_len+1)]
                    if len(seq) > 1:
                        sents[1].append((len(docs[1]), seq_id, seq))
                        doc_seqs.append([w for w in seq[1:] if w not in ignore]) #output sentence = seq[1:]
                        seq_id += 1
            docs[1].append(doc_seqs)

        if line_id % 1000 == 0 and verbose:
            sys.stdout.write(str(line_id) + " processed\r")
            sys.stdout.flush()

    return sents, docs, docids, (np.mean(sent_lens), max(sent_lens), min(sent_lens), np.mean(doc_lens), max(doc_lens), min(doc_lens))

def Get_TM_vocab(idxvocab,tm_ignore):
    TM_vocab = np.delete(idxvocab, list(tm_ignore))
    return TM_vocab

def Bag_of_words(TM_original_train,idxvocab,tm_ignore):  # get the bag of words form of data
    TM_doc = []
    for d in range(len(TM_original_train)):
        docw = TM_original_train[d]
        docw = pad([item for sublst in docw for item in sublst][:cf.doc_len], cf.doc_len, 0)
        TM_doc.append(docw)
    TM_train_bow = np.zeros([len(idxvocab), len(TM_doc)])
    for doc_index in range(len(TM_doc)):
        for word in TM_doc[doc_index]:
            TM_train_bow[word][doc_index] += 1
    TM_train_bow = np.delete(TM_train_bow, list(tm_ignore), axis = 0)
    return TM_doc, TM_train_bow

# def Bow_sents(sents,idxvocab,tm_ignore):  # get the bag of words form of data
#     sent_bow = []
#     num = 10000
#     for ep in range(int(np.ceil(len(sents)/num))):
#         if ep != np.ceil(len(sents)/num)-1:
#             sent_bow1 = np.zeros([len(idxvocab),num])
#             for sent_index in range(num):
#                 for word in sents[ep*num + sent_index]:
#                     sent_bow1[word][ep*num + sent_index] += 1
#                 sent_bow1 = np.delete(sent_bow1, list(tm_ignore), axis = 0)
#             sent_bow.append(sent_bow1)
#         else:
#             sent_bow2 = np.zeros([len(idxvocab), len(sents)-num*ep])
#             for sent_index in range(len(sents)-num*ep):
#                 for word in sents[ep * num + sent_index]:
#                     sent_bow2[word][ep * num + sent_index] += 1
#                 sent_bow2 = np.delete(sent_bow2, list(tm_ignore), axis=0)
#             sent_bow.append(sent_bow2)
#         return sent_bow



def init_embedding(model, idxvocab):
    word_emb = []
    for vi, v in enumerate(idxvocab):
        if v in model:
            word_emb.append(model[v])
        else:
            word_emb.append(np.random.uniform(-0.5/model.vector_size, 0.5/model.vector_size, [model.vector_size,]))
    return np.array(word_emb)

def pad(lst, max_len, pad_symbol):
    return lst + [pad_symbol] * (max_len - len(lst))

def get_batches(sents,Theta,pad_id = 0):
    lm_num_batches =  int(math.ceil(float(len(sents)) / cf.batch_size))
    batch_ids = [item for item in range(lm_num_batches)]
    sent_len = cf.lm_sent_len
    batch_size = cf.batch_size
    random.shuffle(batch_ids)
    random.shuffle(sents)
    X = []
    Y = []
    M = []
    T = []
    for idx in batch_ids:
        x, y, m, t = [], [], [], []
        if idx != max(batch_ids) :
            for docid, seqid, seq in sents[(idx * batch_size):((idx + 1) * batch_size)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
                t.append(Theta[:,docid])
        else:
            for docid, seqid, seq in sents[(idx * batch_size):len(sents)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
                t.append(Theta[:, docid])
            for padingnum in range((max(batch_ids)+1)*batch_size - len(sents)):
                x.append([pad_id] * sent_len)
                y.append([pad_id] * sent_len)
                m.append([0.0] * sent_len)
                t.append([pad_id] * cf.topic_embedding_size)
        X.append(x)
        Y.append(y)
        M.append(m)
        T.append(t)
    return X,Y,M,T



def get_batches_tm_lm(sents,docbow,V, pad_id = 0 ):
    lm_num_batches = int(math.ceil(float(len(sents)) / cf.batch_size))
    batch_ids = [item for item in range(lm_num_batches)]
    sent_len = cf.lm_sent_len
    batch_size = cf.batch_size
    random.shuffle(batch_ids)
    random.shuffle(sents)
    X = []
    Y = []
    D = []
    M = []
    for idx in batch_ids:
        x, y, d, m = [], [], [], []
        if idx != max(batch_ids):
            for docid, seqid , seq in sents[(idx * batch_size):((idx + 1) * batch_size)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                d.append(docbow[: ,docid])
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
        else:
            for docid, seqid, seq in sents[(idx * batch_size):len(sents)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                d.append(docbow[:, docid])
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
            for padingnum in range((max(batch_ids) + 1) * batch_size - len(sents)):
                x.append([pad_id] * sent_len)
                y.append([pad_id] * sent_len)
                d.append(np.array([pad_id] * V))
                m.append([0.0] * sent_len)
        X.append(x)
        Y.append(y)
        D.append(d)
        M.append(m)
    return X, Y, D, M

def get_batches_lm(sents,pad_id = 0):
    lm_num_batches =  int(math.ceil(float(len(sents)) / cf.batch_size))
    batch_ids = [item for item in range(lm_num_batches)]
    sent_len = cf.lm_sent_len
    batch_size = cf.batch_size
    random.shuffle(batch_ids)
    random.shuffle(sents)
    X = []
    Y = []
    M = []
    for idx in batch_ids:
        x, y, m = [], [], []
        if idx != max(batch_ids) :
            for docid, seqid, seq in sents[(idx * batch_size):((idx + 1) * batch_size)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
        else:
            for docid, seqid, seq in sents[(idx * batch_size):len(sents)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
            for padingnum in range((max(batch_ids)+1)*batch_size - len(sents)):
                x.append([pad_id] * sent_len)
                y.append([pad_id] * sent_len)
                m.append([0.0] * sent_len)
        X.append(x)
        Y.append(y)
        M.append(m)
    return X,Y,M

def get_batches_tm(docbow,pad_id=0.0):
    tm_num_batches = int(math.ceil(float(docbow.shape[1]) / cf.batch_size))
    batch_ids = [item for item in range(tm_num_batches)]
    random.shuffle(batch_ids)
    D = []
    for idx in batch_ids:
        d = []
        if idx != max(batch_ids):
            for docid in range((idx * cf.batch_size),((idx+1) * cf.batch_size)):
                d.append(docbow[: ,docid])
        else:
            for docid in range((idx * cf.batch_size),docbow.shape[1]):
                d.append(docbow[:, docid])
            for padingnum in range((max(batch_ids) + 1) * cf.batch_size - docbow.shape[1]):
                d.append(np.array([pad_id] * docbow.shape[0]))
        D.append(d)
    return D


def get_sent_num(sents,doc_lenth):
    sent_num = np.zeros([doc_lenth,])
    for docid, seqid, seq in sents:
        sent_num[docid] += 1
    return np.int32(sent_num)



def get_doc_sent(sents, docbow, sent_num, sent_J, pad_id=0):
    index = 0
    D = []
    for id_doc in range(len(sent_num)):
        doc = sents[index : index + sent_num[id_doc]]
        J = int(np.ceil(float(sent_num[id_doc]) / sent_J ))
        for s_id in range(J * sent_J - sent_num[id_doc]):
            doc.append((id_doc, s_id + sent_num[id_doc], [pad_id] * cf.lm_sent_len))  ######## pading 0 of sentence to the doc which is less than 8 sentences
        for id_sent in range(J):
            sent = doc[id_sent * sent_J : (id_sent+1)*sent_J]
            D.append(sent)
        index += sent_num[id_doc]
    return D

def Bow_sents(s,V,tm_ignore):
    s_b = np.zeros(V)
    for w in s:
        s_b[w] += 1
    s_b = np.delete(s_b, list(tm_ignore), axis=0)
    return s_b

def get_batch_tm_lm_all(Doc, docbow, V, tm_ignore, Batch_Size,  pad_id = 0 ):
    x, y, d, m,n = [], [], [], [],[]
    for idx_sent in range(Batch_Size):
        xx, yy, dd, mm = [], [], [], []
        for docid, seqid, seq in Doc[idx_sent]:
            xx.append(pad(seq[:-1], cf.lm_sent_len, pad_id))
            yy.append(pad(seq[1:], cf.lm_sent_len, pad_id))
            dd.append(docbow[: ,docid] - Bow_sents(seq,V,tm_ignore))

            if seq[0] == pad_id:
                mm.append([0.0] * cf.lm_sent_len)
            else:
                mm.append([1.0] * (len(seq) - 1) + [0.0] * (cf.lm_sent_len - len(seq) + 1))

        x.append(xx)
        y.append(yy)
        d.append(dd)
        m.append(mm)
    return  np.array(x), np.array(y), np.array(d), np.array(m)


def get_batch_tm_lm_all_pre_sentence(Doc, docbow, V, tm_ignore, Batch_Size, pad_id=0):
    x, y, d, m = [], [], [], []
    for idx_sent in range(Batch_Size):
        xx, yy, dd, mm = [], [], [], []
        s_pre_bow = np.zeros([V-len(tm_ignore)])
        for docid, seqid, seq in Doc[idx_sent]:
            xx.append(pad(seq[:-1], cf.lm_sent_len, pad_id))
            yy.append(pad(seq[1:], cf.lm_sent_len, pad_id))
            sent_bow = Bow_sents(seq, V, tm_ignore)
            dd.append(s_pre_bow.copy())
            s_pre_bow += sent_bow
            # s_pre_bow = sent_bow
            if seq[0] == pad_id:
                mm.append([0.0] * cf.lm_sent_len)
            else:
                mm.append([1.0] * (len(seq) - 1) + [0.0] * (cf.lm_sent_len - len(seq) + 1))
        x.append(xx)
        y.append(yy)
        d.append(dd)
        m.append(mm)
    return np.array(x), np.array(y), np.array(d), np.array(m)

def get_batch_tm_lm_pre_fix_next(Doc, V, tm_ignore, Batch_Size, pad_id = 0 ):
    x, y, d, m, n = [], [], [], [], []
    Doc_len = [len(Doc[id].copy()) for id in range(Batch_Size)]
    for idx_sent in range(Batch_Size):
        xx, yy, dd, mm, nn = [], [], [], [], []
        s_pre_bow = np.zeros([V-len(tm_ignore)])
        for docid, seqid, seq in Doc[idx_sent]:
            xx.append(pad(seq[:-1], cf.lm_sent_len, pad_id))
            yy.append(pad(seq[1:], cf.lm_sent_len, pad_id))
            sent_bow = Bow_sents(seq, V, tm_ignore)
            nn.append(sent_bow)
            dd.append(s_pre_bow.copy())
            s_pre_bow += sent_bow
            if seqid < Doc_len[idx_sent]:
                mm.append([1.0] * (len(seq) - 1) + [0.0] * (cf.lm_sent_len - len(seq) + 1))
            else:
                mm.append([0.0] * cf.lm_sent_len)

        x.append(xx)
        y.append(yy)
        d.append(dd)
        m.append(mm)
        n.append(nn)
    return  np.array(x), np.array(y), np.array(d), np.array(m), np.array(n), max_sent_len
