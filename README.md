# rGBN_RNN

Code for a three-layer rGBN-RNN model proposed in 《RECURRENT HIERARCHICAL TOPIC-GUIDED NEURAL LANGUAGE MODELS》, submitted to ICLR 2020.
This is a basic version of the proposed model rGBN-RNN.

Dependencies:
This code is written in python. To use it you will need:
Python 3.5+
Tensorflow-gpu 1.9.0
A recent version of NumPy and SciPy
pickle
gensim

Getting started:
1. Download datasets from the links presented in supplementary of our submitted paper, and put into the data directory. 
2. Download the pre-trained 300-dimension word2vec Google News vectors and the stoplists, and put into the data directory.
3. Train the GBN-RNN model using 'gbn_rnn_layer3.py', the data-preprocessed code is included.

Note:
If you run this code in Windows system , you may need to install visual studio firstly , and change '***.so'  to  '***.dll'  files in  'gbn_sampler.py'.
