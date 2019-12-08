# coding=utf-8

import os
from prepro import load_vocab
from utils import load_json, load_embeddings

class Config(object):
    def __init__(self, task, vec_method, model_name):
        self.ckpt_path = './ckpt_{}/{}_{}/'.format(model_name,task, vec_method)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        source_dir = os.path.join('.', 'data', vec_method, task)
        # 词集合
        self.word_vocab, _ = load_vocab(os.path.join(source_dir, 'words.vocab'))
        # 词集合的长度
        self.vocab_size = len(self.word_vocab)
        if vec_method == "glove":
            self.word_emb = load_embeddings(os.path.join(source_dir, 'glove.filtered.npz'))
        else:
            self.word_emb = load_embeddings(os.path.join(source_dir, 'word2vec.filtered.npz'))

    # log and model file paths
    max_to_keep = 5  # max model to keep while training
    no_imprv_patience = 100   # 最大的准确率下降的轮数

    # word embeddings
    use_word_emb = True
    finetune_emb = False
    word_dim = 300

    # TextCNN conv
    conv_filter_sizes = [3, 4, 5, 6]
    num_filters = 128
    conv_filter = 3
    k = 50

    # hyperparameters
    l2_reg = 0.001
    grad_clip = 5.0
    decay_lr = True  # 是否根据学习的轮数降低学习率
    lr = 0.01
    lr_decay = 0.05
    dropout_keep_prob = 0.5

    # lstm
    hidden_layers_1 = 1
    hidden_layers_2 = 1
    steps = 10
    hidden_units = 64
    # # highway network
    # use_highway = False
    # highway_num_layers = 2
    #
    # # model parameters
    # num_layers = 15
    # num_units = 13
    # num_units_last = 100

    # conv parameters
