# coding=utf-8

import os
import re
import sys
import chardet
import numpy as np
from tqdm import tqdm
import json
import nltk
import operator
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

# 读取数据，切词处理
# 训练glove 和 word2vec模型，保存模型
# 对切词结果进行embed

PAD = '__PAD__'
UNK = '__UNK__'


def clean_text(text):
    """Remove special tokens and clean up text"""
    text = text.replace("\n", " ")
    text = text.replace("``", '"').replace("''", '"').replace("`", "'")  # convert quote symbols
    text = text.replace("n 't", "n't").replace("can not", "cannot")
    text = re.sub(' +', ' ', text)
    text = re.sub(r"[^A-Za-z0-9(),!?\'\\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ", text)
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()

def split_word(path):
    with open(path, "rb") as f:
        text = f.read()
        content_type = chardet.detect(text)["encoding"]
        if content_type != None:
            text = text.decode(content_type)
        else:
            text = text.decode("utf-8")
        text = clean_text(text)
    return text

# def split_word(path):
#     with open(path, "r", encoding="utf-8") as f:
#         text = f.read()
#         text = clean_text(text)
#     return text

def load_20news():
    path = "./raw/20news_18828/20news-18828/"
    path2 = "./raw/20news_18828/split_res/"
    dirs_names = os.listdir(path)
    total_news = "./raw/20news_18828/20news_total_splited.txt"
    # 保存类别的id和每个类别的文件数目
    labels = dict()
    with open(total_news, "w", encoding="utf-8") as t:
        for i, name in enumerate(dirs_names):
            dir_path = path + name
            dir_nums = os.listdir(dir_path)
            labels[name] = (i, len(dir_nums))
            split_res_path = path2 + "{}.txt".format(name)
            if os.path.exists(split_res_path):
                continue
            with open(split_res_path, "w", encoding="utf-8") as w:
                for d in dir_nums:
                    pt = dir_path + "/" + d
                    text = split_word(pt)
                    w.write(text)
                    w.write("\n")
                    t.write(text)
                    t.write("\n")
            print(name + " write end, total {} texts".format(len(dir_nums)))
    if  not os.path.exists(path2 + "labels.json"):
        with open(path2 + "labels.json", "w") as f:
            json.dump(labels, f)

def load_imdb():
    train_path = "./raw/imdb50000/train/"
    path =  "./raw/imdb50000/split_res/"
    path2 = "./raw/imdb50000/"
    total_comments_path = "./raw/imdb50000/total_comments.txt"
    neg_pos_dirs = os.listdir(train_path)
    labels = dict()
    with open(total_comments_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(neg_pos_dirs):
            dir_path = train_path + name
            dir_nums = os.listdir(dir_path)
            labels[name] = (i, len(dir_nums))
            split_res_path = path + "{}.txt".format(name)
            if os.path.exists(split_res_path):
                continue
            with open(split_res_path, "w", encoding="utf-8") as w:
                for d in dir_nums:
                    pt = dir_path + "/" + d
                    text = split_word(pt)
                    w.write(text)
                    w.write("\n")
                    f.write(text)
                    f.write("\n")
            print(name + " write end, total {} texts".format(len(dir_nums)))
    if not os.path.exists(path2 + "labels.json"):
        with open(path2 + "labels.json", "w") as f:
            json.dump(labels, f)

def load_cornell():
    tokens_path = "./raw/cornell/tokens/"
    path = "./raw/cornell/split_res/"
    path2 = "./raw/cornell/"
    total_comments_path = "./raw/cornell/total_comments.txt"
    neg_pos_dirs = os.listdir(tokens_path)
    labels = dict()
    with open(total_comments_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(neg_pos_dirs):
            dir_path = tokens_path + name
            dir_nums = os.listdir(dir_path)
            labels[name] = (i, len(dir_nums))
            split_res_path = path + "{}.txt".format(name)
            if os.path.exists(split_res_path):
                continue
            with open(split_res_path, "w", encoding="utf-8") as w:
                for d in dir_nums:
                    pt = dir_path + "/" + d
                    text = split_word(pt)
                    w.write(text)
                    w.write("\n")
                    f.write(text)
                    f.write("\n")
            print(name + " write end, total {} texts".format(len(dir_nums)))
    if not os.path.exists(path2 + "labels.json"):
        with open(path2 + "labels.json", "w") as f:
            json.dump(labels, f)

def load_ling_spam():
    tokens_path = "./raw/ling-spam/tokens/"
    path = "./raw/ling-spam/split_res/"
    path2 = "./raw/ling-spam"
    total_comments_path = "./raw/ling-spam/total_comments.txt"
    neg_pos_dirs = os.listdir(tokens_path)
    labels = dict()
    with open(total_comments_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(neg_pos_dirs):
            dir_path = tokens_path + name
            dir_nums = os.listdir(dir_path)
            labels[name] = (i, len(dir_nums))
            split_res_path = path + "{}.txt".format(name)
            if os.path.exists(split_res_path):
                continue
            with open(split_res_path, "w", encoding="utf-8") as w:
                for d in dir_nums:
                    pt = dir_path + "/" + d
                    text = split_word(pt)
                    w.write(text)
                    w.write("\n")
                    f.write(text)
                    f.write("\n")
            print(name + " write end, total {} texts".format(len(dir_nums)))
    if not os.path.exists(path2 + "labels.json"):
        with open(path2 + "labels.json", "w") as f:
            json.dump(labels, f)

def build_vocab(data_sets, threshold=0):  # 阈值表示设置次数少于threshold的词不要
    word_count = {}
    for dataset in data_sets:
        for word in dataset.split(" "):
            word_count[word] = word_count.get(word, 0) + 1
    word_count = reversed(sorted(word_count.items(), key=operator.itemgetter(1)))
    word_vocab = set([w[0] for w in word_count if w[1] >= threshold])
    return word_vocab

def load_glove_vocab(glove_vocab_path):
    with open(glove_vocab_path, 'r', encoding='utf-8') as f:
        vocab = {line.strip().split()[0] for line in tqdm(f, desc='Loading GloVe vocabularies')}
    print('\t -- totally {} tokens in GloVe embeddings.\n'.format(len(vocab)))
    return vocab

def write_vocab(vocab, filename):
    """write vocabulary to file"""
    if os.path.exists(filename):
        pass
    else:
        sys.stdout.write('Writing vocab to {}...'.format(filename))
        with open(filename, 'w') as f:
            for i, word in enumerate(vocab):
                f.write('{}\n'.format(word)) if i < len(vocab) - 1 else f.write(word)

def load_vocab(filename):
    """read vocabulary from file into dict"""
    word_idx = dict()
    idx_word = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            word_idx[word] = idx
            idx_word[idx] = word
    return word_idx, idx_word

def save_filtered_vectors(word_idx, gw_path, save_path, word_dim, vec_method):
    """Prepare pre-trained word embeddings for dataset"""
    embeddings = np.zeros([len(word_idx), word_dim])  # embeddings[0] for PAD
    scale = np.sqrt(3.0 / word_dim)
    embeddings[1] = np.random.uniform(-scale, scale, [1, word_dim])  # for UNK
    if vec_method == "glove":
        with open(gw_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='Filtering GloVe embeddings'):
                line = line.strip().split(' ')
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                if word in word_idx:
                    idx = word_idx[word]
                    embeddings[idx] = np.asarray(embedding)
    elif vec_method == "word2vec":
        word2vec = KeyedVectors.load_word2vec_format(gw_path, binary=True)
        print("Word2vec embeddings read end")
        for word in word_idx.keys():
            if word in word2vec.vocab:
                idx = word_idx[word]
                embeddings[idx] = np.asarray(word2vec[word])
    sys.stdout.write('Saving filtered embeddings...')
    np.savez_compressed(save_path, embeddings=embeddings)
    sys.stdout.write(' done.\n')

def fit_word_to_id(word, word_ids):
    """Convert word str to word index and char indices"""
    word_id = word_ids[word] if word in word_ids else word_ids[UNK]
    return word_id

def build_dataset(docs_path, word_ids, filename):
    data_sets = []
    docs_names = os.listdir(docs_path)
    num_labels = len(docs_names)
    for i, name in enumerate(docs_names):
        with open(docs_path + "/" +name, "r", encoding="utf-8") as f:
            documents = f.readlines()
        for sentence in documents:
            words = []
            for word in sentence.split(" "):
                words += [fit_word_to_id(word, word_ids)]
            label = [1 if j == i else 0 for j in range(num_labels)]
            data_sets.append({"sentence" : words, "label" : label})
    with open(filename, 'w') as fn:
        json.dump(data_sets, fn)
    sys.stdout.write('dump dataset to {}.\n'.format(filename))

def load_word2vec_vocab(filename):
    """Read word2vec word vocabulary from embeddings"""
    google_bin = KeyedVectors.load_word2vec_format(filename, binary=True)
    # google all word vectors
    vocab = google_bin.wv.vocab
    print('\t -- totally {} tokens in word2vec embeddings.\n'.format(len(vocab)))
    return set(vocab)

def glove_generate(dir_path, total_name, glove_vocab, glove_path, docs_path):
    """ dir_path: 是所有文档的合并的路径
        total_name: 所有文档集合的文件名
        glove_vocab: 训练出的glove词汇结果vocab.txt
        glove_path: glove_vocab存在的路径
        docs_path: 所有文档的路径，同一类的是一个txt文件
    Writebale：glove_filter.npz是所有词的词向量结果;
            datasets.json是{'sentence': words, 'label': label}，
            其中words是每个文本的词的id，label是one-hot编码结果
    """
    with open(dir_path + total_name, "r", encoding="utf-8") as f:
        data_sets = f.readlines()  # 所有的数据，没有切分的词
    word_vocab = build_vocab(data_sets)  # 返回的是该数据集的所有词汇
    if glove_vocab is None:
        glove_vocab = load_glove_vocab(glove_path)  # 导入glove_vocab词汇集
    word_vocab = [PAD, UNK] + list(word_vocab & glove_vocab)
    write_vocab(word_vocab, filename=os.path.join(dir_path, "words.vocab"))
    word_ids, idx_word = load_vocab(os.path.join(dir_path, "words.vocab"))
    # 用于保存词向量，按词次序保存的值
    save_filtered_vectors(word_ids, glove_path, os.path.join(dir_path, 'preproceed', "glove", 'glove.filtered.npz'),
                          word_dim=300, vec_method="glove")
    build_dataset(docs_path, word_ids, os.path.join(dir_path, 'preproceed', "glove", 'data_sets.json'))

def word2vec_generate(dir_path, total_name, wv_vocab, wv_path, docs_path):
    """ dir_path: 是所有文档的合并的路径
        total_name: 所有文档集合的文件名
        wv_vocab: 训练出的glove词汇结果vocab.txt
        wv_path: glove_vocab存在的路径
        docs_path: 所有文档的路径，同一类的是一个txt文件
    Writebale：glove_filter.npz是所有词的词向量结果;
                datasets.json是{'sentence': words, 'label': label}，
                其中words是每个文本的词的id，label是one-hot编码结果
    """
    with open(dir_path + total_name, "r", encoding="utf-8") as f:
        data_sets = f.readlines()  # 所有的数据，没有切分的词
    word_vocab = build_vocab(data_sets)  # 返回的是该数据集的所有词汇
    if wv_vocab is None:
        wv_vocab = load_word2vec_vocab(wv_path)  # 导入glove_vocab词汇集
    word_vocab = [PAD, UNK] + list(word_vocab & wv_vocab)
    write_vocab(word_vocab, filename=os.path.join(dir_path, "words.vocab"))
    word_ids, idx_word = load_vocab(os.path.join(dir_path, "words.vocab"))
    # 用于保存词向量，按词次序保存的值
    save_filtered_vectors(word_ids, wv_path, os.path.join(dir_path, 'preproceed', "word2vec", 'word2vec.filtered.npz'),
                          word_dim=300, vec_method="word2vec")
    build_dataset(docs_path, word_ids, os.path.join(dir_path, 'preproceed', "word2vec", 'data_sets.json'))


if __name__ == "__main__":
    # load_20news()
    # path = "./raw/20news_18828/"
    # glove_generate(dir_path=path, total_name="20news_total_splited.txt", glove_vocab=None, glove_path=os.path.join(path, "glove", "vectors.txt"),
    #                docs_path=os.path.join(path, "split_res"))
    # word2vec_vocab = load_word2vec_vocab('./emb_dir/GoogleNews-vectors-negative300.bin')
    # word2vec_generate(dir_path=path, total_name="20news_total_splited.txt", wv_vocab=word2vec_vocab,
    #                 wv_path="./emb_dir/GoogleNews-vectors-negative300.bin", docs_path=os.path.join(path, "split_res"))

    # 处理IMDB影视集合
    # load_imdb()
    # path = "./raw/imdb50000/"
    # glove_generate(dir_path=path, total_name="total_comments.txt", glove_vocab=None, glove_path=os.path.join(path, "glove", "vectors.txt"),
    #                docs_path=os.path.join(path, "split_res"))
    # word2vec_vocab = load_word2vec_vocab('./emb_dir/GoogleNews-vectors-negative300.bin')
    # word2vec_generate(dir_path=path, total_name="total_comments.txt", wv_vocab=word2vec_vocab,
    #                 wv_path="./emb_dir/GoogleNews-vectors-negative300.bin", docs_path=os.path.join(path, "split_res"))


    # 处理cornell数据集
    # load_cornell()
    # path = "./raw/cornell/"
    # glove_generate(dir_path=path, total_name="total_comments.txt", glove_vocab=None, glove_path=os.path.join(path, "glove", "vectors.txt"),
    #                docs_path=os.path.join(path, "split_res"))
    # word2vec_vocab = load_word2vec_vocab('./emb_dir/GoogleNews-vectors-negative300.bin')
    # word2vec_generate(dir_path=path, total_name="total_comments.txt", wv_vocab=word2vec_vocab,
    #                 wv_path="./emb_dir/GoogleNews-vectors-negative300.bin", docs_path=os.path.join(path, "split_res"))


    # 处理ling-spam数据集
    # load_ling_spam()
    path = "./raw/ling-spam/"
    glove_generate(dir_path=path, total_name="total_comments.txt", glove_vocab=None,
                   glove_path=os.path.join(path, "glove", "vectors.txt"),
                   docs_path=os.path.join(path, "split_res"))
    word2vec_vocab = load_word2vec_vocab('./emb_dir/GoogleNews-vectors-negative300.bin')
    word2vec_generate(dir_path=path, total_name="total_comments.txt", wv_vocab=word2vec_vocab,
                      wv_path="./emb_dir/GoogleNews-vectors-negative300.bin", docs_path=os.path.join(path, "split_res"))
