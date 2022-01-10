import numpy as np
import tensorflow as tf
import _pickle
import random
import subprocess
from tokenizer import *
from subprocess import Popen, PIPE

def qid_space(path_tr, path_tr_hr, weight_hr, qid2title_add_fi):

    qid2title_add = {}
    # LR language
    with open(path_tr, "r") as f:
        for line in f:
            args = line[:-1].split(" ")
            mnt = args[2]
            qid = args[1]

            if qid not in qid2title_add and weight_hr != -1:
                qid2title_add[qid] = None

    # HR language
    with open(path_tr_hr, "r") as f:
        for line in f:
            args = line[:-1].split(" ")
            mnt = args[2]
            qid = args[1]

            if qid not in qid2title_add:
                qid2title_add[qid] = None

    cnt = 0
    with open("../interlanguage_links_wikidata-20190901.csv") as f:
        for line in f:
            args = line.split(",")
            qid = args[0]
            lang = args[1]
            if qid in qid2title_add and lang=="en":
                name = ",".join(args[2:])
                qid2title_add[qid] = name[:-1]
                cnt += 1
                if cnt == len(qid2title_add):
                    break

    for qid in list(qid2title_add.keys()):
        if qid2title_add[qid] == None:
            del qid2title_add[qid]

    f = open(qid2title_add_fi, "wb")
    _pickle.dump(qid2title_add, f)
    f.close()
    return qid2title_add

def create_data(path_tr, path_tr_hr, data_tr_sampled, num_data, qid2title, weight_hr):

    mention_ent_lr = {}
    # LR language
    with open(path_tr, "r") as f:
        for line in f:
            args = line[:-1].split(" ")
            mnt = args[2]
            qid = args[1]
            if mnt not in mention_ent_lr and qid in qid2title:
                mention_ent_lr[mnt] = []
            if qid in qid2title:
                title = qid2title[qid]
                title = tokenize(title)
                mention_ent_lr[mnt].append((title, qid, 1))

    # HR language
    mention_ent_hr = {}
    with open(path_tr_hr, "r") as f:
        for line in f:
            args = line[:-1].split(" ")
            mnt = args[2]
            qid = args[1]
            if mnt not in mention_ent_hr and qid in qid2title:
                mention_ent_hr[mnt] = []
            if qid in qid2title:
                title = qid2title[qid]
                title = tokenize(title)
                mention_ent_hr[mnt].append((title, qid, 0))


    # Store data in lists, then shuffle HR data, select numData, concatenate, shuffle,  create file
    mm_ee_lr = []
    cnt_lr = 0
    for mnt in mention_ent_lr:
        
        for ent in set(mention_ent_lr[mnt]):
            mm_ee_lr.append((mnt, ent[0], ent[1], ent[2]))
            cnt_lr += 1

    mm_ee_hr = []
    cnt_hr = 0
    for mnt in mention_ent_hr:

        for ent in set(mention_ent_hr[mnt]):
            mm_ee_hr.append((mnt, ent[0], ent[1], ent[2]))
            cnt_hr += 1

    print("Length mm-ee LR language {}".format(cnt_lr))
    print("Length mm-ee HR language {}".format(cnt_hr))

    random.seed(1234)
    random.shuffle(mm_ee_hr)
    mm_ee_hr = mm_ee_hr[:num_data]
 
    random.shuffle(mm_ee_lr)
    mm_ee_lr = mm_ee_lr[:num_data]

    if weight_hr == -1: # Zero-shot setting
        mm_ee = mm_ee_hr
    else:
        mm_ee = mm_ee_lr
        mm_ee.extend(mm_ee_hr)

    random.shuffle(mm_ee)

    f_tr = open(data_tr_sampled, "w")
    for tup in mm_ee:
        f_tr.write("{} {} {} {}\n".format(tup[0], tup[1], tup[2], tup[3]))
    f_tr.close()


class tok_ngram(object):

    def __init__(self, data_tr, path, gram_range=[2,3,4,5]):
        self.vocab_mnt = {}
        self.vocab_ent = {}
        self.data_tr = data_tr
        self.path = path
        self.gram_range = gram_range

    def train(self):

        self.titles_seen = []
        with open(self.data_tr, "r") as f:
            for line in f:
                args = line.split(" ")
                self.ngrams_train(args[0], args[1])
                self.titles_seen.append(args[1])

        self.titles_seen = list(set(self.titles_seen))
        # Dump into pkl file
        self.vocab_mnt = {gram:n for n, gram in enumerate(self.vocab_mnt)}
        self.id2ngram_mnt = {self.vocab_mnt[w]:w for w in self.vocab_mnt}
        self.vocab_ent = {gram:n for n, gram in enumerate(self.vocab_ent)}
        self.id2ngram_ent = {self.vocab_ent[w]:w for w in self.vocab_ent}
        h = open(self.path, "wb")
        _pickle.dump(self.vocab_mnt, h)
        _pickle.dump(self.vocab_ent, h)
        _pickle.dump(self.titles_seen, h)
        h.close()

    def load(self):
        f = open(self.path, "rb")
        self.vocab_mnt = _pickle.load(f)
        self.vocab_ent = _pickle.load(f)
        self.titles_seen = _pickle.load(f)
        f.close()

        self.id2ngram_mnt = {self.vocab_mnt[w]:w for w in self.vocab_mnt}
        self.id2ngram_ent = {self.vocab_ent[w]:w for w in self.vocab_ent}

    def ngrams_train(self, word, entity):
        word = "ж" + word + "ж"
        for n in range(min(self.gram_range), max(self.gram_range)+1):
            for pos in range(len(word)-n+1):
                self.vocab_mnt[word[pos:(pos+n)]] = True

        entity = "ж" + entity + "ж"
        for n in range(min(self.gram_range), max(self.gram_range)+1):
            for pos in range(len(entity)-n+1):
                self.vocab_ent[entity[pos:(pos+n)]] = True

    def encode(self, query, dtype):
        subwords, subidx = self.get_ngrams(query, dtype)
        return subwords, subidx

    def get_ngrams(self, query, dtype):
        if dtype == "mnt":
            vocab = self.vocab_mnt
        elif dtype == "ent":
            vocab = self.vocab_ent

        subwords, subidx = [], []
        query = "ж" + query + "ж"
        for n in range(min(self.gram_range), max(self.gram_range)+1):
            for pos in range(len(query)-n+1):
                sbwrd = query[pos:(pos+n)]
                if sbwrd in vocab:
                    subwords.append(sbwrd)
                    subidx.append(vocab[sbwrd])
        return subwords, subidx


# multiple data2token and data2label files should be merged in one single file!
def generate_sample(data_tr, tokenizer, qids_name, voc_size_mnt, voc_size_ent, BATCH, nlines):
    indices_mnt = []
    values_mnt = []

    indices_ent = []
    values_ent = []

    indices_ent_neg = []
    values_ent_neg = []

    #nlines = int(Popen(["wc", "-l", data_tr], stdout=PIPE).communicate()[0].decode("utf-8").split(" ")[0])

    indctr = np.zeros([BATCH], dtype=np.float32)
    with open(data_tr, "r") as f:
        ent_negs = np.random.choice(qids_name, size=nlines)
        n_sample = 0
        cnt_nneg = 0
        for line in f:
            args = line[:-1].split(" ")
            mnt = args[0]
            ent = args[1]
            ind = args[3]
            sbwrds_mnt, enc_mnt = tokenizer.encode(mnt, "mnt")
            sbwrds_ent, enc_ent = tokenizer.encode(ent, "ent")
            sbwrds_ent_neg, enc_ent_neg = tokenizer.encode(ent_negs[cnt_nneg], "ent")
            cnt_nneg += 1

            # Mention
            indices_mnt.extend([[n_sample]]*len(enc_mnt))
            values_mnt.extend(enc_mnt)
                
            # Entity
            indices_ent.extend([[n_sample]]*len(enc_ent))
            values_ent.extend(enc_ent)

            # Negative Entity
            indices_ent_neg.extend([[n_sample]]*len(enc_ent_neg))
            values_ent_neg.extend(enc_ent_neg)

            indctr[n_sample] = int(ind)
            n_sample += 1

            if n_sample == BATCH:
                yield (indctr, indices_mnt, values_mnt, [voc_size_mnt], indices_ent, values_ent, [voc_size_ent], 
                        indices_ent_neg, values_ent_neg, [voc_size_ent])
                
                indices_mnt = []
                values_mnt = []

                indices_ent = []
                values_ent = []

                indices_ent_neg = []
                values_ent_neg = []

                indctr = np.zeros([BATCH], dtype=np.int32)
                n_sample = 0


def evaluation(dev_file, tokenizer, voc_size_mnt):
   
    indices = []
    values = []

    dev_mnt_str = []
    dev_ent = []
    diff = []

    n_sample = 0 

    with open(dev_file, "r") as f:
        for line in f:
            args = line[:-1].split(" ")
            qid = args[0]
            mnt = args[1]
            difficulty = int(args[2])
            sbwrds_mnt, enc_mnt = tokenizer.encode(mnt, "mnt")
            
            indices.extend([[n_sample]]*len(enc_mnt))
            values.extend(enc_mnt)
            dev_ent.append(qid)
            dev_mnt_str.append(mnt)
            diff.append(difficulty)
            n_sample += 1

    dev_mnt = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape=[voc_size_mnt])

    return dev_ent, dev_mnt, dev_mnt_str, diff
    

def parse_QID_candidates(qid2title, tokenizer):
    indices_ent = []
    values_ent = []
    qid2id = {}
    id2qid = {}
    voc_size_ent = len(tokenizer.vocab_ent)
    n_sample = 0
    for qid in qid2title:
        title = qid2title[qid]
        title = tokenize(title)
        sbwrds_ent, enc_ent = tokenizer.encode(title, "ent")
        if len(enc_ent) > 0:
            qid2id[qid] = n_sample
            id2qid[n_sample] = qid
            indices_ent.extend([[n_sample]]*len(enc_ent))
            values_ent.extend(enc_ent)
            n_sample += 1

    cands_qid = tf.sparse.SparseTensor(indices = indices_ent, values = values_ent, dense_shape=[voc_size_ent])

    return cands_qid, qid2id, id2qid

def normalise(A):
    lengths = (A**2).sum(axis=1, keepdims=True)**.5
    return A/lengths
