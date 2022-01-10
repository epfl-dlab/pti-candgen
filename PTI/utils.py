import numpy as np
import _pickle
import subprocess
from collections import Counter
from scipy.sparse import csr_matrix

def create_data(path_tr, path_tr_hr, data_info, weight_hr):


    # LR language
    mnt2ent_lr = {}
    ent2ind = {}
    cnt = 0
    with open(path_tr, "r") as f:
        for line in f:
            args = line[:-1].split(" ")
            mnt = args[2]
            qid = args[1]

            if mnt not in mnt2ent_lr:
                mnt2ent_lr[mnt] = []
            mnt2ent_lr[mnt].append(qid)

            if qid not in ent2ind and weight_hr != -1:
                ent2ind[qid] = cnt 
                cnt += 1

    # HR language
    mnt2ent_hr = {}
    with open(path_tr_hr, "r") as f:
        for line in f:
            args = line[:-1].split(" ")
            mnt = args[2]
            qid = args[1]

            if mnt not in mnt2ent_hr:
                mnt2ent_hr[mnt] = []
            mnt2ent_hr[mnt].append(qid)
            #print(mnt2ent_hr)
            if qid not in ent2ind:
                ent2ind[qid] = cnt
                cnt += 1

    if weight_hr == -1:
        mnt2ent_lr = {}

    f_info = open(data_info, "wb")
    _pickle.dump([mnt2ent_hr, mnt2ent_lr, ent2ind], f_info)
    f_info.close()

    return mnt2ent_hr, mnt2ent_lr, ent2ind


class tok_ngram(object):

    def __init__(self, data_tr, path, gram_range=[2,3,4,5]):
        self.vocab = {}
        self.data_tr = data_tr
        self.gram_range = gram_range
        self.path = path

    def train(self):

        for mnt2ent in self.data_tr:
            for mnt in mnt2ent:
                self.ngrams_train(mnt)

        # Dump into pkl file
        self.vocab = {gram:n for n, gram in enumerate(self.vocab)}
        self.id2ngram = {self.vocab[w]:w for w in self.vocab}
        h = open(self.path, "wb")
        _pickle.dump(self.vocab, h)
        h.close()

    def load(self):
        self.vocab = _pickle.load(open(self.path, "rb"))
        self.id2ngram = {self.vocab[w]:w for w in self.vocab}

    def encode(self, word):
        subwords, subidx = self.get_ngrams(word)
        return subwords, subidx

    def ngrams_train(self, word):
        word = "ж" + word + "ж"
        for n in range(min(self.gram_range), max(self.gram_range)+1):
            for pos in range(len(word)-n+1):
                self.vocab[word[pos:(pos+n)]] = True

    def get_ngrams(self, word):
        subwords, subidx = [], []
        word = "ж" + word + "ж"
        for n in range(min(self.gram_range), max(self.gram_range)+1):
            for pos in range(len(word)-n+1):
                sbwrd = word[pos:(pos+n)]
                if sbwrd in self.vocab:
                    subwords.append(sbwrd)
                    subidx.append(self.vocab[word[pos:(pos+n)]])
        return subwords, subidx


def generate_ngram2ent(mnt2ent, tokenizer, path):
    
    ngram2ent = {}
    ent2ngram = {}
    for mnt in mnt2ent:
        ngrams = tokenizer.encode(mnt)[0]
        ents = mnt2ent[mnt]
        for ng in ngrams:
            if ng not in ngram2ent:
                ngram2ent[ng] = []
            ngram2ent[ng].extend(ents)
            
        for ent in ents:
            if ent not in ent2ngram:
                ent2ngram[ent] = []
            ent2ngram[ent].extend(ngrams)

    for ng in ngram2ent:
        ngram2ent[ng] = Counter(ngram2ent[ng])

    for ent in ent2ngram:
        ent2ngram[ent] = Counter(ent2ngram[ent])

    # SAVE n-gram to ent
    f = open(path, "wb")
    _pickle.dump([ngram2ent, ent2ngram], f)
    f.close()

    return ngram2ent, ent2ngram


def merge_tables(ngram2ent_hr, ngram2ent_lr, weight_hr):

    assert len(ngram2ent_hr) > len(ngram2ent_lr)
    
    # Combining both counters. HR counter is weighted by a factor weight_hr
    for ng in ngram2ent_lr:

        if ng in ngram2ent_hr:

            for ent in ngram2ent_hr[ng]:
                ngram2ent_hr[ng][ent] = weight_hr*ngram2ent_hr[ng][ent]

            ngram2ent_hr[ng] = ngram2ent_hr[ng] + ngram2ent_lr[ng]

        else:
            ngram2ent_hr[ng] = ngram2ent_lr[ng]

    return ngram2ent_hr

def compute_priors(ngram2ent, ent2ngram, ent2ind, ng2ind, mu=0, thresh=0):

    rows = []
    cols = []
    data = []
    n_entries = 0
    n_removed = 0
    for ng in ngram2ent:
        ng_row = ng2ind[ng]
        total = sum(ngram2ent[ng].values())
        vsize = len(ngram2ent[ng])
        for ent in ngram2ent[ng]:
            n_entries += 1
            ent_col = ent2ind[ent]
            value = (ngram2ent[ng][ent] + mu)/ (total + mu*vsize)
            if value >= thresh:
                rows.append(ng_row)
                cols.append(ent_col)                
                data.append(value)
            else:
                n_removed += 1

    extLT = csr_matrix((data, (rows, cols)), shape=(len(ng2ind), len(ent2ind))) 

    rows = []
    cols = []
    data = []
    for ent in ent2ngram:
        ent_row = ent2ind[ent]
        total = sum(ent2ngram[ent].values())
        vsize = len(ent2ngram[ent])
        for ng in ent2ngram[ent]:
            n_entries += 1
            ng_col = ng2ind[ng]
            value = (ent2ngram[ent][ng] + mu)/ (total + mu*vsize)
            if value >= thresh:
                rows.append(ng_col)
                cols.append(ent_row)
                data.append(value)
            else:
                n_removed += 1

    extLT_e2g = csr_matrix((data, (rows, cols)), shape=(len(ng2ind), len(ent2ind)))
    print("Total Entries {}. Removed Entries {}. Induced Sparsity {}.".format(n_entries, n_removed, n_removed/n_entries))

    return extLT, extLT_e2g


def evaluation(dev_file, tokenizer, ent2ind, voc_size):
    data = []
    row = []
    col = []
    Y = []
    diff = []
    n_sample = 0

    dev_mnt = []
    with open(dev_file, "r") as f:
        for line in f:
            args = line[:-1].split(" ")
            feat = args[1]
            qid = args[0]
            difficulty = int(args[2])
            enc_sbwrds, enc_feat = tokenizer.encode(feat)
            dev_mnt.append((enc_feat, qid, difficulty, feat))
            diff.append(difficulty)

    return dev_mnt, diff
