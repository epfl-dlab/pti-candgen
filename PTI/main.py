import os
import sys
import _pickle
import numpy as np
from collections import Counter, defaultdict
from utils import evaluation, create_data, tok_ngram, generate_ngram2ent, merge_tables, compute_priors
import time

# HYPERPARAMETERS
weight_hr = float(sys.argv[1]) # Alpha. -1 means Zero-shot 
weight_e2g = float(sys.argv[2]) # Lambda
lang = sys.argv[3] # Target language
hr_lang = sys.argv[4] # Pivot language


# DATA PREPROCESSING
path_tr = "../mentions_dumps/{}/mentions_tr.txt".format(lang)
path_tr_hr = "../mentions_dumps/{}/mentions_tr.txt".format(hr_lang)

if not os.path.exists("data/{}".format(lang)):
    os.mkdir("data/{}".format(lang))

if weight_hr == -1: # Zero-shot
    data_info = "data/{}/info_ngram_lookup_ZS_hr={}.pkl".format(lang, hr_lang)
else:
    data_info = "data/{}/info_ngram_lookup_hr={}.pkl".format(lang, hr_lang)

if os.path.exists(data_info):
    mnt2ent_hr, mnt2ent_lr, ent2ind = _pickle.load(open(data_info, "rb"))
else:
    mnt2ent_hr, mnt2ent_lr, ent2ind = create_data(path_tr, path_tr_hr, data_info, weight_hr)

# N-GRAMS TOKENIZER
if weight_hr == -1: # Zero-shot
    path_tkn = "data/{}/ngram_lookup_ZS_hr={}_vocabulary.pkl".format(lang, hr_lang)
    path_cnt_hr = "data/{}/ngram_counter_ZS_{}.pkl".format(lang, hr_lang)
    path_cnt_lr = "data/{}/ngram_counter_ZS_{}.pkl".format(lang, lang)
else:
    path_tkn = "data/{}/ngram_lookup_hr={}_vocabulary.pkl".format(lang, hr_lang)
    path_cnt_hr = "data/{}/ngram_counter_{}.pkl".format(lang, hr_lang)
    path_cnt_lr = "data/{}/ngram_counter_{}.pkl".format(lang, lang)

data_tr = [mnt2ent_hr, mnt2ent_lr]
if  os.path.exists(path_tkn):
    tokenizer = tok_ngram(data_tr, path_tkn)
    tokenizer.load()
    voc_size = len(tokenizer.vocab)
    ngram2ent_hr, ent2ngram_hr = _pickle.load(open(path_cnt_hr, "rb"))
    ngram2ent_lr, ent2ngram_lr = _pickle.load(open(path_cnt_lr, "rb"))
else:
    tokenizer = tok_ngram(data_tr, path_tkn)
    tokenizer.train()
    voc_size = len(tokenizer.vocab)
    start = time.time()
    ngram2ent_hr, ent2ngram_hr = generate_ngram2ent(mnt2ent_hr, tokenizer, path_cnt_hr)
    ngram2ent_lr, ent2ngram_lr = generate_ngram2ent(mnt2ent_lr, tokenizer, path_cnt_lr)


# Counts-level merging
# MERGE
ngram2ent = merge_tables(ngram2ent_hr, ngram2ent_lr, weight_hr)
ent2ngram = merge_tables(ent2ngram_hr, ent2ngram_lr, weight_hr)
# Convert to prior probabilities
ng2ind = tokenizer.vocab
extLT, extLT_e2g  = compute_priors(ngram2ent, ent2ngram, ent2ind, ng2ind)

# VALIDATION
dev_file  = "../mentions_dumps/{}/mentions_dev_type.txt".format(lang)
dev_data, dev_diff = evaluation(dev_file, tokenizer, ent2ind, voc_size)

cnt_nones = 0
easy, medium, hard, total = [], [], [], []
for ne, query in  enumerate(dev_data):
    ngram_query = query[0]
    len_ngram = len(ngram_query)
    qid = query[1]
    if len(ngram_query) != 0:    
        scores_LT = extLT[ngram_query].mean(0)
        scores_LT_e2g = extLT_e2g[ngram_query].mean(0)
        scores = scores_LT + weight_e2g*scores_LT_e2g
        if qid in ent2ind:
            is_shortlisted = int(ent2ind[qid] in scores.argpartition(-30)[0, -30:])
        else:
            is_shortlisted = 0
            cnt_nones += 1
    else:
        is_shortlisted = 0

    total.append(is_shortlisted)
    diff = dev_diff[ne]
    if diff == 0:
        easy.append(is_shortlisted)
    if diff == 1:
        medium.append(is_shortlisted)
    if diff == 2:
        hard.append(is_shortlisted)

recall_micro = np.round(np.mean(total), 4)
recall_macro = np.round(np.mean([np.mean(easy), np.mean(medium), np.mean(hard)]), 4)
print("PERFORMANCE ON VALIDATION SET. Number of samples {}. Recall@30 micro {} and macro {}.".format(len(total), recall_micro, recall_macro))
print("TYPE I -- Number of samples {}. Recall@30 {}.".format(len(easy), np.round(np.mean(easy), 4)))
print("TYPE II -- Number of samples {}. Recall@30 {}.".format(len(medium), np.round(np.mean(medium), 4)))
print("TYPE III -- Number of samples {}. Recall@30 {}.".format(len(hard), np.round(np.mean(hard), 4)))
print("Queries where completion is not in the candidate space: {}.".format(cnt_nones))

# TEST
test_file  = "../mentions_dumps/{}/mentions_tt_type.txt".format(lang)
test_data, test_diff = evaluation(test_file, tokenizer, ent2ind, voc_size)

for k in [10, 20, 30]:
    easy, medium, hard, total = [], [], [], []
    cnt_nones = 0
    for ne, query in enumerate(test_data):
        ngram_query = query[0]
        len_ngram = len(ngram_query)
        qid = query[1]
        if len(ngram_query) != 0:
            scores_LT = extLT[ngram_query].mean(0)
            scores_LT_e2g = extLT_e2g[ngram_query].mean(0)
            scores = scores_LT + weight_e2g*scores_LT_e2g
            if qid in ent2ind:
                is_shortlisted = int(ent2ind[qid] in scores.argpartition(-k)[0, -k:])
            else:
                is_shortlisted = 0
                cnt_nones += 1
        else:
            is_shortlisted = 0
        total.append(is_shortlisted)
        diff = test_diff[ne]
        if diff == 0:
            easy.append(is_shortlisted)
        if diff == 1:
            medium.append(is_shortlisted)
        if diff == 2:
            hard.append(is_shortlisted)

    recall_micro = np.round(np.mean(total), 4)
    recall_macro = np.round(np.mean([np.mean(easy), np.mean(medium), np.mean(hard)]), 4)
    print("PERFORMANCE ON TEST SET. Number of samples {}. Recall@{} micro {} and macro {}.".format(len(total), k, recall_micro, recall_macro))
    print("TYPE I -- Number of samples {}. Recall@{} {}.".format(len(easy), k, np.round(np.mean(easy), 4)))
    print("TYPE II -- Number of samples {}. Recall@{} {}.".format(len(medium), k, np.round(np.mean(medium), 4)))
    print("TYPE III -- Number of samples {}. Recall@{} {}.".format(len(hard), k, np.round(np.mean(hard), 4)))
    print("Queries where completion is not in the candidate space: {}.".format(cnt_nones))

