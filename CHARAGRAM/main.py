import os
import sys
import _pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from model import Speller
from utils import generate_sample, qid_space, evaluation, normalise, create_data, tok_ngram, parse_QID_candidates
from subprocess import Popen, PIPE
import time

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
tf.config.threading.set_intra_op_parallelism_threads(10) #or lower values
tf.config.threading.set_inter_op_parallelism_threads(10) #or lower values

# FIXED HYPERPARAMETERS
BATCH = 64
EMBED_SIZE = 300
EPOCHS = 200

# TUNABLE HYPERPARAMETERS
weight_hr = float(sys.argv[1]) # mu. -1 indicates Zero-shot
num_data = int(sys.argv[2]) #80000

# LANGUAGES
lang = sys.argv[3] #"scn"
hr_lang = sys.argv[4] #"it"


if not os.path.exists("data/{}".format(lang)):
    os.mkdir("data/{}".format(lang))

path_tr = "../mentions_dumps/{}/mentions_tr.txt".format(lang)
path_tr_hr = "../mentions_dumps/{}/mentions_tr.txt".format(hr_lang)

# EN world used in Charagram
qid2title = _pickle.load(open("../qid2title_CHAR.pkl", "rb"))

# EXTENDING EN CANDIDATE SPACE WITH HR&LR
if weight_hr == -1: #Zero-Shot
    qid2title_add_fi = "data/{}/qid2title_ZS_hr={}.pkl".format(lang, hr_lang)
else:
    qid2title_add_fi = "data/{}/qid2title_hr={}.pkl".format(lang, hr_lang)

if not os.path.exists(qid2title_add_fi):
    qid2title_add = qid_space(path_tr, path_tr_hr, weight_hr, qid2title_add_fi)
else:
    qid2title_add = _pickle.load(open(qid2title_add_fi, "rb"))

# Merge EN and HR&LR worlds 
qid2title.update(qid2title_add)

# DATA
if weight_hr == -1: #Zero-Shot
    data_tr_sampled = "data/{}/mentions_tr_ZS_hr={}_size={}.txt".format(lang, hr_lang, num_data)
else:
    data_tr_sampled = "data/{}/mentions_tr_hr={}_size={}.txt".format(lang, hr_lang, num_data)

if not os.path.exists(data_tr_sampled):
     create_data(path_tr, path_tr_hr, data_tr_sampled, num_data, qid2title, weight_hr)

# TOKENIZER
if weight_hr == -1:  #Zero-Shot
    path = "data/{}/charagram_ZS_hr={}_vocabulary_size={}.pkl".format(lang, hr_lang, num_data)
else:
    path = "data/{}/charagram_hr={}_vocabulary_size={}.pkl".format(lang, hr_lang, num_data)

if  os.path.exists(path):
    tokenizer = tok_ngram(data_tr_sampled, path)
    tokenizer.load()
else:
    tokenizer = tok_ngram(data_tr_sampled, path)
    tokenizer.train()

voc_size_mnt = len(tokenizer.vocab_mnt)
voc_size_ent = len(tokenizer.vocab_ent)
qids_name_tr = tokenizer.titles_seen
nlines = int(Popen(["wc", "-l", data_tr_sampled], stdout=PIPE).communicate()[0].decode("utf-8").split(" ")[0])

# DATA FEEDER
def gen():
    return generate_sample(data_tr_sampled, tokenizer, qids_name_tr, voc_size_mnt, voc_size_ent, BATCH, nlines)

# DATASET
dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int64, tf.int32, tf.int64, tf.int64, tf.int32, tf.int64,
    tf.int64, tf.int32, tf.int64))
dataset = dataset.map(lambda q, s_mnt, t_mnt, v_mnt, s_ent, t_ent, v_ent, s_entn, t_entn, v_entn: 
        (q, tf.SparseTensor(s_mnt, t_mnt, v_mnt), tf.SparseTensor(s_ent, t_ent, v_ent), tf.SparseTensor(s_entn, t_entn, v_entn)), 
        num_parallel_calls=4)
dataset = dataset.prefetch(BATCH*2)

# VALIDATION
dev_file  = "../mentions_dumps/{}/mentions_dev_type.txt".format(lang)
dev_ent, dev_mnt, dev_mnt_str, difficulty = evaluation(dev_file, tokenizer, voc_size_mnt)

cands_qid, qid2id, id2qid= parse_QID_candidates(qid2title, tokenizer)

# MODEL INITIALIZATION
model = Speller(voc_size_mnt, voc_size_ent, EMBED_SIZE)
optimizer = tf.optimizers.SGD(0.1)
total_samples = int(Popen(["wc", "-l", data_tr_sampled], stdout=PIPE).communicate()[0].decode("utf-8").split(" ")[0])
total_batches = total_samples / BATCH

# LOCATION TO SAVE
checkpoint_dir = './models/{}/charagram_char_hr={}_{}_{}'.format(lang, hr_lang, num_data, weight_hr)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)
weight_hr = np.sqrt(weight_hr*weight_hr)


best_micro_recall = 0
best_macro_recall = 0
last_update = 0
for ep in range(EPOCHS):
    
    loss_epoch = 0
    print("EPOCH {}".format(ep))
    
    for (n_batch, (ind_batch, mnt_batch, ent_pos_batch, ent_neg_batch)) in enumerate(dataset):
        
        with tf.GradientTape() as tape:
            sim_mnt_pos, sim_mnt_neg = model(mnt_batch, ent_pos_batch, ent_neg_batch)
            loss = tf.reduce_mean(tf.maximum(weight_hr, ind_batch)*tf.maximum(0., 1. + sim_mnt_pos - sim_mnt_neg))
        
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        loss_epoch += loss
        
        # Report progress
        if not (n_batch+1) % max(1, np.round(total_batches / 40)):
            progress = np.round(((n_batch+1) / total_batches)*100)
            avg_loss = loss_epoch / (n_batch+1)
            print("Epoch {}. Progress {}. Average loss in the epoch until now is {}".format(ep, progress, avg_loss))
 
    # VALIDATION
    qid_emb = model.ent_rpr(cands_qid).numpy()
    qid_emb = normalise(qid_emb)
    mnt_embed = model.mnt_rpr(dev_mnt).numpy()
    mnt_embed = normalise(mnt_embed) 
    scores = np.dot(mnt_embed, qid_emb.T)
    # Check if the right entity is in the TOP-30 (much faster than computing np.argsort)
    k = 30
    ranking = np.argpartition(scores, -k)[:, -k:]
    easy, medium, hard, total = [], [], [], []
    cnt_nones = 0
    for which in range(len(dev_ent)):
        if dev_ent[which] in qid2id:
            is_shortlisted = int(qid2id[dev_ent[which]] in ranking[which])
        else:
            is_shortlisted = 0
            cnt_nones += 1
        total.append(is_shortlisted)

        diff = difficulty[which]
        # Break down the performance by difficulty   
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

    if recall_micro > best_micro_recall:
        print("Saving model...")
        checkpoint.save(file_prefix = checkpoint_prefix)
        best_micro_recall = recall_micro
        last_update = 0
    else: 
        last_update += 1
    
    if last_update == 50:
        break


# TEST with the best validated model
latest = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint.restore(latest)

test_file  = "../mentions_dumps/{}/mentions_tt_type.txt".format(lang)
test_ent, test_mnt, test_mnt_str, difficulty = evaluation(test_file, tokenizer, voc_size_mnt)

qid_emb = model.ent_rpr(cands_qid).numpy()
qid_emb = normalise(qid_emb)
mnt_embed = model.mnt_rpr(test_mnt).numpy()
mnt_embed = normalise(mnt_embed)
scores = np.dot(mnt_embed, qid_emb.T)
# Check if the right entity is in the TOP-K(much faster than np.argsort)
for k in [10, 20, 30]:
    cnt_nones = 0
    ranking = np.argpartition(scores, -k)[:, -k:]
    easy, medium, hard, total = [], [], [], []
    for which in range(len(test_ent)):
        if test_ent[which] in qid2id:
            is_shortlisted = int(qid2id[test_ent[which]] in ranking[which])
        else:
            is_shortlisted = 0
            cnt_nones += 1
        total.append(is_shortlisted)

        diff = difficulty[which]
        # Break down the performance by difficulty   
        if diff == 0:
            easy.append(is_shortlisted)
        if diff == 1:
            medium.append(is_shortlisted)
        if diff == 2:
            hard.append(is_shortlisted)

    recall_micro = np.round(np.mean(total), 4)
    recall_macro = np.round(np.mean([np.mean(easy), np.mean(medium), np.mean(hard)]), 4)
    print("PERFORMANCE ON TEST SET (BEST MODEL). Number of samples {}. Recall@{} micro {} and macro {}.".format(len(total), k, recall_micro, recall_macro))
    print("TYPE I -- Number of samples {}. Recall@{} {}.".format(len(easy), k, np.round(np.mean(easy), 4)))
    print("TYPE II -- Number of samples {}. Recall@{} {}.".format(len(medium), k, np.round(np.mean(medium), 4)))
    print("TYPE III -- Number of samples {}. Recall@{} {}.".format(len(hard), k, np.round(np.mean(hard), 4)))
    print("Queries where completion is not in the candidate space: {}.".format(cnt_nones))
