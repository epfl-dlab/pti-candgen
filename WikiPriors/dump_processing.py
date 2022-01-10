import pickle
import os
from collections import Counter
import time

def construct_p2tw2t(mention_dump_path, probmap_path):
    p2t = {}; w2t = {}
    fmentions = open(mention_dump_path)
    print(f'Processing {mention_dump_path} mention dumps file!')
    for line in fmentions:
        line = line.strip().split(" ")
        if len(line)!=3:
            #print(line)
            continue
        qid = line[1]
        mention = line[2]
        mention_toks = mention.split("_")
        if mention not in p2t:
            p2t[mention] = Counter()
        p2t[mention][qid] += 1

        for tok in mention_toks:
            if tok not in w2t:
                w2t[tok] = Counter()
            w2t[tok][qid] += 1
        
    for mention in p2t:
        qids = p2t[mention]
        total = sum(qids.values())
        for qid in qids:
            qids[qid] /= total

    for tok in w2t:
        qids = w2t[tok]
        total = sum(qids.values())
        for qid in qids:
            qids[qid] /= total

    with open(f'{probmap_path}/p2t.pickle', 'wb') as fpkl:
        pickle.dump(p2t, fpkl)

    with open(f'{probmap_path}/w2t.pickle', 'wb') as fpkl:
        pickle.dump(w2t, fpkl)

def load_probmap(lang, probmap_dir):
    probmap_path = f'{probmap_dir}/{lang}'

    tstart = time.time()
    p2t = pickle.load(open(f'{probmap_path}/p2t.pickle',"rb"))
    w2t = pickle.load(open(f'{probmap_path}/w2t.pickle',"rb"))
    tend = time.time()
    print(f"It took {tend-tstart} seconds to load {probmap_path}/p2t.pickle and {probmap_path}/w2t.pickle files")
    return p2t, w2t

def construct_probmap(lang, data_dir, probmap_dir):
    mention_dump_path = f'{data_dir}/{lang}/mentions_tr.txt'
    if not os.path.exists(probmap_dir):
        os.makedirs(probmap_dir)
    probmap_path = f'{probmap_dir}/{lang}'
    if not os.path.exists(probmap_path):
        os.makedirs(probmap_path)

    if not os.path.exists(f'{probmap_path}/p2t.pickle'):
        print(f'Creating {probmap_path}/p2t.pickle and {probmap_path}/w2t.pickle files')
        tstart = time.time()
        construct_p2tw2t(mention_dump_path, probmap_path)
        tend = time.time()
        print(f'It took {tend-tstart} seconds to create {probmap_path}/p2t.pickle and {probmap_path}/w2t.pickle files')
    else:
        print(f'Using existent {probmap_path}/p2t.pickle and {probmap_path}/w2t.pickle files')
