import argparse
import pickle
import time
from dump_processing import construct_probmap, load_probmap
from cand_gen import get_cands_by_words, get_cands_by_surface 
from collections import Counter
import ipdb
import numpy as np

def get_cands(query_mention, ncands, p2t_pl, w2t_pl, p2t_tl=None, w2t_tl=None):
    candidates = Counter()
    if p2t_tl:
        get_cands_by_surface(candidates, query_mention, p2t_tl)

    if len(candidates) < ncands:
        #print(f'Found {len(candidates)} cands via mention string using the target language. Using pivot language now for obtaining additional candidates!')
        get_cands_by_surface(candidates, query_mention, p2t_pl)

    if len(candidates) < ncands:
        if w2t_tl:
            #print(f'Found {len(candidates)} cands via mention string using the target and pivot language. Using individual words and the target language now for obtaining additional candidates!')
            get_cands_by_words(candidates, query_mention, w2t_tl)

    if len(candidates) < ncands:
        #print(f'Found {len(candidates)} cands via mention string using the target and pivot language, and via individual words using the target language. Using individual words and the pivot language now for obtaining additional candidates!')
        get_cands_by_words(candidates, query_mention, w2t_pl)

    return candidates

def generate_candidates(data_dir, probmap_dir, plang, tlang, ncands, zeroshot=True):
    if zeroshot:
        print(f"Zeroshot setting: using {plang} as pivot language")
        fout = open(f'{tlang}_mentions_tt_type_wikipriors_zeroshot.txt', "w")
    else:
        print(f"Supervised setting: using {tlang} as target language and {plang} as pivot language")
        fout = open(f'{tlang}_mentions_tt_type_wikipriors_supervised.txt', "w")

    p2t_pl, w2t_pl = load_probmap(plang, probmap_dir)
    if not zeroshot:
        p2t_tl, w2t_tl = load_probmap(tlang, probmap_dir)

    query_path = f'{data_dir}/{tlang}'
    fqueries = open(f'{query_path}/mentions_tt_type.txt')
    num_queries = 0
    recall_dict = {}
    
    for query in fqueries:
        query = query.strip().split(" ")
        query_gt = query[0]; query_mention = query[1]; query_type = query[2]
        '''
        if num_queries > 50:
            break
        '''
        if zeroshot:
            candidates = get_cands(query_mention, ncands, p2t_pl, w2t_pl)
        else:
            candidates = get_cands(query_mention, ncands, p2t_pl, w2t_pl, p2t_tl, w2t_tl)

        if len(candidates) == 0:
            #print(query,0)
            fout.write(query_gt+" "+query_mention+" "+query_type+" 0\n")
            rank = 0
        else:
            cand_qids = list(list(zip(*candidates.most_common(ncands)))[0])
            try:
                rank = cand_qids.index(query_gt) + 1
            except ValueError:
                rank = 0
            '''
            for (qid, prior) in candidates.most_common(ncands):
                print(query)
                print(qid, prior)
            '''
            fout.write(query_gt+" "+query_mention+" "+query_type+" "+str(rank)+"\n")

        recall = 0 if rank == 0 else 1
        if query_type not in recall_dict:
            recall_dict[query_type] = [recall]
        else:
            recall_dict[query_type].append(recall)
        num_queries += 1

    overall_recall = 0; num_queries = 0
    for query_type in recall_dict:
        overall_recall += np.sum(recall_dict[query_type])
        num_queries += len(recall_dict[query_type])
        print(f'Query-type: {query_type}, Recall: {np.sum(recall_dict[query_type]) / len(recall_dict[query_type])}')
    print(f'Overall Recall: {overall_recall/num_queries}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tlang', default='scn')
    parser.add_argument('--plang', default='it')
    parser.add_argument('--ncands', default=30, help='Int: number of candidates. Default: 30')
    parser.add_argument('--zeroshot', action="store_true", default=False, help='Bool: used for zeroshot candgen setup.')

    parser.add_argument('--data_dir', default='../mentions_dumps')
    parser.add_argument('--probmap_dir', default='./probmap')

    args = parser.parse_args()
    tlang = args.tlang
    plang = args.plang

    ncands = args.ncands
    zeroshot = args.zeroshot

    data_dir = args.data_dir
    probmap_dir = args.probmap_dir

    construct_probmap(tlang, data_dir, probmap_dir)
    construct_probmap(plang, data_dir, probmap_dir)

    generate_candidates(data_dir, probmap_dir, plang, tlang, ncands, zeroshot)

if __name__ == '__main__':
    main()
