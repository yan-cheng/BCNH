import argparse
import logging
import os
import numpy as np
from tqdm import tqdm
import torch
import json
from dataset import QueryDataset, DictionaryDataset
from model import BCNH

LOGGER = logging.getLogger()

def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def check_label(predicted_cui, golden_cui):
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|")))) > 0)


def evaluate_topk_acc(data):
    queries = data['queries']
    k = check_k(queries)
    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1] # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])
            # When all mentions in a query are predicted correctly,
            # we consider it as a hit 
            if mention_hit == len(mentions):
                hit += 1
        data['acc{}'.format(i+1)] = hit/len(queries)
    return data


def predict_topk(model, eval_dictionary, eval_queries, topk, score_mode='hybrid'):
    sparse_weight = model.get_sparse_weight().item() # must be scalar value
    print("sparse weight : {}".format(sparse_weight))

    # embed dictionary
    dict_sparse_embeds = model.embed_sparse(names=eval_dictionary[:, 0], show_progress=False)
    dict_dense_embeds = model.embed_dense(names=eval_dictionary[:, 0], show_progress=False)

    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries)):
        mentions = eval_query[0].replace("+", "|").split("|")
        golden_cui = eval_query[1].replace("+", "|")
        dict_mentions = []
        for mention in mentions:
            mention_sparse_embeds = model.embed_sparse(names=np.array([mention]))
            mention_dense_embeds = model.embed_dense(names=np.array([mention]))
            # get score matrix
            sparse_score_matrix = model.get_score_matrix(
                query_embeds=mention_sparse_embeds,
                dict_embeds=dict_sparse_embeds
            )
            dense_score_matrix = model.get_score_matrix(
                query_embeds=mention_dense_embeds,
                dict_embeds=dict_dense_embeds
            )
            if score_mode == 'hybrid':
                score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
            elif score_mode == 'dense':
                score_matrix = dense_score_matrix
            elif score_mode == 'sparse':
                score_matrix = sparse_score_matrix
            else:
                raise NotImplementedError()

            candidate_idxs = model.retrieve_candidate(
                score_matrix=score_matrix,
                topk=topk
            )
            np_candidates = eval_dictionary[candidate_idxs].squeeze()
            dict_candidates = []
            for np_candidate in np_candidates:
                dict_candidates.append({
                    'name':np_candidate[0],
                    'cui':np_candidate[1],
                    'label':check_label(np_candidate[1], golden_cui)
                })
            dict_mentions.append({
                'mention':mention,
                'golden_cui':golden_cui, # golden_cui can be composite cui
                'candidates':dict_candidates
            })
        queries.append({'mentions':dict_mentions})
    result = {'queries':queries}
    return result

def evaluate(model, eval_dictionary, eval_queries, topk, score_mode='hybrid'):
    result = predict_topk(model, eval_dictionary, eval_queries, topk, score_mode)
    result = evaluate_topk_acc(result)
    return result


if __name__ == "__main__":
    """
    python src/eval.py \
        --model_dir /data/yancheng/code/Projects/BioSH/exp/NCBI/CENorm10-0 \
        --dictionary_path /data/yancheng/code/Projects/BioSH/datasets/NCBI_Disease/test_dictionary.txt\
        --data_dir /data/yancheng/code/Projects/BioSH/datasets/NCBI_Disease/processed_test \
        --output_dir /data/yancheng/code/Projects/BioSH/exp/NCBI_R/CENorm
    """
    parser = argparse.ArgumentParser(description='BioSyn evaluation')
    # Required
    parser.add_argument('--model_dir', required=True, help='Directory for model')
    parser.add_argument('--dictionary_path', type=str, required=True, help='dictionary path')
    parser.add_argument('--data_dir', type=str, required=True, help='data set to evaluate')
    # Run settings
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--topk',  type=int, default=20)
    parser.add_argument('--score_mode',  type=str, default='hybrid', help='hybrid/dense/sparse')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--filter_composite', action="store_true", help="filter out composite mention queries")
    parser.add_argument('--filter_duplicate', action="store_true", help="filter out duplicate queries")
    parser.add_argument('--save_predictions', action="store_true")
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)
    args = parser.parse_args()
    
    # logging
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)
    print(args)

    # load dictionary and data
    eval_dictionary = DictionaryDataset(dictionary_path = args.dictionary_path).data
    eval_queries = QueryDataset(
        data_dir=args.data_dir,
        filter_composite=args.filter_composite,
        filter_duplicate=args.filter_duplicate
    ).data

    bcnh = BCNH().load_model(
        path=args.model_dir,
        max_length=args.max_length,
        use_cuda=args.use_cuda
    )
    with torch.no_grad():
        result_evalset = evaluate(
            model=bcnh,
            eval_dictionary=eval_dictionary,
            eval_queries=eval_queries,
            topk=args.topk,
            score_mode=args.score_mode
        )
    
        LOGGER.info("acc@1={}".format(result_evalset['acc1']))
        LOGGER.info("acc@5={}".format(result_evalset['acc5']))
        
        if args.save_predictions:
            output_file = os.path.join(args.output_dir, "predictions_eval.json")
            with open(output_file, 'w') as f:
                json.dump(result_evalset, f, indent=2)
