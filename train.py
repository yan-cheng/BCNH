import argparse
import logging
import time
import os
import json
import random
from tqdm import tqdm
import numpy as np
import torch
from dataset import QueryDataset, CandidateDataset, DictionaryDataset
from model import BCNH, RankNet
from eval import evaluate

LOGGER = logging.getLogger()


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000
    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_dictionary(dictionary_path):
    """
    load dictionary

    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(dictionary_path=dictionary_path)
    return dictionary.data

def load_queries(data_dir, filter_composite, filter_duplicate):
    """
    load query data

    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    return dataset.data

def train(args, epoch, model, train_loader):
    LOGGER.info("Train Epoch {}:".format(epoch))
    train_loss = 0
    train_steps = 0
    model.train()
    for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        model.optimizer.zero_grad()
        batch_x, batch_y = data
        scores = model(batch_x)
        loss = model.loss(scores, batch_y)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1
    train_loss /= (train_steps + 1e-9)
    return train_loss

def main(args):
    init_logging()
    init_seed(args.seed)
    print(args)

    # prepare for output
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # save all config
    with open(os.path.join(args.model_dir, "args.json"), "w+") as f:
        args_dict = {}
        for _ in args._get_kwargs():
            args_dict.update({_[0]:_[1]})
        json.dump(args_dict, f, indent=4)

    # load dictionary and queries
    train_dictionary = load_dictionary(dictionary_path=args.train_dictionary_path)
    train_queries = load_queries(
        data_dir=args.train_dir,
        filter_composite=True, # args.filter_composite
        filter_duplicate=True, # args.filter_duplicate
    )
    # filter only names
    names_in_train_dictionary = train_dictionary[:, 0]
    names_in_train_queries = train_queries[:, 0]

    # Init the model
    bcnh = BCNH()
    encoder, tokenizer = bcnh.load_bert(
        path=args.bert_dir,
        max_length=args.max_length,
        use_cuda=args.use_cuda,
    )
    bcnh.train_sparse_encoder(corpus=names_in_train_dictionary)
    sparse_weight = bcnh.init_sparse_weight(
        initial_sparse_weight=args.initial_sparse_weight,
        use_cuda=args.use_cuda
    )

    # load rank model
    model = RankNet(
        encoder=encoder,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sparse_weight=sparse_weight,
        hyper_norm_scale=args.hyper_norm_scale,
        use_cuda=args.use_cuda
    )

    # Sparse embedding and retrieve sparse candidates
    LOGGER.info("Sparse embedding and retrieve sparse candidates...")
    train_query_sparse_embeds = bcnh.embed_sparse(names=names_in_train_queries, show_progress=True)
    train_dict_sparse_embeds = bcnh.embed_sparse(names=names_in_train_dictionary, show_progress=True)
    train_sparse_score_matrix = bcnh.get_score_matrix(train_query_sparse_embeds, train_dict_sparse_embeds)
    train_sparse_candidate_idxs = bcnh.retrieve_candidate(
        score_matrix=train_sparse_score_matrix,
        topk=args.topk
    )

    # prepare for data loader of train and dev
    train_set = CandidateDataset(
        queries=train_queries,
        dicts=train_dictionary,
        tokenizer=bcnh.tokenizer,
        topk=args.topk,
        hyper_num=args.hyper_num,
        d_ratio=args.dense_ratio,
        s_score_matrix=train_sparse_score_matrix,
        s_candidate_idxs=train_sparse_candidate_idxs,
        taxonomy=args.taxonomy
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True
    )

    # load test dictionary and data
    test_dictionary = load_dictionary(dictionary_path=args.test_dictionary_path)
    test_queries = load_queries(
        data_dir=args.test_dir,
        filter_composite=args.filter_composite,
        filter_duplicate=args.filter_duplicate
    )

    # Train
    for epoch in range(1, args.epoch+1):
        LOGGER.info("Epoch {}/{}".format(epoch, args.epoch))
        LOGGER.info("train_set dense embedding for iterative candidate retrieval")
        # replace dense candidates in the train_set
        train_query_dense_embeds = bcnh.embed_dense(names=names_in_train_queries, show_progress=True)
        train_dict_dense_embeds = bcnh.embed_dense(names=names_in_train_dictionary, show_progress=True)
        train_dense_score_matrix = bcnh.get_score_matrix(
            query_embeds=train_query_dense_embeds,
            dict_embeds=train_dict_dense_embeds
        )
        train_dense_candidate_idxs = bcnh.retrieve_candidate(
            score_matrix=train_dense_score_matrix,
            topk=args.topk
        )
        train_set.set_dense_candidate_idxs(d_candidate_idxs=train_dense_candidate_idxs)

        # train
        train_loss = train(args, epoch, model, train_loader)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss, epoch))

        # Save
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.model_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            bcnh.save_model(checkpoint_dir)

        # save model last epoch
        if epoch == args.epoch:
            bcnh.save_model(args.model_dir)

    # do test
    model.eval()
    with torch.no_grad():
        result_testset = evaluate(
            model=bcnh,
            eval_dictionary=test_dictionary,
            eval_queries=test_queries,
            topk=args.topk,
            score_mode=args.score_mode
        )
        LOGGER.info("acc@1={}".format(result_testset['acc1']))
        LOGGER.info("acc@5={}".format(result_testset['acc5']))

    if args.save_predictions:
        output_file = os.path.join(args.model_dir, "predictions_eval.json")
        with open(output_file, 'w') as f:
            json.dump(result_testset, f, indent=2)


if __name__ == '__main__':

    """
    CUDA_VISIBLE_DEVICES=7 python train.py \
        --bert_dir ./pretrained/pt_biobert1.1/ \
        --model_dir exp/BCNH \
        --train_dictionary_path ./datasets/NCBI_Disease/train_dictionary.txt \
        --train_dir ./datasets/NCBI_Disease/processed_train_dev \
        --dev_dictionary_path ./datasets/NCBI_Disease/dev_dictionary.txt \
        --dev_dir ./datasets/NCBI_Disease/processed_dev \
        --test_dictionary_path ./datasets/NCBI_Disease/test_dictionary.txt \
        --test_dir ./datasets/NCBI_Disease/processed_test \
        --epoch 10 \
        --hyper_num 10 \
        --hyper_norm_scale 1 \
        --taxonomy ./datasets/NCBI_Disease/CTD_diseases_MEDIC_2021.02.01.json
    """

    parser = argparse.ArgumentParser(description='Model train')
    # Data directory
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--bert_dir', type=str)
    parser.add_argument('--train_dictionary_path', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--dev_dictionary_path', type=str)
    parser.add_argument('--dev_dir', type=str)
    parser.add_argument('--test_dictionary_path', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--taxonomy', type=str)
    # Dataloader Preprocess
    parser.add_argument('--filter_composite', action="store_true", help="filter out composite mention queries")
    parser.add_argument('--filter_duplicate', action="store_true", help="filter out duplicate queries")
    # Train
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--initial_sparse_weight', type=float, default=0)
    parser.add_argument('--dense_ratio', type=float, default=0.5)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--hyper_num', type=int, default=10)
    parser.add_argument('--hyper_norm_scale', type=float, default=1)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--save_checkpoint_all', type=bool, default=False)
    # Test
    parser.add_argument('--score_mode', type=str, default="hybrid")
    parser.add_argument('--save_predictions', type=bool, default=True)
    args = parser.parse_args()

    main(args)