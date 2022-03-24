import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from transformers import BertConfig, BertTokenizer, BertModel
from ranknet import SparseEncoder, RankNet


LOGGER = logging.getLogger()


class BCNH(object):
    """
    Wrapper class for dense encoder and sparse encoder
    """

    def __init__(self):
        self.tokenizer = None
        self.encoder = None
        self.sparse_encoder = None
        self.sparse_weight = None

    def init_sparse_weight(self, initial_sparse_weight, use_cuda):
        """
        Parameters
        ----------
        initial_sparse_weight : float
            initial sparse weight
        """
        if use_cuda:
            self.sparse_weight = nn.Parameter(torch.empty(1).cuda())
        else:
            self.sparse_weight = nn.Parameter(torch.empty(1))
        self.sparse_weight.data.fill_(initial_sparse_weight) # init sparse_weight
        return self.sparse_weight

    def train_sparse_encoder(self, corpus):
        self.sparse_encoder = SparseEncoder().fit(corpus)
        return self.sparse_encoder

    def get_dense_encoder(self):
        return self.encoder

    def get_dense_tokenizer(self):
        return self.tokenizer

    def get_sparse_encoder(self):
        return self.sparse_encoder

    def get_sparse_weight(self):
        return self.sparse_weight

    def save_model(self, path):
        # save bert model, bert config
        self.encoder.save_pretrained(path)
        # save bert vocab
        self.tokenizer.save_vocabulary(path)
        # save sparse encoder
        sparse_encoder_path = os.path.join(path, 'sparse_encoder.pk')
        self.sparse_encoder.save_encoder(path=sparse_encoder_path)
        sparse_weight_file = os.path.join(path, 'sparse_weight.pt')
        torch.save(self.sparse_weight, sparse_weight_file)
        logging.info("Sparse weight saved in {}".format(sparse_weight_file))

    def load_model(self, path, max_length=25, use_cuda=True):
        self.load_bert(path, max_length, use_cuda)
        self.load_sparse_encoder(path)
        self.load_sparse_weight(path)
        return self

    def load_bert(self, path, max_length, use_cuda):
        self.use_cuda = use_cuda
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(path, max_length=max_length)
        config = BertConfig.from_json_file(os.path.join(path, "config.json"))
        self.encoder = BertModel.from_pretrained(path, config=config) # dense encoder
        if use_cuda:
            self.encoder = self.encoder.cuda()
        return self.encoder, self.tokenizer

    def load_sparse_encoder(self, path):
        self.sparse_encoder = SparseEncoder().load_encoder(path=os.path.join(path, 'sparse_encoder.pk'))
        return self.sparse_encoder

    def load_sparse_weight(self, path):
        sparse_weight_file = os.path.join(path, 'sparse_weight.pt')
        self.sparse_weight = torch.load(sparse_weight_file)
        return self.sparse_weight

    def get_score_matrix(self, query_embeds, dict_embeds, is_sparse=False):
        score_matrix = np.matmul(query_embeds, dict_embeds.T)
        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):

        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix, -topk)[:, -topk:]
        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)
        return topk_idxs

    def embed_sparse(self, names, show_progress=False):
        batch_size = 1024
        sparse_embeds = []

        if show_progress:
            iterations = tqdm(range(0, len(names), batch_size))
        else:
            iterations = range(0, len(names), batch_size)

        for start in iterations:
            end = min(start + batch_size, len(names))
            batch = names[start:end]
            batch_sparse_embeds = self.sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def embed_dense(self, names, show_progress=False):
        self.encoder.eval() # prevent dropout
        batch_size = 1024
        dense_embeds = []

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(range(0, len(names), batch_size))
            else:
                iterations = range(0, len(names), batch_size)

            for start in iterations:
                end = min(start + batch_size, len(names))
                batch = [str(_) for _ in names[start:end]]
                batch_tokenized_names = self.tokenizer.batch_encode_plus(batch, padding=True, return_tensors="pt")
                input_ids = batch_tokenized_names["input_ids"].cuda() if self.use_cuda else batch_tokenized_names["input_ids"]
                attention_mask = batch_tokenized_names["attention_mask"].cuda() if self.use_cuda else batch_tokenized_names["attention_mask"]
                token_type_ids = batch_tokenized_names["token_type_ids"].cuda() if self.use_cuda else batch_tokenized_names["token_type_ids"]
                batch_dense_embeds = self.encoder(input_ids, attention_mask, token_type_ids)
                batch_dense_embeds = batch_dense_embeds[0][:, 0].cpu().detach().numpy()
                dense_embeds.append(batch_dense_embeds) # 4*768
        dense_embeds = np.concatenate(dense_embeds, axis=0)

        return dense_embeds