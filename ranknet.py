import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tokenizer import BertTokenizer
from bert import BertModel


LOGGER = logging.getLogger(__name__)


class SparseEncoder(object):
    def __init__(self, use_cuda=False):
        self.encoder = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
        self.use_cuda = use_cuda

    def fit(self, train_corpus):
        self.encoder.fit(train_corpus)
        return self

    def transform(self, mentions):
        vec = self.encoder.transform(mentions).toarray()
        vec = torch.FloatTensor(vec) # return torch float tensor
        if self.use_cuda:
            vec = vec.cuda()
        return vec

    def cuda(self):
        self.use_cuda = True

        return self

    def cpu(self):
        self.use_cuda = False
        return self

    def __call__(self, mentions):
        return self.transform(mentions)

    def vocab(self):
        return self.encoder.vocabulary_

    def save_encoder(self, path):
        with open(path, 'wb') as fout:
            pickle.dump(self.encoder, fout)
            logging.info("Sparse encoder saved in {}".format(path))

    def load_encoder(self, path):
        with open(path, 'rb') as fin:
            self.encoder =  pickle.load(fin)
            logging.info("Sparse encoder loaded from {}".format(path))
        return self


class RankNet(nn.Module):

    def __init__(self, encoder, learning_rate, weight_decay, sparse_weight, hyper_norm_scale, use_cuda):
        LOGGER.info("RerankNet! learning_rate={} weight_decay={} sparse_weight={} use_cuda={}".format(
            learning_rate, weight_decay, sparse_weight, use_cuda
        ))
        super(RankNet, self).__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sparse_weight = sparse_weight
        self.hyper_norm_scale = hyper_norm_scale
        self.use_cuda = use_cuda
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
            {'params' : self.sparse_weight, 'lr': 0.01, 'weight_decay': 0}],
            lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def forward(self, x):
        query_token, candidate_tokens, candidate_s_scores, hypernym_tokens = x
        if self.use_cuda:
            hypernym_tokens = hypernym_tokens.cuda()
        # topk candidate scores
        batch_size, topk, _ = candidate_tokens.shape
        if self.use_cuda:
            query_token = query_token.cuda()
            candidate_tokens = candidate_tokens.cuda()
            candidate_s_scores = candidate_s_scores.cuda()
        query_embed = self.encoder(query_token)[0][:, 0].unsqueeze(dim=1)                          # [batch_size, 1, hidden]
        candidate_tokens = self.reshape_candidates_for_encoder(candidate_tokens)                   # [batch_size*topk, max_length]
        candidate_embeds = self.encoder(candidate_tokens)[0][:, 0].view(batch_size, topk, -1)      # [batch_size, topk, hidden]
        candidate_d_scores = torch.bmm(query_embed, candidate_embeds.permute(0, 2, 1)).squeeze(1)
        candidate_scores = self.sparse_weight * candidate_s_scores + candidate_d_scores
        # query-hypernym norm
        batch_size, hyper_num, _ = hypernym_tokens.shape
        hypernym_tokens = self.reshape_candidates_for_encoder(hypernym_tokens)                     # [batch_size*hyper_num, max_length]
        hypernym_embeds = self.encoder(hypernym_tokens)[0][:, 0].view(batch_size, hyper_num, -1)   # [batch_size, hyper_num, hidden]
        query_norm = torch.norm(query_embed, dim=-1).repeat(1, hyper_num)                          # [batch_size, hyper_num]
        hypernym_norm = torch.norm(hypernym_embeds, dim=-1)                                        # [batch_size, hyper_num]
        norm_dist = (query_norm - hypernym_norm) / (query_norm + hypernym_norm)                    # [batch_size, hyper_num]
        return candidate_scores, norm_dist

    def loss(self, outputs, targets):
        candidate_scores, norm_dist = outputs
        candidate_label, hyper_label = targets
        if self.use_cuda:
            candidate_label = candidate_label.cuda()
            hyper_label = hyper_label.cuda()
        loss1 = self.cross_entropy(candidate_scores, candidate_label)
        loss2 = self.norm(norm_dist, hyper_label)
        return loss1 + self.hyper_norm_scale*loss2

    def cross_entropy(self, scores, targets):
        targets_prob = F.softmax(targets, dim=-1)
        scores_log_prob = F.log_softmax(scores, dim=-1)
        loss = torch.mean(torch.sum(- targets_prob * scores_log_prob))
        return loss

    def norm(self, norm_dist, target):
        """
        norm_dist: norm dist of query-hypernyms, [batch_size, hyper_num]
        target: hypernym labels, [batch_size, hyper_num]
        """
        norm_dist = norm_dist * target                 # only select the hypernym pairs
        hyper_norm_loss = norm_dist[norm_dist != 0]    # filter sets with no zeros
        if len(hyper_norm_loss) == 0:
            hyper_norm_loss = hyper_norm_loss.sum()    # will return zero loss
        else:
            hyper_norm_loss = hyper_norm_loss.mean()   # will return mean loss
        return hyper_norm_loss

    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates
