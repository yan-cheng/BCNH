import pickle
import os
import glob
import logging
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset
from utils import TaxonomyGraph


LOGGER = logging.getLogger(__name__)


class QueryDataset(Dataset):

    def __init__(self, data_dir, 
            filter_composite=False,
            filter_duplicate=False
        ):
        """     
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_composite={} filter_duplicate={}".format(
            data_dir, filter_composite, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_composite=filter_composite,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_composite, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        for concept_file in tqdm(concept_files):
            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()

            for concept in concepts:
                concept = concept.split("||")
                mention = concept[3].strip()
                cui = concept[4].strip()
                is_composite = (cui.replace("+","|").count("|") > 0)

                if filter_composite and is_composite:
                    continue
                else:
                    data.append((mention,cui))
        
        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data)
        
        return data


class DictionaryDataset():
    """
    A class used to load dictionary data
    """
    def __init__(self, dictionary_path):
        """
        Parameters
        ----------
        dictionary_path : str
            The path of the dictionary
        draft : bool
            use only small subset
        """
        LOGGER.info("DictionaryDataset! dictionary_path={}".format(
            dictionary_path 
        ))
        self.data = self.load_data(dictionary_path)
        
    def load_data(self, dictionary_path):
        data = []
        with open(dictionary_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line == "": continue
                cui, name = line.split("||")
                data.append((name,cui))
        
        data = np.array(data)
        return data


class CandidateDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, queries, dicts, tokenizer, topk, hyper_num, d_ratio, s_score_matrix, s_candidate_idxs, taxonomy):

        LOGGER.info("CandidateDataset! len(queries)={} len(dicts)={} topk={} hyper_num={} d_ratio={}".format(
            len(queries),len(dicts), topk, hyper_num, d_ratio))

        self.query_names, self.query_ids = [row[0] for row in queries], [row[1] for row in queries]
        self.dict_names, self.dict_ids = [row[0] for row in dicts], [row[1] for row in dicts]
        self.dict_ids_np = np.array(self.dict_ids)
        self.topk = topk
        self.n_dense = int(topk * d_ratio)
        self.n_sparse = topk - self.n_dense
        self.tokenizer = tokenizer
        self.s_score_matrix = s_score_matrix
        self.s_candidate_idxs = s_candidate_idxs
        self.d_candidate_idxs = None

        self.hyper_num = hyper_num if hyper_num else 0
        self.taxonomy = TaxonomyGraph(taxonomy)

        self.query_vocabs = []
        self.process_query_items(taxonomy)

    def set_dense_candidate_idxs(self, d_candidate_idxs):
        self.d_candidate_idxs = d_candidate_idxs

    def process_query_items(self, taxonomy_path):
        dump_file = taxonomy_path.rstrip(".json") + ".pk"
        if os.path.exists(dump_file):
            with open(dump_file, "rb") as f:
                self.query_vocabs = pickle.load(f)
        else:
            for query_idx in tqdm(range(len(self.query_ids)), desc="Process query infos..."):
                query_id = self.query_ids[query_idx]
                query_ids = query_id.split("|")
                synonym_ids = set(query_ids)
                hypernym_ids = set()
                hyponym_ids = set()
                for query_id in query_ids:
                    hypernym_ids = hypernym_ids.union(
                        set(self.taxonomy.get_successors(query_id, relation="Hypernym")))
                    hyponym_ids = hyponym_ids.union(
                        set(self.taxonomy.get_successors(query_id, relation="Hyponym")))
                
                vocabs = [[], [], []]
                vocab_ids_list = [list(synonym_ids), list(hypernym_ids), list(hyponym_ids)]

                for _, vocab_ids in enumerate(vocab_ids_list):
                    for vocab_id in vocab_ids:
                        indices = [idx for idx, dict_id in enumerate(self.dict_ids) if vocab_id in dict_id]
                        for i in indices:
                            name = self.dict_names[i]
                            sparse_score = float(self.s_score_matrix[query_idx][i])
                            data_item = (i, vocab_id, name, sparse_score)
                            if data_item not in vocabs[_]:
                                vocabs[_].append(data_item)
                
                self.query_vocabs.append(vocabs)
            
            with open(dump_file, "wb+") as f:
                pickle.dump(self.query_vocabs, f)

    def __getitem__(self, query_idx):
        assert (self.s_candidate_idxs is not None)
        assert (self.s_score_matrix is not None)
        assert (self.d_candidate_idxs is not None)

        query_name = self.query_names[query_idx]
        query_token = self.tokenizer.encode(
            query_name, 
            truncation=True, 
            max_length=self.tokenizer.init_kwargs["max_length"],
            pad_to_max_length=True)

        # combine sparse and dense candidates as many as top-k
        s_candidate_idx = self.s_candidate_idxs[query_idx]
        d_candidate_idx = self.d_candidate_idxs[query_idx]
        
        # fill with sparse candidates first
        topk_candidate_idx = s_candidate_idx[:self.n_sparse]
        
        # fill remaining candidates with dense
        for d_idx in d_candidate_idx:
            if len(topk_candidate_idx) >= self.topk:
                break
            if d_idx not in topk_candidate_idx:
                topk_candidate_idx = np.append(topk_candidate_idx, d_idx)

        # sanity check
        assert len(topk_candidate_idx) == self.topk
        assert len(topk_candidate_idx) == len(set(topk_candidate_idx))

        candidate_names = [self.dict_names[candidate_idx] for candidate_idx in topk_candidate_idx]
        candidate_s_scores = self.s_score_matrix[query_idx][topk_candidate_idx]
        query_token = np.array(query_token).squeeze()
        
        candidate_tokens = self.tokenizer.batch_encode_plus(
            candidate_names, truncation=True,
            max_length=self.tokenizer.init_kwargs["max_length"], 
            pad_to_max_length=True)["input_ids"]
        candidate_tokens = np.array(candidate_tokens)
        
        query_vocab = self.query_vocabs[query_idx]
        synonyms, hypernyms, hyponyms = query_vocab
        # process hypernyms
        hyper_labels = [1 for _ in range(len(hypernyms))]
        random.shuffle(hypernyms)

        while len(hypernyms) < self.hyper_num:
            random_idx = random.randint(0, len(self.dict_ids)-1)
            fake_hyper_cui = self.dict_ids[random_idx].split("|")[0]
            fake_hyper_name = str(self.dict_names[random_idx])
            fake_hyper_s_scores = self.s_score_matrix[query_idx][random_idx]
            hypernyms.append((random_idx, fake_hyper_cui, fake_hyper_name, fake_hyper_s_scores))
            hyper_labels.append(0)
        hypernyms = hypernyms[:self.hyper_num]
        hyper_labels = hyper_labels[:self.hyper_num]
        hyper_names = np.array([_[2] for _ in hypernyms])
        hypernym_tokens = np.array(self.tokenizer.batch_encode_plus(
                                        hyper_names, truncation=True,
                                        max_length=self.tokenizer.init_kwargs["max_length"], 
                                        pad_to_max_length=True)["input_ids"])
        hyper_labels = np.array(hyper_labels).astype(np.float32)
        candidate_labels = self.check_candidate_labels(
            query_idx, topk_candidate_idx, syn_lable=2, hyper_label=1).astype(np.float32)
        return (query_token, candidate_tokens, candidate_s_scores, hypernym_tokens), (candidate_labels, hyper_labels)

    def __len__(self):
        return len(self.query_names)

    def check_candidate_labels(self, query_idx, candidate_indices, syn_lable=1, hyper_label=0, hypo_label=0, neg_label=0):
        candidate_ids = [self.dict_ids[_].split("|") for _ in candidate_indices]
        synonyms, hypernyms, hyponyms = self.query_vocabs[query_idx]
        synonym_ids = set([_[1] for _ in synonyms])
        hypernym_ids = set([_[1] for _ in hypernyms])
        hyponym_ids = set([_[1] for _ in hyponyms])
        labels = []
        for candidate_id in candidate_ids:
            intersection_syn = synonym_ids.intersection(set(candidate_id))
            intersection_hyper = hypernym_ids.intersection(set(candidate_id))
            intersection_hypo = hyponym_ids.intersection(set(candidate_id))
            if len(intersection_syn) > 0:
                labels.append(syn_lable)
            elif len(intersection_hyper) > 0:
                labels.append(hyper_label)
            elif len(intersection_hypo) > 0:
                labels.append(hypo_label)
            else:
                labels.append(neg_label)
        return np.array(labels)
