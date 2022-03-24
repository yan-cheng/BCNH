from string import punctuation
import subprocess
import re
import json
import networkx as nx


class Abbr_resolver():

    def __init__(self, ab3p_path):
        self.ab3p_path = ab3p_path
        
    def resolve(self, corpus_path):
        result = subprocess.run([self.ab3p_path, corpus_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        line = result.stdout.decode('utf-8')
        if "Path file for type cshset does not exist!" in line:
            raise Exception(line)
        elif "Cannot open" in line:
            raise Exception(line)
        elif "failed to open" in line:
            raise Exception(line)
        lines = line.split("\n")
        result = {}
        for line in lines:
            if len(line.split("|"))==3:
                # sf|lf|P-precision
                sf, lf, _ = line.split("|")
                sf = sf.strip()
                lf = lf.strip()
                result[sf] = lf
        
        return result


class TextPreprocess():
    """
    Text Preprocess module
    Support lowercase, removing punctuation, typo correction
    """
    def __init__(self, 
            lowercase=True, 
            remove_punctuation=True,
            ignore_punctuations="",
            typo_path=None):
        """
        Parameters
        ==========
        typo_path : str
            path of known typo dictionary, a txt file which each line is "false_typing||right_typing"
        """
        self.lowercase = lowercase
        self.typo_path = typo_path
        self.rmv_puncts = remove_punctuation
        self.punctuation = punctuation
        for ig_punc in ignore_punctuations:
            self.punctuation = self.punctuation.replace(ig_punc,"")
        self.rmv_puncts_regex = re.compile(r'[\s{}]+'.format(re.escape(self.punctuation)))
        
        if typo_path:
            self.typo2correction = self.load_typo2correction(typo_path)
        else:
            self.typo2correction = {}

    def load_typo2correction(self, typo_path):
        typo2correction = {}
        with open(typo_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                s = line.strip()
                tokens = s.split("||")
                value = "" if len(tokens) == 1 else tokens[1]
                typo2correction[tokens[0]] = value    

        return typo2correction 

    def remove_punctuation(self,phrase):
        phrase = self.rmv_puncts_regex.split(phrase)
        phrase = ' '.join(phrase).strip()

        return phrase

    def correct_spelling(self, phrase):
        phrase_tokens = phrase.split()
        phrase = ""

        for phrase_token in phrase_tokens:
            if phrase_token in self.typo2correction.keys():
                phrase_token = self.typo2correction[phrase_token]
            phrase += phrase_token + " "
       
        phrase = phrase.strip()
        return phrase

    def run(self, text):
        if self.lowercase:
            text = text.lower()

        if self.typo_path:
            text = self.correct_spelling(text)

        if self.rmv_puncts:
            text = self.remove_punctuation(text)

        text = text.strip()

        return text


class TaxonomyGraph:

    def __init__(self, file_path) -> None:
        self.atoms = {} # the dict of atoms mapping to ids
        self.network = nx.DiGraph()

        f = open(file_path, "r", encoding="utf-8")
        json_dict = json.load(f)
        f.close()
        for k, v in json_dict.items():
            if "Synonym" in v["Attribute"]:
                synonyms = v["Attribute"]["Synonym"]
                for syn in synonyms:
                    self.atoms[syn] = k
            node_attributes = {k: v["Attribute"]}
            self.network.add_node(k)
            nx.set_node_attributes(self.network, node_attributes)
            if "Relation" in v:
                for rel, node_ids in v["Relation"].items():
                    for node_id in node_ids:
                        if rel == "Hypernym":
                            self.network.add_edge(k, node_id, Name="Hypernym")
                            self.network.add_edge(node_id, k, Name="Hyponym")
                        else:
                            self.network.add_edge(k, node_id, Name=rel)

    def get_node_id(self, atom_str):
        """
        Return the node id for a string (atom).
        """
        if atom_str in self.atoms:
            return self.atoms[atom_str]
        else:
            raise ValueError("String '{}' is not in the vocab.".format(atom_str))

    def get_node_info_by_id(self, node_id):
        """
        Return the node's information.
        """
        if self.network.has_node(node_id):
            result = {'ID':node_id}
            result.update(self.network.nodes[node_id])
            return result
        else:
            raise ValueError("Node '{}' is not in the network.".format(node_id))

    def get_node_info_by_str(self, atom_str):
        node_id = self.get_node_id(atom_str)
        return self.get_node_info_by_id(node_id)

    def get_successors(self, node_id, relation=None):
        """
        Get the node's successors with the edge from node satisfing relation.

        Arguments:
            node_id: node id string.
            relation: the edge relation type between source and successors, when 
                      the relation is None, any successors will be counted. 
                      Relation can be ["Hypernym", "Hyponym"].
        Returns:
           list(node_id) the successors list.
        """
        result = []
        if not self.network.has_node(node_id):
            return None
        neighbors = self.network.successors(node_id)
        if relation is None:
            return [_ for _ in neighbors]
        for neighbor_id in neighbors:
            if len(neighbor_id) <= 4:
                # print(neighbor_id)
                continue # ids maybe 'C' which stands for Diseases category rather a valid ID
            if relation == self.network[node_id][neighbor_id]["Name"]:
                result.append(neighbor_id)
        return result

    def get_remote_successors(self, node_id, relation=None, hop=5):
        """
        Get the node's remote successors with the edges from node satisfing relation.

        Arguments:
            node_id: node id string.
            relation: the edge relation type between source and remote successors, when 
                      the relation is None, any remote successors will be counted.
                      Relation can be ["Hypernym", "Hyponym"].
            hop: the max hop between node and its remote successors.
        Returns:
           [list(1_hop_node_id), list(2_hop_node_id), ...] the remote successors list.
        """
        # relation in [None, "Hypernym", "Hyponym"]
        result = [[node_id]]
        visited = set(result[-1])
        for _ in range(hop):
            tmp = []
            for id_ in result[-1]:
                neighbors_ = self.get_successors(id_, relation)
                for _ in neighbors_:
                    if _ not in visited:
                        tmp.append(_)
            result.append(tmp)
            for _ in tmp:
                visited.add(_)
        result = [i for i in result[1:] if i]
        return result

    def find_path(self, source_id, target_id, relation=None, cutoff=5):
        """
        Find the path between source and target node.

        Arguments:
            source_id: source node id string.
            target_id: target node id string.
            relation: the edge relation type between source and target, when 
                      the relation is None, any directed edge will take effect.
            cutoff: the max steps to stop searching the path.
        
        Returns:
           (length, list(node_id)) the path's length and path.
        """
        def fun(x, y, edge_attr):
            if edge_attr['Name'] == relation or relation is None:
                return 1
            else:
                return None
        try:
            length, path = nx.single_source_dijkstra(self.network, source_id,
                target_id, cutoff=cutoff, weight=fun)
            return length, path
        except KeyError:
            return 0, []
