from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import random

class Retriever:
    
    def normalize(self, emb):
        return emb/np.linalg.norm(emb, axis=-1,keepdims=True)
    
    def __init__(self, emb_dict):
        
        # to query faster, stack all embeddings and record keys
        self.emb_keys = list(emb_dict.keys())
        emb_dim = emb_dict[self.emb_keys[0]].shape[-1]
    
        self.emb_values = np.zeros((len(self.emb_keys), emb_dim))
        for i, k in enumerate(self.emb_keys):
            self.emb_values[i] = emb_dict[k]
        
        # normalize for cosine distance (kdtree only support euclidean when p=2)
        self.emb_values = self.normalize(self.emb_values)
        self.kdtree = KDTree(self.emb_values)
        
    def topk_nearest_dialogs(self, query_emb, k=5):
        query_emb = self.normalize(query_emb)
        if k == 1:
            return [self.emb_keys[i] for i in self.kdtree.query(query_emb, k=k, p=2)[1]]
        return [self.emb_keys[i] for i in self.kdtree.query(query_emb, k=k,p=2)[1][0]]
    
    def topk_nearest_distinct_dialogs(self, query_emb, k=5):
        return self.topk_nearest_dialogs(query_emb, k=k)
    
    def random_retrieve(self,k=5):
        return random.sample(self.emb_keys,k)

    

class IndexRetriever:
    
    # sample selection
    def random_sample_selection_by_turn(self, embs, ratio=0.1):
        n_selected = int(ratio*len(embs))
        print(f"randomly select {ratio} of turns, i.e. {n_selected} turns")
        selected_keys = random.sample(list(embs),n_selected)
        return {k:v for k,v in embs.items() if k in selected_keys}
    
    def random_sample_selection_by_dialog(self, embs, ratio=0.1):
        dial_ids = set([turn_label.split('_')[0] for turn_label in embs.keys()])
        n_selected = int(len(dial_ids)*ratio)
        print(f"randomly select {ratio} of dialogs, i.e. {n_selected} dialogs")
        selected_dial_ids = random.sample(dial_ids, n_selected)
        return {k:v for k,v in embs.items() if k.split('_')[0] in selected_dial_ids}

    def pre_assigned_sample_selection(self, embs, examples):
        selected_dial_ids = set([dial['ID'] for dial in examples])
        return {k:v for k,v in embs.items() if k.split('_')[0] in selected_dial_ids}

    
    def __init__(self, datasets, embedding_filenames, search_index_filename, sampling_method="none", ratio=1.0):
        
        # data_items: list of datasets in this notebook. Please include datasets for both search and query
        # embedding_filenames: list of strings. embedding dictionary npy files. Should contain embeddings of the datasets. No need to be same
        # search_index:  string. a single npy filename, the embeddings of search candidates
        # sampling method: "random_by_turn", "random_by_dialog", "kmeans_cosine", "pre_assigned"
        # ratio: how much portion is selected
        
        self.data_items = []
        for dataset in datasets:
            self.data_items += dataset
        
        # save all embeddings and dial_id_turn_id in a dictionary
        self.all_embeddings = {}
        for fn in embedding_filenames:
            this_embs = np.load(fn, allow_pickle=True).item()
            for k,v in this_embs.items():
                self.all_embeddings[k] = v
        
        # load the search index embeddings
        self.search_embs = np.load(search_index_filename, allow_pickle=True).item()
        
        # sample selection of search index
        if sampling_method == "none":
            self.retriever = Retriever(self.search_embs)
        elif sampling_method == 'random_by_dialog':
            self.retriever = Retriever(self.random_sample_selection_by_dialog(self.search_embs, ratio=ratio))
        elif sampling_method == 'random_by_turn':
            self.retriever = Retriever(self.random_sample_selection_by_turn(self.search_embs, ratio=ratio))
        elif sampling_method == 'pre_assigned':
            self.retriever = Retriever(self.pre_assigned_sample_selection(self.search_embs, self.data_items))
        else:
            raise ValueError("selection method not supported")
        
    def data_item_to_embedding(self, data_item):
        ID = data_item['ID']
        turn = data_item['turn_id']
        label = f"{ID}_turn_{turn}"
        
        return self.all_embeddings[label]
    
    def label_to_data_item(self, label):
        ID, _, turn_id = label.split('_')
        turn_id = int(turn_id)
        
        for d in self.data_items:
            if d['ID'] == ID and d['turn_id'] == turn_id:
                return d
        raise ValueError(f"label {label} not found. check data items input")
    
    def item_to_nearest_examples(self, data_item, k=5):
        # the nearest neighbor is at the end
        return [self.label_to_data_item(l) 
                for l in self.retriever.topk_nearest_distinct_dialogs(
                    self.data_item_to_embedding(data_item), k=k)
               ][::-1]

    def label_to_nearest_labels(self, label, k=5):
        data_item = self.label_to_data_item(label)
        return [l for l in self.retriever.topk_nearest_distinct_dialogs(
                    self.data_item_to_embedding(data_item), k=k)
                ][::-1]
    
    def random_examples(self, data_item, k=5):
        return [self.label_to_data_item(l)
                for l in self.retriever.random_retrieve(k=k)
               ]
        