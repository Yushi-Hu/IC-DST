from sentence_transformers.evaluation import SentenceEvaluator
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score
import numpy as np
from typing import List
from embed_based_retriever import EmbeddingRetriever
import json
from retriever_evaluation import evaluate_retriever_on_dataset

logger = logging.getLogger(__name__)

class RetrievalEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.
    The results are written in a CSV. If a CSV already exists, then values are appended.
    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.
    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    """

    def __init__(self, TRAIN_FN, index_set, name: str = '', batch_size: int = 32, show_progress_bar: bool = False, write_csv: bool = True):
        self.train_fn = TRAIN_FN
        self.index_set = index_set
        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "retrieval_evaluation" + ("_"+name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps",
                            "turnsv_score", "turns_score","allsv_score", "alls_score"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Evaluation of the model on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model)


        #Main score is the max of Average Precision (AP)
        main_score = scores['turnsv']['score'] + scores['turns']['score']

        file_output_data = [epoch, steps]

        for header_name in self.csv_headers:
            if '_' in header_name:
                sim_fct, metric = header_name.split("_", maxsplit=1)
                file_output_data.append(scores[sim_fct][metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        return main_score


    def compute_metrices(self, model):

        def store_embed(dataset, output_filename):
            embeddings = model.encode(dataset.turn_utts, convert_to_numpy=True)
            output = {}
            for i in range(len(embeddings)):
                output[dataset.turn_labels[i]] = embeddings[i:i+1]
            np.save(output_filename, output)
            return
        
        store_embed(self.index_set, "train_index.npy")

        # retriever to evaluation
        with open(self.train_fn) as f:
            train_set = json.load(f)
        with open("../../data/mw24_100p_dev.json") as f:
            test_set_21 = json.load(f)

        retriever = EmbeddingRetriever(datasets=[train_set],
                                    model_path="",
                                    model=model,
                                    search_index_filename="train_index.npy",
                                    sampling_method="pre_assigned")

        turn_sv, turn_s, dial_sv, dial_s = evaluate_retriever_on_dataset(test_set_21,retriever)

        output_scores = {'turnsv': {"score": turn_sv}, "turns": {"score": turn_s}, 
                            "allsv": {"score": dial_sv}, "alls": {"score": dial_s}}


        return output_scores


class RetrievalEvaluatorAll(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.
    The results are written in a CSV. If a CSV already exists, then values are appended.
    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.
    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    """

    def __init__(self, TRAIN_FN, index_set, name: str = '', batch_size: int = 32, show_progress_bar: bool = False, write_csv: bool = True):
        self.train_fn = TRAIN_FN
        self.index_set = index_set
        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel(
            ) == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "retrieval_evaluation" + \
            ("_"+name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps",
                            "turnsv_score", "turns_score", "allsv_score", "alls_score"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Evaluation of the model on " +
                    self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model)

        #Main score is the max of Average Precision (AP)
        main_score = scores['allsv']['score'] + scores['alls']['score']

        file_output_data = [epoch, steps]

        for header_name in self.csv_headers:
            if '_' in header_name:
                sim_fct, metric = header_name.split("_", maxsplit=1)
                file_output_data.append(scores[sim_fct][metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        return main_score

    def compute_metrices(self, model):

        def store_embed(dataset, output_filename):
            embeddings = model.encode(dataset.turn_utts, convert_to_numpy=True)
            output = {}
            for i in range(len(embeddings)):
                output[dataset.turn_labels[i]] = embeddings[i:i+1]
            np.save(output_filename, output)
            return

        store_embed(self.index_set, "train_index.npy")

        # retriever to evaluation
        with open(self.train_fn) as f:
            train_set = json.load(f)
        with open("../../data/mw24_100p_dev.json") as f:
            test_set_21 = json.load(f)

        retriever = EmbeddingRetriever(datasets=[train_set],
                                       model_path="",
                                       model=model,
                                       search_index_filename="train_index.npy",
                                       sampling_method="pre_assigned")

        turn_sv, turn_s, dial_sv, dial_s = evaluate_retriever_on_dataset(
            test_set_21, retriever)

        output_scores = {'turnsv': {"score": turn_sv}, "turns": {"score": turn_s},
                         "allsv": {"score": dial_sv}, "alls": {"score": dial_s}}

        return output_scores
