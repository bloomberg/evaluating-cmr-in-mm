import torch
import os

if torch.cuda.is_available():
    PROJECT_PATH = "/notebooks/evaluating-cmr-in-mm/"
else:
    PROJECT_PATH = "/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-mm/"
import sys

sys.path.append(PROJECT_PATH)
import torch
import argparse
from typing import List
import pickle
from tqdm import tqdm
import pandas as pd

ordered_perturbations = [
    "char_swap",
    "missing_char",
    "extra_char",
    "nearby_char",
    "probability_based_letter_change",
    "synonym_noun",
    "synonym_adj",
    "distraction_true",
    "distraction_false",
    "shuffle_nouns_and_adj",
    "shuffle_all_words",
    "shuffle_allbut_nouns_and_adj",
    "shuffle_within_trigrams",
    "shuffle_trigrams",
]

rsum_types = ["rsum-decreased", "rsum-increased", "rsum-unchanged"]

models = ["align", "altclip", "clip", "groupvit"]


def jaccard_sim(a: List, b: List, rounding_factor=4):
    def intersection(a: List, b: List):
        return list(set(a) & set(b))

    def union(a: List, b: List):
        return set(a).union(set(b))

    return round(len(intersection(a, b)) / len(union(a, b)), rounding_factor)


def main(args):
    print(args)

    root = args.root
    dataset = args.dataset
    task = args.task

    exisisting_dataframe_paths = []
    for model in models:
        for perturbation in ordered_perturbations:
            for rsum_diff_type in rsum_types:
                filename = f"{perturbation}-{rsum_diff_type}.pkl"
                dataframe_path = os.path.join(
                    root, dataset, model, task, "splits", filename
                )
                if os.path.exists(dataframe_path):
                    exisisting_dataframe_paths.append(dataframe_path)
    print("Collected perturbations paths")

    dataset_name = []
    model_i_names = []
    model_j_names = []
    perturbations_i = []
    perturbations_j = []
    jaccard_similarities = []
    for dataframe_path_i in tqdm(exisisting_dataframe_paths):
        for dataframe_path_j in tqdm(exisisting_dataframe_paths):
            try:
                with open(dataframe_path_i, "rb") as f_i:
                    df_i = pickle.load(f_i)
            except:
                print("Problem with ", dataframe_path_i)

            try:
                with open(dataframe_path_j, "rb") as f_j:
                    df_j = pickle.load(f_j)
            except:
                print("Problem with ", dataframe_path_j)

            if not df_i.empty and not df_j.empty:
                perturbation_name_i = dataframe_path_i.split("/")[-1].split(".")[0]
                perturbation_name_j = dataframe_path_j.split("/")[-1].split(".")[0]

                queries_i = df_i["t2i_queries_none"].tolist()
                queries_j = df_j["t2i_queries_none"].tolist()

                # print(f'{jaccard_similarity}\t:{perturbation_name_i} vs. {perturbation_name_j} ')
                assert (
                    dataframe_path_i.split("/")[-5] == dataframe_path_i.split("/")[-5]
                )
                dataset = dataframe_path_i.split("/")[-5]

                model_i_name = dataframe_path_i.split("/")[-4]
                model_j_name = dataframe_path_j.split("/")[-4]

                dataset_name.append(dataset)
                model_i_names.append(model_i_name)
                model_j_names.append(model_j_name)
                perturbations_i.append(perturbation_name_i)
                perturbations_j.append(perturbation_name_j)

                try:
                    jaccard_similarity = jaccard_sim(queries_i, queries_j)
                except:
                    print(f"Problem with files: {dataframe_path_i}, {dataframe_path_j}")
                    jaccard_similarities.append("none")
                jaccard_similarities.append(jaccard_similarity)

    data = {
        "dataset_name": dataset_name,
        "model_i_name": model_i_names,
        "model_j_name": model_j_names,
        "perturbation_i": perturbations_i,
        "perturbation_j": perturbations_j,
        "jac_sim": jaccard_similarities,
    }

    jac_sim_results = pd.DataFrame(data=data)
    print("Df size: ", jac_sim_results.shape)

    file_path = os.path.join(root, "jac_sim", f"{dataset}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(obj=jac_sim_results, file=f)
    print("Saved the jaccard sim scores to ", file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-mm/results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "f30k", "f30k_aug", "coco_aug"],
        help="dataset: coco, f30k",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2i",
        choices=["t2i", "i2t"],
    )
    args = parser.parse_args()
    main(args)
