import os 
import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner

def prepare_sequence(path_to_file:str):
    """
    Prepare sequence strings of a given file, for langauge model by adding spaces

    parameters
    ---------

    path_to_file: `str`
    Path to file with sequences to be turned to embeddings


    return
    ------
    List of sequences with correct format
    """

    data = pd.read_csv(path_to_file)
    sequences = data["sequence"]
    print(sequences)
    sequences = sequences.apply(add_space)
    return sequences

def add_space(row):
    if not isinstance(row, float):
        row = " ".join(row)
    return row

def get_pseudo_likelihood(probs, sequences):
    probs_all = []
    for i,sequence_probs in enumerate(probs):
        wt_probs_full = []
        for pos in range(sequence_probs.shape[0]):

            wt_j = sequences[i][pos]
            #Can comment if PLM gives probabilities also for gaps
            if wt_j == "-" or wt_j == "*":
                continue
            wt_prob = sequence_probs.iloc[pos,:][wt_j]
            wt_probs_full.append(np.log(wt_prob))
        probs_all.append(np.average(wt_probs_full))
    return probs_all


