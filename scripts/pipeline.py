import pandas as pd
import numpy as np
import os
import sys
import argparse

sys.path.append("../src")

from ablang_model import Ablang
from ESM_model import ESM
from sapiens_model import Sapiens
from protbert import ProtBert

#### handle command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', help="Choose from: Ablang,ProtBert,Sapiens,ESM") 
parser.add_argument('--file_path')
parser.add_argument('--sequences_column')
parser.add_argument('--output_folder')

args = parser.parse_args()

sequences_column = args.sequences_column
model_name = args.model_name
file_path = args.file_path
save_path = args.output_folder

#### Read input file 
sequence_file  = pd.read_csv(file_path)

#### Create if output folder (if necessary)
if not os.path.exists(save_path):
    os.mkdir(save_path)

#### Initialize the model and perform calculations, if Ablang or Sapiens: select heavy of light chain
### the class method "calc_pseudo_likelihood_sequence" calculates the evolike of each sequence
if model_name in ["Ablang","Sapiens"]:
    sequence_file["evo_likelihood"] = "dummy"
    if "chain" in sequence_file.columns:
        is_heavy_chain = list(sequence_file["chain"] == "IGH")
        is_light_chain = list(sequence_file["chain"] != "IGH")

        if model_name == "Ablang":
            sequence_file.loc[is_heavy_chain,"evo_likelihood"] = Ablang(chain="heavy").calc_pseudo_likelihood_sequence(list(sequence_file[is_heavy_chain][sequences_column]))
            sequence_file.loc[is_light_chain,"evo_likelihood"] = Ablang(chain="light").calc_pseudo_likelihood_sequence(list(sequence_file[is_light_chain][sequences_column]))
        elif model_name == "Sapiens":
            sequence_file.loc[is_heavy_chain,"evo_likelihood"] = Sapiens(chain_type="H").calc_pseudo_likelihood_sequence(list(sequence_file[is_heavy_chain][sequences_column]))
            sequence_file.loc[is_light_chain,"evo_likelihood"] = Sapiens(chain_type="L").calc_pseudo_likelihood_sequence(list(sequence_file[is_light_chain][sequences_column]))
    else:
        print("Column 'chain' not found in input CSV. When running Ablang or Sapiens, a column named 'chain' should be present to indicate heavy or light chain.")
else:
    model = model()
    sequence_file["evo_likelihood"] = model.calc_pseudo_likelihood_sequence(list(sequence_file[sequences_column]))

#### Save output CSV
sequence_file.to_csv(os.path.join(save_path,f"evo_likelihood_{model_name}.csv"), index=False)   