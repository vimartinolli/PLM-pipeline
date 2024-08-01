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
parser.add_argument('--calc_list', help="Example: ['pseudolikelihood' 'probability_matrix' 'embeddings']")

args = parser.parse_args()

sequences_column = args.sequences_column
model_name = args.model_name
file_path = args.file_path
save_path = args.output_folder
calc_list = args.calc_list

#### Read input file 
sequence_file  = pd.read_csv(file_path)
if "sequence_id" not in sequence_file.columns:
    print("Column 'sequence_id' not found in input CSV. Make sure your input file contains unique sequence identifiers.")
    sys.exit()

#### Create if output folder (if necessary)
if not os.path.exists(save_path):
    os.mkdir(save_path)

#### Initialize the model
if model_name == "Ablang":   
    model_hc = Ablang(chain="heavy")
    model_lc = Ablang(chain="light")
elif model_name == "Sapiens":
    model_hc = Sapiens(chain_type="H")
    model_lc = Sapiens(chain_type="L")
elif model_name == "ESM":
    model = ESM()
elif model_name == "ProtBert":
    model = ProtBert()
else:
    print("model_name is unknown.")
    sys.exit()

#### Ablang and Sapiens have different models for heavy and light chains
if model_name in ["Ablang","Sapiens"]:
    if "chain" not in sequence_file.columns:
        print("Column 'chain' not found in input CSV. When running Ablang or Sapiens, a column named 'chain' should be present to indicate heavy or light chain.")
        sys.exit()
   
    #### Perform calculations
    if "pseudolikelihood" in calc_list:
        #Calculate pseudolikelihood, add to sequence_file, and save as CSV
        sequence_file["evo_likelihood"] = "dummy"
        is_heavy_chain = list(sequence_file["chain"] == "IGH")
        is_light_chain = list(sequence_file["chain"] != "IGH")
        sequence_file.loc[is_heavy_chain,"evo_likelihood"] = model_hc.calc_pseudo_likelihood_sequence(list(sequence_file[is_heavy_chain][sequences_column]))
        sequence_file.loc[is_light_chain,"evo_likelihood"] = model_lc.calc_pseudo_likelihood_sequence(list(sequence_file[is_light_chain][sequences_column]))
        sequence_file.to_csv(os.path.join(save_path,f"evo_likelihood_{model_name}.csv"), index=False)

    if "probability_matrix" in calc_list:
        #For each sequence, calculate the probability matrix per position and save as CSV
        for index in sequence_file.index:
            if sequence_file["chain"][index] == "IGH":
                model = model_hc
            elif sequence_file["chain"] != "IGH":
                model = model_lc
            prob_matrix = model.calc_probability_matrix(sequence_file[sequences_column][index])
            prob_matrix.to_csv(os.path.join(save_path,f"prob_matrix_seq{sequence_file["sequence_id"][index]}_{model_name}.csv"), index = False)

    if "embeddings" in calc_list:
        #Calculate embeddings, add to sequence_file, and save as CSV
        sequence_file_hc = sequence_file[sequence_file["chain"] == "IGH"]
        sequence_file_lc = sequence_file[sequence_file["chain"] == "IGl"]
        embeds_hc = model_hc.fit_transform(sequences=list(sequence_file_hc[sequences_column]))
        embeds_lc = model_lc.fit_transform(sequences=list(sequence_file_lc[sequences_column]))
        embeds_hc = pd.concat([sequence_file_hc,embeds_hc],axis=1)
        embeds_lc = pd.concat([sequence_file_lc,embeds_lc],axis=1)
        embeds = pd.concat([embeds_hc, embeds_lc], ignore_index=True)  
        embeds.to_csv(os.path.join(save_path,f"embeddings_{model_name}.csv"), index=False) 

else: #If model is not Ablang or Sapiens:
    #### Perform calculations
    if "pseudolikelihood" in calc_list:
        #Calculate pseudolikelihood, add to sequence_file, and save as CSV
        sequence_file["evo_likelihood"] = model.calc_pseudo_likelihood_sequence(list(sequence_file[sequences_column]))
        sequence_file.to_csv(os.path.join(save_path,f"evo_likelihood_{model_name}.csv"), index=False)
    
    if "probability_matrix" in calc_list:
        #For each sequence, calculate the probability matrix per position and save as CSV
        for index in sequence_file.index:
            prob_matrix = model.calc_probability_matrix(sequence_file[sequences_column][index])
            prob_matrix.to_csv(os.path.join(save_path,f"prob_matrix_seq{sequence_file["sequence_id"][index]}_{model_name}.csv"), index = False)
    
    if "embeddings" in calc_list:
        #Calculate embeddings, add to sequence_file, and save as CSV
        embeds = model.fit_transform(sequences=list(sequence_file[sequences_column]))
        embeds = pd.concat([sequence_file,embeds],axis=1)  
        embeds.to_csv(os.path.join(save_path,f"embeddings_{model_name}.csv"), index=False) 
        


   