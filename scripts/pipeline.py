#Import packages
import pandas as pd
import numpy as np
import os
import sys
import argparse

#Set path to src
sys.path.append("../src")

#Import extra functions
from utils import calculate_mutations

#Import PLMs
from ablang_model import Ablang
from ESM1b_model import ESM1b
from sapiens_model import Sapiens
from protbert import ProtBert

#### Handle command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', help="Choose from: Ablang,ProtBert,Sapiens,ESM1b") 
parser.add_argument('--file_path')
parser.add_argument('--sequences_column')
parser.add_argument('--sequence_id_column', default="sequence_id", help="Column name in the input file where sequence ID's are stored.")
parser.add_argument('--output_folder')
parser.add_argument('--calc_list', nargs="*", help="Example: pseudolikelihood, probability_matrix, suggest_mutations, embeddings")
parser.add_argument('--number_mutations', help="Choose the number of mutations you want the model to suggest (Default is 1)")
parser.add_argument('--cache_dir', default = "default", help="Potential cache directory for ESM1b pretrained model.")

args = parser.parse_args()

sequences_column = args.sequences_column
model_name = args.model_name
file_path = args.file_path
save_path = args.output_folder
seq_id_column = args.sequence_id_column
cache_dir = args.cache_dir
number_mutations = int(args.number_mutations) if args.number_mutations else 1 #If number_mutations is not supplied, set to 1
calc_list = args.calc_list
if("suggest_mutations" in calc_list):
    calc_list.append("probability_matrix") #If "suggest_mutations" is in calc_list, also calculate probability matrices"
####

#### Read input file 
sequence_file  = pd.read_csv(file_path)
if seq_id_column not in sequence_file.columns:
    print("Column " + seq_id_column + " not found in input CSV. Make sure your input file contains unique sequence identifiers.")
    sys.exit()

#### Create output folder (if necessary)
if not os.path.exists(save_path):
    os.mkdir(save_path)

#### Initialize the model
if model_name == "Ablang":   
    model_hc = Ablang(chain="heavy")
    model_lc = Ablang(chain="light")
elif model_name == "Sapiens":
    model_hc = Sapiens(chain_type="H")
    model_lc = Sapiens(chain_type="L")
elif model_name == "ESM1b":
    model = ESM1b(cache_dir=cache_dir)
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

    #Calculate pseudolikelihood, add to sequence_file, and save as CSV
    if "pseudolikelihood" in calc_list:
        sequence_file["evo_likelihood"] = "dummy"
        is_heavy_chain = list(sequence_file["chain"] == "IGH")
        is_light_chain = list(sequence_file["chain"] != "IGH")
        sequence_file.loc[is_heavy_chain,"evo_likelihood"] = model_hc.calc_pseudo_likelihood_sequence(list(sequence_file[is_heavy_chain][sequences_column]))
        sequence_file.loc[is_light_chain,"evo_likelihood"] = model_lc.calc_pseudo_likelihood_sequence(list(sequence_file[is_light_chain][sequences_column]))
        sequence_file.to_csv(os.path.join(save_path,f"evo_likelihood_{model_name}.csv"), index=False)

    #Calculate probability matrices and potentially suggested mutations
    if "probability_matrix" in calc_list:   
        # If "suggest_mutations" is in the calc_list, create a dataframe to store all the mutations of all sequences
        if("suggest_mutations" in calc_list):
            all_mutations_df = pd.DataFrame()

        # Calculate probability matrix (and mutations) for each sequence
        for index in sequence_file.index:
            # Use either the heavy- or light chain model
            if sequence_file["chain"][index] == "IGH":
                model = model_hc
            elif sequence_file["chain"][index] != "IGH":
                model = model_lc
           
            # Calculates and saves the probability matrix for each sequence
            prob_matrix = model.calc_probability_matrix(sequence_file[sequences_column][index])
            seq_id = sequence_file[seq_id_column][index]
            prob_matrix.to_csv(os.path.join(save_path,f"prob_matrix_seq_{seq_id}_{model_name}.csv"), index = False)

            # Calculate the suggested mutations for this sequence
            if("suggest_mutations" in calc_list):
                mutations_df = calculate_mutations(
                    sequences_file=sequence_file.loc[[index]],
                    prob_matrix=prob_matrix,
                    num_mutations=number_mutations,
                    seq_id_column=seq_id_column,
                    sequences_column=sequences_column
                    )
                # Concatenates the mutations obtained from the current sequence to the global DataFrame
                all_mutations_df = pd.concat([all_mutations_df, mutations_df], ignore_index=True)
            
        # Saves all the mutations for all sequences
        if("suggest_mutations" in calc_list):
            output_file = os.path.join(save_path, f"{model_name}_{number_mutations}_mutations.csv")
            all_mutations_df.to_csv(output_file, index=False)
            print(f"All mutations saved to: {output_file}")

    if "embeddings" in calc_list:
        #Calculate embeddings, add to sequence_file, and save as CSV
        sequence_file_hc = sequence_file[sequence_file["chain"] == "IGH"]
        sequence_file_lc = sequence_file[sequence_file["chain"] != "IGH"]
        embeds_hc = model_hc.fit_transform(sequences=list(sequence_file_hc[sequences_column]))
        embeds_lc = model_lc.fit_transform(sequences=list(sequence_file_lc[sequences_column]))
        embeds_hc = pd.concat([sequence_file_hc,embeds_hc],axis=1)
        embeds_lc = pd.concat([sequence_file_lc,embeds_lc],axis=1)
        embeds = pd.concat([embeds_hc, embeds_lc], ignore_index=True)  
        embeds.to_csv(os.path.join(save_path,f"embeddings_{model_name}.csv"), index=False) 

else: #If model is not Ablang or Sapiens:

    #### Perform calculations

    #Calculate pseudolikelihood, add to sequence_file, and save as CSV
    if "pseudolikelihood" in calc_list:
        sequence_file["evo_likelihood"] = model.calc_pseudo_likelihood_sequence(list(sequence_file[sequences_column]))
        sequence_file.to_csv(os.path.join(save_path,f"evo_likelihood_{model_name}.csv"), index=False)
    
    #Calculate probability matrices and potentially suggested mutations
    if "probability_matrix" in calc_list:   
        # If "suggest_mutations" is in the calc_list, create a dataframe to store all the mutations of all sequences
        if("suggest_mutations" in calc_list):
            all_mutations_df = pd.DataFrame()
        
        # Calculate probability matrix (and mutations) for each sequence
        for index in sequence_file.index:
            
            # Calculates and saves the probability matrix for each sequence
            seq_id = sequence_file[seq_id_column][index]
            prob_matrix = model.calc_probability_matrix(sequence_file[sequences_column][index])
            prob_matrix.to_csv(os.path.join(save_path,f"prob_matrix_seq_{seq_id}_{model_name}.csv"), index = False)
            
            # Calculate the suggested mutations for this sequence
            if("suggest_mutations" in calc_list):
                mutations_df = calculate_mutations(
                    sequences_file=sequence_file.loc[[index]],
                    prob_matrix=prob_matrix,
                    num_mutations=number_mutations,
                    seq_id_column=seq_id_column,
                    sequences_column=sequences_column
                    )
                # Concatenates the mutations obtained from the current sequence to the global DataFrame
                all_mutations_df = pd.concat([all_mutations_df, mutations_df], ignore_index=True)
            
        # Saves all the mutations for all sequences
        if("suggest_mutations" in calc_list):
            output_file = os.path.join(save_path, f"{model_name}_{number_mutations}_mutations.csv")
            all_mutations_df.to_csv(output_file, index=False)
            print(f"All mutations saved to: {output_file}")

    if "embeddings" in calc_list:
        #Calculate embeddings, add to sequence_file, and save as CSV
        embeds = model.fit_transform(sequences=list(sequence_file[sequences_column]))
        embeds = pd.concat([sequence_file,embeds],axis=1)  
        embeds.to_csv(os.path.join(save_path,f"embeddings_{model_name}.csv"), index=False)
