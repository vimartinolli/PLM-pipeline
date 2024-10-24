import os 
import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner
import re # Newly added
import warnings # Newly added
from transformers import logging # Newly added
import logging # Newly added
from transformers import logging as transformers_logging # Newly added
logging.basicConfig(level=logging.WARNING) # Newly added
transformers_logging.set_verbosity_error() # Newly added

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

# Newly added
def calculate_mutations(sequences_file, prob_matrix_folder, num_mutations, output_file, model_name):

    # Check if the necessary columns exist in the input
    if 'sequence' not in sequences_file.columns or 'sequence_id' not in sequences_file.columns:
        raise ValueError("The sequences file must contain columns named 'sequence' and 'sequence_id'")
        
    # Extract the sequences and sequence IDs from the DataFrame
    sequences = sequences_file[['sequence_id', 'sequence']].to_dict(orient='records')

    # Create an empty DataFrame to store all mutations across different sequences
    all_mutations_df = pd.DataFrame()

    # Check if the probability matrix folder exists
    if not os.path.exists(prob_matrix_folder):
        raise FileNotFoundError(f"The probability matrix folder '{prob_matrix_folder}' does not exist.")

    # Iterate over all sequences
    for seq in sequences:

        sequence_id = seq['sequence_id']
        sequence = seq['sequence']

        # Construct the expected filename for the probability matrix based on sequence_id
        prob_matrix_path = os.path.join(prob_matrix_folder, f"prob_matrix_seq{sequence_id}_{model_name}.csv")

        # Check if the file exists
        if not os.path.exists(prob_matrix_path):
            print(f"Warning: Probability matrix for sequence_id {sequence_id} and model_name '{model_name}'not found at {prob_matrix_path}. Skipping.")
            continue

        try:
            # Read the probability matrix
            prob_matrix = pd.read_csv(prob_matrix_path)

            # Verify if all values are numeric
            if not prob_matrix.map(lambda x: isinstance(x, (int, float, complex)) and not pd.isna(x)).all().all():
                raise ValueError(f"The probability matrix '{prob_matrix_path}' contains non-numeric values or missing data.")

            # Convert to numeric
            prob_matrix = prob_matrix.apply(pd.to_numeric, errors='raise')

        except Exception as e:
            print(f"Error loading probability matrix from {prob_matrix_path}: {e}")
            continue

        # Initialize an empty list to store the mutations for the current sequence
        mutations = []

        # Iterate over each position in the sequence
        for position in range(min(len(sequence), len(prob_matrix))):
        
            original_token = sequence[position]

            # Get probabilities for the current position
            position_probs = prob_matrix.iloc[position]

            # Select the three tokens with the highest probabilities
            top_tokens = position_probs.nlargest(3)

            # Iterate over the selected tokens to propose mutations
            for mutated_token, highest_prob_value in top_tokens.items():
            
                # Skip mutation if the mutated token is the same as the original token
                if mutated_token == original_token:
                    continue

                # Get the probability of the original token
                original_prob = position_probs.get(original_token, 0)

                # Calculate delta between the predicted token and the original token
                delta = highest_prob_value - original_prob

                mutations.append({
                    'sequence_id': sequence_id,
                    'sequence': sequence,
                    'position': position + 1,
                    'original_token': original_token,
                    'mutated_token': mutated_token,
                    'delta': delta,
                    'predicted_probabilities': highest_prob_value,
                    'original_probabilities': original_prob,
                    'delta_probabilities': delta
                })

        # Sort mutations by delta value in descending order to prioritize the most significant mutations
        mutations.sort(key=lambda x: x['delta'], reverse=True)

        # Select the top 'num_mutations' based on delta values
        selected_mutations = mutations[:int(num_mutations)]

        # Convert selected mutations to a DataFrame and add to the overall DataFrame
        if selected_mutations:
            selected_mutations_df = pd.DataFrame(selected_mutations)
            selected_mutations_df['mutations'] = selected_mutations_df.apply(lambda row: f"{row['original_token']}{row['position']}{row['mutated_token']}", axis=1)
            all_mutations_df = pd.concat([all_mutations_df, selected_mutations_df], ignore_index=True)

    # Group mutations by sequence_id and combine into a single vector
    if not all_mutations_df.empty:
        grouped_mutations_df = all_mutations_df.groupby('sequence_id').agg({
            'sequence': 'first',
            'mutations': lambda x: '|'.join(x),
            'predicted_probabilities': lambda x: '|'.join(map(str, x)),
            'original_probabilities': lambda x: '|'.join(map(str, x)),
            'delta_probabilities': lambda x: '|'.join(map(str, x))
        }).reset_index()

        # Save all mutations to a CSV file
        grouped_mutations_df.to_csv(output_file, index=False)
        print(f"All mutations saved to: {output_file}")
    else:
        print("No mutations found to save.")