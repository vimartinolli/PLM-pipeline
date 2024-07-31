# PLM-pipeline

The pipeline takes as input a list of sequences in a CSV. These are passed into a pretrained language model to do downstream calculations (likelihood etc.).

## Set up environment

- Create a new environment:
`conda create -n plm`

- Activate the environment:
`conda activate plm`

- To install necessary libraries:
`pip install -r requirements.txt`

## Run the pipeline
Run the script `pipeline.py ` with the following arguments:
- `model_name` class name of the model to use. Currently available: Ablang, ProtBert, Sapiens, ESM
- `file_path` path with the location of the input CSV file
- `sequence_column` column name of the sequence to use in the input CSV
- `output_folder` folder name to store output CSV

## Adding new models
- Create a .py file which will contain the model class in the folder `src/`. Be careful that the name of the file is not the same as any of the packages that we are using
- Make a model class file like the others (simple example is the ablang_model). 
- Each class consists of a init function, where you initialize things, like first making the wanted model and adding it to the self.model variable. 
- Then it contains functions for downstream calculations:
    - `fit_transform` which will transform the sequences using the model and create embeddings.
    - `calc_pseudo_likelihood_sequence` calculates the pseudolikelihood of whole sequence (sum of log-scaled per-residue probablities, divided by sequence length)
    - `calc_probability_matrix` calculates the probability matrix of each amino acid per position of a sequence
