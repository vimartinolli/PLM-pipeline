# PLM-pipeline

## Workflow

The workflow takes as input a list of sequences in a CSV. These are passed into a pretrained language model to do downstream calculations (likelihood etc.).
#
## Running the workflow

- Create a new environment:
`conda create -n plm`

- Activate the environment:
`conda activate plm`

- To install necessary libraries:
`pip install -r requirements.txt`

## Arguments

- `data_path` path with the location of the CSV file


## Adding new models

- Create a .py file which will contain the model class in the folder `src/`. Be careful that the name of the file is not the same as any of the packages that we are using
- Make a model class file like the others (simple example is the ablang_model). 
- Each class consists of a init function, where you initialize things, like first making the wanted model and adding it to the self.model variable. 
- Then it contains a fit_predict which will use the predict function of the model, to print the output in a csv file.

## Computing likelihoods
- Use the `ESM` or the `Sapiens` with the `prob` method option, to get the probabilities_pseudo.pkl containing a list with the per sequence per position aminoacid likelihoods and the whole sequence log-pseudolikelihood.
