# DCDA
DCDA: Deep Learning-Based CircRNA-Disease Association Prediction

## Requirements
  Python==3.7.2

You can find required python libraries in requirements.txt.

## Data
In this project, [CircR2Disease](http://bioinfo.snnu.edu.cn/CircR2Disease/) and [Circ2Disease](http://bioinformatics.zju.edu.cn/Circ2Disease/index.html) datasets are used to train and evaluate our methodology and [HMDD v3.2](https://www.cuilab.cn/hmdd) database is used as an extra source while creating features.

Since circRNAs and diseases can have different namings in different sources, we created a [circRNA](/inputs/synonym_dicts/circrna_synonyms_dict.pkl) and [disease](/inputs/synonym_dicts/disease_synonyms_dict.pkl) synonym dictionaries and mapped circRNA and disease names to same names in datasets. Cleaned datasets are in [cleaned directory](/data/cleaned).

## Usage

### Feature Generation

In order to train a model, first you need to create features to be used in the methodology. You can create features using feature_generation.py script. While running this script, you must specify the dataset name (either 'CircR2Disease' or 'Circ2Disease'). Generated features will go under [features directory](/inputs/features). An example of usage is as follows:

  python feature_generation.py --data_name CircR2Disease
  
### Cross Validation

If you want to run cross validation on the methodology, you can use cross_validation.py script. You can change parameters of the methodology by changing variables in the script. While running this script, you must specify which dataset's features are going to be used (either 'CircR2Disease' or 'Circ2Disease'). Cross validation results and Roc AUC curve plot of folds will be saved to [cv results directory](outputs/results/cv). An example of usage is as follows:

  python cross_validation.py --data_name CircR2Disease
  
### Training Model

In order to create a full model, you should use train.py script. This script trains models in the methodology with all data. You can change parameters of the methodology by changing variables in the script. While running this script, you must specify which dataset's features are going to be used (either 'CircR2Disease' or 'Circ2Disease'). Trained models will go under [models directory](outputs/models). An example of usage is as follows:

  python train.py --data_name CircR2Disease
  
### Prediction

You can predict circRNA-disease association scores using predict.py script. For a given disease name, this script genearates novel circRNA-disease pairs that are not in the main dataset and predicts the association score. While running this script, you must specify which dataset's features are going to be used (either 'CircR2Disease' or 'Circ2Disease') and a disease name. Predictions are saved to [prediction results directory](outputs/results/predictions). An example of usage is as follows:

  python predict.py --data_name CircR2Disease --disease_name Lung Neoplasms




