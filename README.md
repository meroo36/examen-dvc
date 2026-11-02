# Examen DVC et Dagshub
Dans ce dépôt vous trouverez l'architecture proposé pour mettre en place la solution de l'examen. 

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```
N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.

Vous pouvez télécharger les données à travers le lien suivant : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.

## Solution

Pipeline (`dvc.yaml`, parameters in `params.yaml`):

| Stage | Script | Outputs |
|---|---|---|
| (raw data) | `dvc add data/raw_data/raw.csv` | `data/raw_data/raw.csv.dvc` |
| split | `src/data/split_data.py` | `data/processed_data/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` |
| normalize | `src/data/normalize_data.py` | `data/processed_data/X_train_scaled.csv`, `X_test_scaled.csv`, `models/scaler.pkl` |
| grid_search | `src/models/grid_search.py` | `models/best_params.pkl` |
| train | `src/models/train_model.py` | `models/gbr_model.pkl` |
| evaluate | `src/models/evaluate_model.py` | `data/predictions.csv`, `metrics/scores.json` |

Model: `GradientBoostingRegressor`, hyperparameters chosen with `GridSearchCV` (5-fold, MSE).
The `date` column is dropped (timestamp, not an operational parameter); scaling is fitted on the training set only.

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc pull        # data and models from the DagsHub remote
dvc repro       # rerun the pipeline
dvc metrics show
```
