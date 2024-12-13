"""
Script that trains ML models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
import pandas as pd
import deepchem as dc
from deepchem.molnet import load_tox21

from sklearn.linear_model import LogisticRegression
from deepchem.models.graph_models import GraphConvModel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from itertools import product
from collections.abc import Iterable

import tensorflow as tf
import copy
import sys
import warnings
import logging

import time
from datetime import datetime

warnings.filterwarnings(action='ignore')
logging.getLogger('tensorflow').disabled = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def model_dataset(model_name):
    # Only for debug!
    np.random.seed(123)

    # Load Tox21 dataset
    n_features = 1024
    if model_name == 'gcn':
        tox21_tasks, tox21_datasets, transformers = load_tox21(data_dir=".",featurizer='GraphConv', splitter='scaffold')
        train_dataset, valid_dataset, test_dataset = tox21_datasets
        return train_dataset, valid_dataset, test_dataset, tox21_tasks, transformers
    else:
        tox21_tasks, tox21_datasets, transformers = load_tox21(data_dir=".",featurizer='ECFP', splitter='scaffold')
        train_dataset, valid_dataset, test_dataset = tox21_datasets
        return train_dataset, valid_dataset, test_dataset, tox21_tasks, transformers

def ParameterGrid(param_dict):
    sequential = False
    if not isinstance(param_dict, dict):
        raise TypeError("Parameter grid is not a dict ({!r}))".format(param_dict))
    if isinstance(param_dict, dict):
        for key in param_dict:
            if not isinstance(param_dict[key], Iterable):
                raise TypeError("Parameter grid value is not iterable"
                               "(key={!r}, value={!r})".format(key, param_dict[key]))
    items = sorted(param_dict.items())
    keys, values = zip(*items)
    
    params_grid=[]
    if sequential:
        arrays = [param_dict[key] for key in keys]
        combined = list(zip(*arrays))
        for item in combined:
            a = dict(zip(keys,item))
            params_grid.append(a)
    else:
        for v in product(*values):
            params_grid.append(dict(zip(keys, v)))
    
    return params_grid

def hpo_range(model_name):
    params_dict={}
    if model_name == "rf":
        params_dict = {
        'max_depth': [1],
        'max_leaf_nodes': [2],
        'max_samples' : [round(float(x), 3) for x in np.linspace(start=0.025, stop=0.333, num=7)],
        'min_samples_split': [2],        
        'n_estimators': np.delete(np.insert(np.arange(0,501,20),0,1),1)
        }
    elif model_name == "gcn":
        params_dict = {
        'epoch': [10, 50, 100],
        'dropout': [round(float(x), 2) for x in np.linspace(start=0.0, stop=0.9, num=10)],
        'graph_conv_layer': [ [32,32] ,  [64,64] , [128,128] ],
        'num_atom_features': [30, 75, 105]  
        }           
    elif model_name == "logreg":
        params_dict = {
        'penalty': ["l2","l1",None],
        'C': [0.001] ,
        'solver': ['lbfgs','liblinear','sag','saga'],
        'max_iter': [int(x) for x in np.linspace(start=100, stop=1000, num=25)],
        'class_weight': ['balanced']
        }
    elif model_name == "svm":
        params_dict = {
        'C': np.logspace(-3,2,6),
        'kernel': ['linear','rbf', 'sigmoid'],
        'gamma' : np.logspace(-3, 2, 6)
        }           
    return params_dict               

def model_train(model_name, hpo_combination, model_dir, tox21_tasks, train_dataset, get_model):
    model_dir = model_dir
    param = hpo_combination
    if model_name == "gcn":
        epoch = param['epoch']
        param1 = copy.deepcopy(param)
        del param1['epoch']
        model = GraphConvModel(len(tox21_tasks), batch_size=batch_size, mode='classification',**param1)
        if get_model:
            return model
        else:
            # Fit trained model
            model.fit(train_dataset, nb_epoch=epoch)
            model.save_checkpoint(model_dir=model_dir)
            return model
    else:
        def builder(model_dir):
            if model_name == "rf":
                sklearn_model = RandomForestClassifier(**param)
                return dc.models.SklearnModel(sklearn_model, model_dir)
            elif model_name == "logreg":
                sklearn_model = LogisticRegression(**param)
                return dc.models.sklearn_models.SklearnModel(sklearn_model, model_dir)
            elif model_name == "svm":
                sklearn_model = SVC(**param)
                return dc.models.SklearnModel(sklearn_model, model_dir) 
        model = dc.models.multitask.SingletaskToMultitask(tox21_tasks, builder, model_dir=model_dir)
        model.fit(train_dataset)
        return model

def get_chunk(parmas, chunk_number, part):
    chunk_number = int(chunk_number)
    part = int(part)
    chunk_size = len(parmas) // chunk_number
    start = 0
    chunks = []
    start_idx = []
    for i in range(chunk_number):
        start_idx.append(start)
        end = start + chunk_size
        chunks.append(parmas[start:end])
        start = end
    if 0 <= part < chunk_number:
        return start_idx[part], chunks[part]
    else:
        return []
    
def model_hpo(model_name, chunk_number, part):
    now = time.time()
    dt = datetime.fromtimestamp(now).strftime("%Y%m%d%H%M%S")
    train_dataset, valid_dataset, test_dataset, tox21_tasks, transformers = model_dataset(model_name)
    
    params_dict = hpo_range(model_name)
    params = ParameterGrid(params_dict)
    idx, sub_params = get_chunk(params, chunk_number, part)
    params_df = pd.DataFrame(sub_params)
    param_path=f"./_params/{model_name}_{part}_params_{dt}.csv"
    params_df.to_csv(param_path)
    print(f"save param path => {param_path}")
    hp_columns = params_df.columns
    
    result = pd.DataFrame()
    for i in range(len(sub_params)):
        sub_index = i
        total_index = idx+i
        param = sub_params[sub_index]
        r = param
        print(f"{total_index}: {r}")
        model_dir=f"tox21_{model_name}/{model_name}_{total_index}"
        print(f"save model path => {model_dir}")
        if model_name == "gcn" and os.path.exists(os.path.join(model_dir)):
            print("restore pretrained model")
            model = model_train(model_name, r, model_dir, tox21_tasks, train_dataset, True)
            model.restore(model_dir=model_dir)
        else:
            model = model_train(model_name, r, model_dir, tox21_tasks, train_dataset, False)
        
        if model is None:
            print(f"{r} can't fit")
            pass
        else:    
            df = model_evaluate(sub_index, total_index, model, model_name, r, test_dataset, transformers, tox21_tasks)
            if len(result) == 0:
                result = df
            else:
                result = pd.concat([result, df])
    result = result[['sub_index','total_index','metric', 'score', *hp_columns, *tox21_tasks]]
    print(f"save result path  => tox21_{model_name}_{part}_{dt}.csv")
    result.to_csv(f"./_result/tox21_{model_name}_{part}_{dt}.csv")
    
def model_evaluate(sub_index, total_index, model, model_name, r, test_dataset, transformers, tox21_tasks):
    # Fit models
    metric_array = {"f1_score":dc.metrics.Metric(dc.metrics.f1_score),
                    "roc_auc_score": dc.metrics.Metric(dc.metrics.roc_auc_score),
                    "precision_score": dc.metrics.Metric(dc.metrics.precision_score),
                    "recall_score": dc.metrics.Metric(dc.metrics.recall_score),
                    "accuracy_score": dc.metrics.Metric(dc.metrics.accuracy_score)}
    result_df=pd.DataFrame()
    for m in metric_array:
        result={}
        score_name = m
        metric = metric_array[score_name]
        task_test = model.evaluate(test_dataset, [metric], transformers, per_task_metrics=True)
        task_score = task_test[1][score_name]
        task_score_1 = task_test[0][score_name]
        task_score_dict = {task : task_score[i] for i, task in enumerate(tox21_tasks)}
        task_score_dict["sub_index"] = sub_index
        task_score_dict["total_index"] = total_index
        task_score_dict["metric"] = score_name
        task_score_dict["score"] = task_score_1
        score_df = pd.DataFrame.from_dict(orient='index', data=task_score_dict)
        score_df = score_df.T
        param_df = pd.DataFrame.from_dict(orient='index',data=r)
        param_df = param_df.T
        new_df = pd.concat([param_df, score_df], axis=1)
        print(f"{m} test result")
        print(task_score_dict)
        if len(result_df) == 0:
            result_df = new_df
        else:
            result_df = pd.concat([result_df, new_df])
        
    return result_df

if __name__ == "__main__":
    model_name=sys.argv[1]
    chunk_number = sys.argv[2]
    part = sys.argv[3]
    batch_size = 50
    model_hpo(model_name, chunk_number, part)


