import pandas as pd
import numpy as np
import yaml
from typing import Tuple, List, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import importlib

def load_data(file_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load data from CSV file and return features, target and feature names."""
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()
    return X, y, feature_names

def binary_to_features(binary_list: List[int], feature_names: List[str]) -> List[str]:
    """Convert binary list to feature names."""
    return [feature_names[i] for i, val in enumerate(binary_list) if val == 1]

def load_pipeline_config(config_path: str) -> Pipeline:
    """
    Load and validate pipeline configuration from YAML file.
    
    Expected YAML format:
    steps:
      - name: step1
        class: sklearn.preprocessing.StandardScaler
        parameters:
          with_mean: true
          with_std: true
      - name: step2
        class: sklearn.ensemble.RandomForestClassifier
        parameters:
          n_estimators: 100
          max_depth: 10
    
    Returns:
        sklearn.Pipeline: Configured pipeline
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'steps' not in config:
        raise ValueError("Pipeline configuration must contain 'steps' key")
    
    pipeline_steps = []
    for step in config['steps']:
        if not all(k in step for k in ['name', 'class']):
            raise ValueError("Each step must have 'name' and 'class' keys")
        
        # Import the estimator class
        module_path, class_name = step['class'].rsplit('.', 1)
        module = importlib.import_module(module_path)
        estimator_class = getattr(module, class_name)
        
        # Create estimator instance with parameters if provided
        parameters = step.get('parameters', {})
        estimator = estimator_class(**parameters)
        
        pipeline_steps.append((step['name'], estimator))
    
    return Pipeline(pipeline_steps)
