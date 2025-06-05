import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from typing import List, Union
import pandas as pd

def evaluate_fitness(
    genome: List[int],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    estimator,
    cv: int = 5,
    scoring: str = "accuracy",
) -> float:
    """
    Evaluate the fitness of a feature subset using cross-validation.
    
    Args:
        genome: Binary list indicating which features to select
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target vector
        estimator: Scikit-learn estimator
        cv: Number of cross-validation folds
        scoring: Scoring metric to use
    
    Returns:
        float: Mean cross-validation score
    """
    # Convert genome to indices of selected features
    selected_indices = [i for i, gene in enumerate(genome) if gene == 1]
    
    # If no features are selected, return a very low score
    if not selected_indices:
        return -np.inf
    
    # Select features
    if isinstance(X, pd.DataFrame):
        X_selected = X.iloc[:, selected_indices]
    else:
        X_selected = X[:, selected_indices]
    
    # Calculate cross-validation score
    scores = cross_val_score(
        estimator,
        X_selected,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    return np.mean(scores)
