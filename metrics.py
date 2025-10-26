import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from typing import List, Dict


def calculate_division_accuracy(correct_scores: np.ndarray, predicted_scores: np.ndarray, num_divisions: int) -> float:
    """
    Calculate accuracy of dividing scores into n equal groups.
    
    Args:
        correct_scores: Ground truth scores
        predicted_scores: Predicted scores
        num_divisions: Number of divisions (2=halves, 4=quartiles, 10=deciles)
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    correct_groups = pd.qcut(correct_scores, q=num_divisions, labels=False, duplicates='drop')
    predicted_groups = pd.qcut(predicted_scores, q=num_divisions, labels=False, duplicates='drop')
    accuracy = np.mean(correct_groups == predicted_groups)
    return accuracy


def evaluate_ranking(correct_scores: List[float], predicted_scores: List[float], partition: str = 'val') -> Dict:
    """
    Evaluate predictions using multiple metrics.
    
    Args:
        correct_scores: List of ground truth scores
        predicted_scores: List of predicted scores
        partition: Either 'train' or 'val' (used for metric prefix)
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    correct_scores = np.array(correct_scores)
    predicted_scores = np.array(predicted_scores)
    
    mse = np.mean((correct_scores - predicted_scores) ** 2)
    
    kendall_tau, _ = kendalltau(correct_scores, predicted_scores)
    spearman_rho, _ = spearmanr(correct_scores, predicted_scores)
    
    halves_accuracy = calculate_division_accuracy(correct_scores, predicted_scores, num_divisions=2)
    quartile_accuracy = calculate_division_accuracy(correct_scores, predicted_scores, num_divisions=4)
    decile_accuracy = calculate_division_accuracy(correct_scores, predicted_scores, num_divisions=10)
    
    metrics = {
        'mse': mse,
        'kendall_tau': kendall_tau,
        'spearman_rho': spearman_rho,
        'halves_accuracy': halves_accuracy,
        'quartile_accuracy': quartile_accuracy,
        'decile_accuracy': decile_accuracy,
    }
    
    metrics = {f'{partition}_{k}': v for k, v in metrics.items()}
    
    return metrics

