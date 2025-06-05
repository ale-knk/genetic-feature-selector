import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def plot_fitness_history(
    history: Dict[str, List],
    title: str = "Evolution of Best Fitness",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot the evolution of fitness values across generations.
    
    Args:
        history: Dictionary containing 'best_fitnesses' list
        title: Title for the plot
        figsize: Figure size as (width, height)
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=figsize)
    plt.plot(history['best_fitnesses'], 'b-', label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_usage(
    history: Dict[str, List],
    feature_names: Optional[List[str]] = None,
    title: str = "Feature Usage Evolution",
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot the evolution of feature usage across generations.
    
    Args:
        history: Dictionary containing 'best_genomes' list
        feature_names: List of feature names. If None, features will be numbered
        title: Title for the plot
        figsize: Figure size as (width, height)
        save_path: If provided, save the plot to this path
    """
    # Convert genomes to numpy array for easier manipulation
    genomes = np.array(history['best_genomes'])
    n_features = genomes.shape[1]
    
    # Calculate feature usage frequency
    feature_usage = np.mean(genomes, axis=0)
    
    # Create feature labels
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
    
    plt.figure(figsize=figsize)
    plt.bar(range(n_features), feature_usage)
    plt.xticks(range(n_features), feature_names, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Usage Frequency')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence(
    history: Dict[str, List],
    window_size: int = 5,
    title: str = "Convergence Analysis",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot the convergence analysis showing the moving average of fitness values.
    
    Args:
        history: Dictionary containing 'best_fitnesses' list
        window_size: Size of the moving average window
        title: Title for the plot
        figsize: Figure size as (width, height)
        save_path: If provided, save the plot to this path
    """
    fitnesses = np.array(history['best_fitnesses'])
    actual_window = min(window_size, len(fitnesses))
    
    plt.figure(figsize=figsize)
    if actual_window > 1:
        moving_avg = np.convolve(fitnesses, np.ones(actual_window)/actual_window, mode='valid')
        plt.plot(fitnesses, 'b-', alpha=0.3, label='Raw Fitness')
        plt.plot(range(actual_window-1, len(fitnesses)), moving_avg, 'r-', 
                label=f'Moving Average (window={actual_window})')
    else:
        plt.plot(fitnesses, 'b-', label='Fitness')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_metrics(
    history: Dict[str, List],
    feature_names: Optional[List[str]] = None,
    window_size: int = 5,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive visualization of all metrics in a single figure.
    
    Args:
        history: Dictionary containing 'best_fitnesses' and 'best_genomes' lists
        feature_names: List of feature names. If None, features will be numbered
        window_size: Size of the moving average window for convergence analysis
        figsize: Figure size as (width, height)
        save_path: If provided, save the plot to this path
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Plot 1: Fitness History
    ax1.plot(history['best_fitnesses'], 'b-')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Score')
    ax1.set_title('Evolution of Best Fitness')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Feature Usage
    genomes = np.array(history['best_genomes'])
    feature_usage = np.mean(genomes, axis=0)
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(len(feature_usage))]
    
    ax2.bar(range(len(feature_usage)), feature_usage)
    ax2.set_xticks(range(len(feature_usage)))
    ax2.set_xticklabels(feature_names, rotation=45, ha='right')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Usage Frequency')
    ax2.set_title('Feature Usage Evolution')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Convergence Analysis
    fitnesses = np.array(history['best_fitnesses'])
    actual_window = min(window_size, len(fitnesses))
    
    if actual_window > 1:
        moving_avg = np.convolve(fitnesses, np.ones(actual_window)/actual_window, mode='valid')
        ax3.plot(fitnesses, 'b-', alpha=0.3, label='Raw Fitness')
        ax3.plot(range(actual_window-1, len(fitnesses)), moving_avg, 'r-', 
                label=f'Moving Average (window={actual_window})')
    else:
        ax3.plot(fitnesses, 'b-', label='Fitness')
    
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Fitness Score')
    ax3.set_title('Convergence Analysis')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) 