import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

def compute_pca_truth_axis(truth_states, lie_states, n_components=2):
    """
    Computes the dominant Truth Axis using Principal Component Analysis.
    Auto-aligns the sign such that projection values strictly increase toward truth.
    
    Args:
        truth_states: np.ndarray or torch.Tensor of shape (N_truth, hidden_dim)
        lie_states: np.ndarray or torch.Tensor of shape (N_lie, hidden_dim)
    """
    if isinstance(truth_states, torch.Tensor):
        truth_states = truth_states.cpu().numpy()
    if isinstance(lie_states, torch.Tensor):
        lie_states = lie_states.cpu().numpy()
        
    X = np.concatenate([lie_states, truth_states], axis=0)
    y = np.concatenate([np.zeros(len(lie_states)), np.ones(len(truth_states))])
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    raw_vector = pca.components_[0]
    
    # Ensure vector points from Lie -> Truth
    proj_truth = np.dot(truth_states.mean(axis=0), raw_vector)
    proj_lie = np.dot(lie_states.mean(axis=0), raw_vector)
    
    if proj_truth >= proj_lie:
        truth_axis = raw_vector
    else:
        truth_axis = -raw_vector
        X_pca[:, 0] = -X_pca[:, 0]
        
    explained_var = pca.explained_variance_ratio_[0]
    
    return {
        "truth_axis": truth_axis,
        "explained_variance_pc1": explained_var,
        "pca_model": pca,
        "X_pca": X_pca,
        "y": y
    }

def layerwise_emergence_probe(layer_states_truth, layer_states_lie):
    """
    Trains independent linear probes at each decoder layer to evaluate
    representational separability and identify the Emergence Window.
    
    Returns accuracy curve across layers, random-direction baseline, and random-label baseline.
    """
    num_layers = layer_states_truth.shape[1]
    layer_accuracies = []
    random_label_accuracies = []
    
    loo = LeaveOneOut()
    
    for l in range(num_layers):
        X_t = layer_states_truth[:, l, :]
        X_l = layer_states_lie[:, l, :]
        
        X = np.concatenate([X_l, X_t], axis=0)
        y = np.concatenate([np.zeros(len(X_l)), np.ones(len(X_t))])
        
        clf = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
        scores = cross_val_score(clf, X, y, cv=loo, scoring='accuracy')
        layer_accuracies.append(scores.mean())
        
        # Shuffled label control
        y_shuffled = np.random.permutation(y)
        scores_rand = cross_val_score(clf, X, y_shuffled, cv=loo, scoring='accuracy')
        random_label_accuracies.append(scores_rand.mean())
        
    return {
        "layers": list(range(num_layers)),
        "accuracy": layer_accuracies,
        "random_label_baseline": random_label_accuracies
    }
