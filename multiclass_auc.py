import numpy as np
import itertools
"""
    MAUCpy
    ~~~~~~
    Contains two equations from Hand and Till's 2001 paper on a multi-class
    approach to the AUC. The a_value() function is the probabilistic approximation
    of the AUC found in equation 3, while MAUC() is the pairwise averaging of this
    value for each of the classes. This is equation 7 in their paper.
"""


def a_value(y_true, y_pred_prob, zero_label=0, one_label=1):
    """
    Approximates the AUC by the method described in Hand and Till 2001,
    equation 3.
    
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the predicted
    probability list.
    
    Args:
        y_true: actual labels of test data 
        y_pred_prob: predicted class probability
        zero_label: label for positive class
        one_label: label for negative class
    Returns:
        The A-value as a floating point.
    """
    
    idx = np.isin(y_true, [zero_label, one_label])
    labels = y_true[idx]
    prob = y_pred_prob[idx, zero_label]
    sorted_ranks = labels[np.argsort(prob)]
    
    n0, n1, sum_ranks = 0, 0, 0
    n0 = np.count_nonzero(sorted_ranks==zero_label)
    n1 = np.count_nonzero(sorted_ranks==one_label)
    sum_ranks = np.sum(np.where(sorted_ranks==zero_label)) + n0

    return (sum_ranks - (n0*(n0+1)/2.0)) / float(n0 * n1)  # Eqn 3


def MAUC(y_true, y_pred_prob, num_classes):
    """
    Calculates the MAUC over a set of multi-class probabilities and
    their labels. This is equation 7 in Hand and Till's 2001 paper.
    
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    
    Args:
        y_true: actual labels of test data 
        y_pred_prob: predicted class probability
        zero_label: label for positive class
        one_label: label for negative class
        num_classes (int): The number of classes in the dataset.
    
    Returns:
        The MAUC as a floating point value.
    """
    # Find all pairwise comparisons of labels
    class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]

    # Have to take average of A value with both classes acting as label 0 as this
    # gives different outputs for more than 2 classes
    sum_avals = 0
    for pairing in class_pairs:
        sum_avals += (a_value(y_true, y_pred_prob, zero_label=pairing[0], one_label=pairing[1]) +
                      a_value(y_true, y_pred_prob, zero_label=pairing[1], one_label=pairing[0])) / 2.0

    return sum_avals * (2 / float(num_classes * (num_classes-1)))  # Eqn 7
