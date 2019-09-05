import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean

""" Print scores for the classifier, evaluating the given test.
    
    Parameters
    ----------
    confusion_matrix:
        bool. Plot normalized confusion matrix.
"""
def score(classifier, test_x, test_y, classes, *, confusion_matrix = False, normalize = False):
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    y_true = test_y
    y_pred = np.array([classifier.predict(np.array(ex)) for ex in test_x])

    accuracies, precisions, recalls, f1s, tp_rates, fp_rates = [], [], [], [], [], []
    for i, klass in enumerate(classes):
        print(f'* Class: {klass}')

        TP = true_positives(i, y_pred, y_true)
        TN = true_negatives(i, y_pred, y_true)
        FN = false_negatives(i, y_pred, y_true)
        FP = false_positives(i, y_pred, y_true)

        # print('TP', TP)
        # print('TN', TN)
        # print('FN', FN)
        # print('FP', FP)
        
        accuracy = (TP + TN)/(TP + FP + TN + FN)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        f1 = (2 * precision * recall)/(precision + recall)
        tp_rate = TP/(TP + FN)
        fp_rate = FP/(FP + TN)

        print(f'Accuracy {accuracy:.5f} | Precision {precision:.5f} | Recall {recall:.5f} | TP-rate {tp_rate:.5f} | FP-rate {fp_rate:.5f} | F1 {f1:.5f} \n')

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        tp_rates.append(tp_rate)
        fp_rates.append(fp_rate)

    print('HARMONIC MEANS\n')
    print(f'    Accuracy: {hmean(accuracies):.5f}')
    print(f'    Precision: {hmean(precisions):.5f}')
    print(f'    Recall: {hmean(recalls):.5f}')
    print(f'    TP rate: {hmean(tp_rates):.5f}')
    if np.sum(fp_rates) > 0:
        print(f'    FP rate: {hmean(fp_rates):.5f}')
    else:
        print(f'    FP rate: {0:.5f}')
    print(f'    F1 score: {hmean(f1s):.5f}')

    if confusion_matrix:
        # Plot confusion matrix
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix'
        plot_confusion_matrix(y_true, y_pred, classes=classes, normalize=normalize, title=title)

        plt.show()

def true_positives(c, y_pred, y_true):
    t_indexes = np.where(y_true == c)
    tp = np.count_nonzero(y_pred[t_indexes] == c)
    return tp

def true_negatives(c, y_pred, y_true):
    n_indexes = np.where(y_true != c)
    tn = np.count_nonzero(y_pred[n_indexes] != c)
    return tn

def false_positives(c, y_pred, y_true):
    p_indexes = np.where(y_pred == c)
    fp = np.count_nonzero(y_true[p_indexes] != c)
    return fp

def false_negatives(c, y_pred, y_true):
    n_indexes = np.where(y_pred != c)
    fn = np.count_nonzero(y_true[n_indexes] == c)
    return fn

def calculate_confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for tl in classes:
        true_ind = np.where(y_true == tl)
        pred = y_pred[true_ind]
        for pl in classes:
            cm[tl, pl] = np.count_nonzero(pred == pl)

    return cm

# Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax