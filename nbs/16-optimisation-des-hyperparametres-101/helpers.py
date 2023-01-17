import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc_curve(results, y):
    fpr = []
    tpr = []
    names = []
    for r in results.items():
        fpr_, tpr_, _ = roc_curve(y, r[1])
        fpr.append(fpr_)
        tpr.append(tpr_)
        names.append(r[0])
    
    fig = plt.figure(1, figsize=(12, 12))
    plt.plot([0, 1], [0, 1], 'k--')
    
    for i, el in enumerate(names):
        plt.plot(fpr[i], tpr[i], label=el)
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')