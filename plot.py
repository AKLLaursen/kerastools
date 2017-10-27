import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

def plots(ims, figsize = (12, 6), rows = 1, interp = False, titles = None):
    
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
            
    f = plt.figure(figsize = figsize)
    for i in range(len(ims)):
        
        sp = f.add_subplot(rows, len(ims) // rows, i + 1)
        sp.axis('Off')
        
        if titles is not None:
            sp.set_title(titles[i], fontsize = 16)
            
        plt.imshow(ims[i], interpolation = None if interp else 'none')


def plot_confusion_matrix(cm, classes, normalize = False,
                          title = 'Confusion matrix', cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
def autolabel(plt, fmt='%.2f'):
    rects = plt.patches
    ax = rects[0].axes
    y_bottom, y_top = ax.get_ylim()
    y_height = y_top - y_bottom
    for rect in rects:
        height = rect.get_height()
        if height / y_height > 0.95:
            label_position = height - (y_height * 0.06)
        else:
            label_position = height + (y_height * 0.01)
        txt = ax.text(rect.get_x() + rect.get_width()/2., label_position,
                fmt % height, ha='center', va='bottom')
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

        
def column_chart(lbls, vals, val_lbls='%.2f'):
    n = len(lbls)
    p = plt.bar(np.arange(n), vals)
    plt.xticks(np.arange(n), lbls)
    if val_lbls: autolabel(p, val_lbls)