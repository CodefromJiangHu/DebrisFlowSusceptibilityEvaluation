from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm


def visualize_groups(classes, groups, name):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    ax.scatter(range(len(groups)), [.5] * len(groups), c=groups, marker='_',
               lw=50, cmap=cmap_data)
    ax.scatter(range(len(groups)), [3.5] * len(groups), c=classes, marker='_',
               lw=50, cmap=cmap_data)
    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['Data\ngroup', 'Data\nclass'], xlabel="Sample index")

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits + 2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + 2.2, -.2], xlim=[0, len(X)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

def get_groups(numbers_Samples, num_groups):
    groups_num = int(numbers_Samples / num_groups)
    groups = np.hstack([[ii] * groups_num for ii in range(num_groups)])
    for tem_index in range(int(numbers_Samples % num_groups)):
        groups = np.append(groups, num_groups-1)
    # print(groups)
    return groups

def plot_cross_validation_samples_distribution(numbers_Samples, X_train, _percentiles_classes,_groups, _n_splits, cv_StratifiedKFold):
    X = X_train
    n_splits = _n_splits
    percentiles_classes = _percentiles_classes  
    classes = np.hstack([[ii] * int(numbers_Samples * perc)
                   for ii, perc in enumerate(percentiles_classes)])
    for tem_index in range(int(numbers_Samples % len(classes))):
        classes = np.append(classes, len(percentiles_classes)-1)
    print(classes)
    groups = _groups
    visualize_groups(classes, groups, 'no groups')
    this_cv = cv_StratifiedKFold
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_cv_indices(this_cv, X, classes, groups, ax, n_splits)
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Testing set', 'Training set'], loc=(1.02, 0.78))
    # Make the legend fit
    plt.tight_layout()
    fig.subplots_adjust(right=.7)
    # output
    plt.savefig("cross_validation_plot_GroupShuffleSplit.png", bbox_inches="tight")
    plt.show()



if __name__ == '__main__':
    np.random.seed(1338)
    # Generate the class/group data
    X = np.random.randn(562, 10)

    n_splits = 5
    percentiles_classes = [0.5, 0.5] 

    num_groups = 10
    groups = get_groups(len(X), num_groups)
    cv = StratifiedShuffleSplit
    this_cv = cv(n_splits=n_splits, random_state=6)
    plot_cross_validation_samples_distribution(562, X, percentiles_classes, groups, n_splits, this_cv)

