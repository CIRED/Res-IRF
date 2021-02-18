import matplotlib.pyplot as plt


def graph(title=None, xlabel=None, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(xmin, xmax), ylim=(ymin, ymax))

    fig.tight_layout()
