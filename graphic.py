import matplotlib.pyplot as plt


def graph(title=None, xlabel=None, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(xmin, xmax), ylim=(ymin, ymax))

    fig.tight_layout()


def make_plot_series(ds, level, index_level=None):
    """
    Option:
    option == 'bar'
    x-axis is discrete (numeric or not) value and based on level index;
    y-axis is numeric and continuous value and based on serie value;
    Return hist
    """
    ds = ds.groupby(level).sum()
    bad_index = ds.index == 'nan'
    ds = ds[~bad_index]

    fig = plt.figure(figsize=(18, 6))

    if index_level:
        ds.loc[index_level].plot(kind='bar')
    else:
        ds.plot(kind='bar')

