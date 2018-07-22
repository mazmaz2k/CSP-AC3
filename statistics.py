import matplotlib.pyplot as plt


def create_statistics_graph(x, y, x_title, y_title, save=False, output_name='statistics.jpg'):
    plt.rcParams['figure.figsize'] = [10, 6]
    marksize = (plt.rcParams['lines.markersize'] ** 2) * 1.6
    plt.scatter(x, y, alpha=0.5, marker='o', s=marksize)
    plt.xlabel(x_title, fontsize=14)
    plt.ylabel(y_title, fontsize=14)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)

    if save is True:
        plt.savefig(output_name)

    plt.show()

    plt.clf() #clear figure





x = [1, 2, 3, 4, 4, 5]
y = [2, 4, 6, 8.1, 8, 13]
create_statistics_graph(x, y, 'string for x', 'string for y', True)
