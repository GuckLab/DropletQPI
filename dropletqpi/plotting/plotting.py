import matplotlib.pyplot as plt


def plot_scatter(x, y, title='', return_fig = True):

    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        plt.xlabel('droplet radius [m]')
        plt.ylabel('refractive index')

        plt.scatter(x, y, s=30, alpha=0.8, color='steelblue')

        # plt.scatter(export_data['radius'].median(),
        #             export_data['refractive index'].median(),
        #             s=60, alpha=0.8, color=(0.2, 0.2, 0.2))
        plt.errorbar(x.median(),
                     y.median(),
                     xerr=x.std(),
                     yerr=y.std(),
                     marker='o', alpha=0.8, color=(0.2, 0.2, 0.2))
        plt.title(title)
        plt.show()

        # if savefig:
        #     plt.savefig(str(savepath) + "\\" + "RI_vs_r"
        #                 + "_" + "full" + ".png", dpi=300,
        #                 transparent=True, bbox_inches='tight')
    if return_fig:
        return fig, ax


def plot_violine(df, title='', return_fig = True):
    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        plt.violinplot(df)
        # plt.xlabel('droplet radius [m]')
        # plt.ylabel('refractive index')

        if return_fig:
            return fig, ax

