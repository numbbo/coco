import cocoex
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from plot_bbob import best_parameter


def plot_all_combinations(dim=5, inst=1):
    print('Attention! This will take a long time and produce huge plots!')

    suite = cocoex.Suite('bbob-mixint', '', '')
    N = 501
    x = np.linspace(-5, 5, N)
    cmap = mpl.colormaps['plasma']
    palette = sns.color_palette([cmap(v) for v in [0.0, 1/3, 2/3, 0.9] for _ in (0, 1)], 8)
    dashes = [(1, 0), (3, 1)] * 4
    plot_folder = 'plots_mixint'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    for fid in range(23, 25):
        print(f'Working on f{fid} ... ', end='')
        fn = suite.get_problem_by_function_dimension_instance(fid, dim, inst)
        f_list = []
        for x2 in range(4):
            for x1 in range(2):
                for x3 in range(8):
                    for x4 in range(16):
                        for i, x5 in enumerate(x):
                            f_list.append({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
                                           'f': fn([x1, x2, x3, x4, x5])})
        df = pd.DataFrame(f_list)
        df['x1x2'] = 'x1 = ' + df['x1'].astype(str) + ' | x2 = ' + df['x2'].astype(str)
        print(f'plotting ... ', end='')
        g = sns.relplot(data=df, x='x5', y='f', col='x4', row='x3', hue='x1x2', style='x1x2', kind='line',
                        palette=palette, dashes=dashes)
        g.fig.suptitle(f'bbob-mixint f{fid}')
        #  g.fig.subplots_adjust(top=0.98)
        g.legend.set_title(None)
        g.set_ylabels(rotation=0)
        print(f'saving ... ', end='')
        # plt.tight_layout()
        plt.savefig(f'{plot_folder}/mixint_f_{fid:02}.png', bbox_extra_artists=(g.legend, ), bbox_inches='tight')
        plt.close()
        print(f'done!')


def plot_some_combinations(dim=5, inst=1):
    suite = cocoex.Suite('bbob-mixint', '', '')
    N = 501
    x = np.linspace(-5, 5, N)
    cmap = mpl.colormaps['plasma']
    palette = sns.color_palette([cmap(v) for v in [0.0, 0.5]], 2)
    dashes = [(1, 0), (3, 1)]
    plot_folder = 'plots_mixint'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    for fid in range(1, 25):
        print(f'Working on f{fid} ... ', end='')
        fn = suite.get_problem_by_function_dimension_instance(fid, dim, inst)
        f_list = []
        for x2 in range(2, 4):
            for x1 in range(2):
                for x3 in range(6, 8):
                    for x4 in range(14, 16):
                        for i, x5 in enumerate(x):
                            f_list.append({'x1': x1, 'x2': x2, '_x3': x3, 'x4': x4, 'x5': x5,
                                           'f': fn([x1, x2, x3, x4, x5])})
        df = pd.DataFrame(f_list)
        df['x3'] = df['_x3'].astype(str) + ' | x4 = ' + df['x4'].astype(str)
        print(f'plotting ... ', end='')
        g = sns.relplot(data=df, x='x5', y='f', col='x3', row='x2', hue='x1', style='x1', kind='line',
                        palette=palette, dashes=dashes)
        g.fig.suptitle(f'bbob-mixint f{fid}')
        g.fig.subplots_adjust(top=0.92)
        # g.legend.set_title(None)
        g.set_ylabels(rotation=0, labelpad=10)
        print(f'saving ... ', end='')
        plt.savefig(f'{plot_folder}/mixint_f_{fid:02}.png')#, bbox_extra_artists=(g.legend,), bbox_inches='tight')
        plt.close()
        print(f'done!')


def plot_each_variable(dim=5, inst=1, save_plots=True):
    suite = cocoex.Suite('bbob-mixint', '', '')
    N = 501
    x5 = np.linspace(-5, 5, N)
    plot_folder = 'plots_mixint'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    scale = 0.8
    for func in range(1, 25):
        fig, axes = plt.subplots(2, 2, figsize=(6.4 * 2 * scale, 4.8 * 2 * scale))
        for x_i, ax in enumerate(axes.flatten()):
            problem = suite.get_problem_by_function_dimension_instance(func, dim, inst)
            x = best_parameter(problem)
            x_values = np.arange(2 ** (x_i + 1))
            f_list = []
            for v in x_values:
                x[x_i] = v
                for i, x5_i in enumerate(x5):
                    x[-1] = x5_i
                    f_list.append({f'x{x_i + 1}': v, 'x5': x5_i, 'f': problem(x)})
            df = pd.DataFrame(f_list)
            sns.lineplot(data=df, x='x5', y='f', hue=f'x{x_i + 1}', ax=ax, palette='plasma')
            ax.scatter(best_parameter(problem)[-1], problem(best_parameter(problem)), marker='o', color='k', zorder=2)
            ax.set_ylabel('f', rotation=0, labelpad=10)

        plt.tight_layout()
        if save_plots:
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
            plt.savefig(f'{plot_folder}/bbob-mixint_variables_f{func}_d{dim}_i{inst:02}.png')
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    plot_each_variable()
