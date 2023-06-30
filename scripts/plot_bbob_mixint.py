import cocoex
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


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
    fn = suite.get_problem_by_function_dimension_instance(fid, 5, 1)
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
    g.fig.subplots_adjust(top=0.98)
    g.legend.set_title(None)
    g.set_ylabels(rotation=0)
    print(f'saving ... ', end='')
    plt.savefig(f'{plot_folder}/mixint_f_{fid:02}.png')
    plt.close()
    print(f'done!')
