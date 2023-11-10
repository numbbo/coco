import cocoex
import numpy as np
import matplotlib.pyplot as plt
import os

COLORS_PLOTLY_BOLD = ['#7F3C8D', '#11A579', '#3969AC', '#F2B701', '#E73F74', '#80BA5A',
                      '#E68310', '#008695', '#CF1C90', '#F97B72', '#A5AA99']
COLORS_CATEGORICAL = [COLORS_PLOTLY_BOLD[i] for i in [0, 5, 3, 4, 2]]


def best_parameter(problem):
    problem._best_parameter('print')
    with open('._bbob_problem_best_parameter.txt', 'rt') as file_:
        return [float(s) for s in file_.read().split()]


def get_landscape(problem, num_intervals, best_param, x_index=0, y_index=1):
    """Returns the problem landscape - the chosen axis-aligned plane through the optimum"""
    X, Y = np.meshgrid(np.linspace(problem.lower_bounds[x_index], problem.upper_bounds[x_index], num_intervals),
                       np.linspace(problem.lower_bounds[y_index], problem.upper_bounds[y_index], num_intervals))
    x = X.reshape(num_intervals * num_intervals)
    y = Y.reshape(num_intervals * num_intervals)
    z = np.vstack([x, y]).transpose()
    x_best = np.tile(best_param, (len(z), 1))
    x_best[:, x_index] = z[:, 0]
    x_best[:, y_index] = z[:, 1]
    z = np.array([problem(x_i) for x_i in x_best])
    Z = z.reshape((num_intervals, num_intervals))
    return X, Y, Z


def get_cuts(problem, num_intervals, best_param, x_index=0):
    """Returns axis-aligned 1-D cuts through the optimum"""
    x = np.linspace(-5, 5, num_intervals)
    x_best = np.tile(best_param, (num_intervals, 1))
    x_best[:, x_index] = x
    z = np.array([problem(x_i) for x_i in x_best])
    return x, z


def get_diagonal_cuts(problem, num_intervals, best_param):
    """Returns diagonal 1-D cuts through the optimum"""
    x = np.linspace(-5 * np.sqrt(problem.dimension), 5 * np.sqrt(problem.dimension), num_intervals)
    x_best = np.tile(best_param, (num_intervals, 1))
    x_best += x[:, np.newaxis]
    z = np.array([problem(x_i) for x_i in x_best])
    x_scaled = np.interp(x, (min(x), max(x)), (-5, +5))
    return x, x_scaled, z


def plot_all_views(func, dim, inst, N=501, plot_folder='plots_bbob', x_index=0, y_index=1,
                   cmap='viridis', cmap_lines='Greys', save_plots=True):
    suite = cocoex.Suite("bbob", "", "")
    problem = suite.get_problem_by_function_dimension_instance(func, dim, inst)
    best_param = best_parameter(problem)
    best_value = problem(best_param)
    best = best_param + [best_value]
    problem_name = f'bbob $f_{{{func}}}$, {dim}-D, inst. {inst}'

    if save_plots:
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

    if dim == 2:
        X, Y, Z = get_landscape(problem, num_intervals=N, best_param=best_param, x_index=x_index, y_index=y_index)

        # Surface plot
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8)
        ax.scatter(*best, c='black', marker='x', zorder=10)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f')
        ax.set_title(problem_name)
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{plot_folder}/bbob_f{func:02}_d{dim:02}_i{inst:02}_surface.png')
        else:
            plt.show()
        plt.close()

        # Heat map with level sets
        fig, ax = plt.subplots(figsize=(6.4 * 0.75, 4.8 * 0.75))
        plot = ax.pcolormesh(X, Y, Z)
        ax.contour(X, Y, Z, cmap=cmap_lines)
        ax.scatter(*best_param, c='white', marker='x', zorder=10)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2', rotation=0)
        ax.set_title(problem_name)
        fig.colorbar(plot)
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{plot_folder}/bbob_f{func:02}_d{dim:02}_i{inst:02}_heat.png')
        else:
            plt.show()
        plt.close()

    else:
        # A matrix of heat maps with level sets
        fig, axes = plt.subplots(dim - 1, dim - 1, figsize=((dim - 1) * 6.4 * 0.5, (dim - 1) * 4.8 * 0.5))
        for r, row in enumerate(axes):
            x2 = r + 1
            for x1, ax in enumerate(row):
                if x1 < x2:
                    X, Y, Z = get_landscape(problem, num_intervals=N, best_param=best_param, x_index=x1, y_index=x2)
                    best_param_x_y = np.take(best_param, [x1, x2])
                    plot = ax.pcolormesh(X, Y, Z)
                    ax.contour(X, Y, Z, cmap=cmap_lines)
                    ax.scatter(*best_param_x_y, c='white', marker='x', zorder=10)
                    ax.set_xlabel(f'x{x1 + 1}')
                    ax.set_ylabel(f'x{x2 + 1}', rotation=0)
                    plt.colorbar(plot, ax=ax)
                else:
                    ax.set_visible(False)
        plt.suptitle(problem_name)
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{plot_folder}/bbob_f{func:02}_d{dim:02}_i{inst:02}_heat.png')
        else:
            plt.show()
        plt.close()

    # Cut through the search space and store the results
    z_min = np.inf
    z_max = - np.inf
    cuts_x = []
    cuts_z = []
    for i in range(min(dim, 5)):
        x_i, z_i = get_cuts(problem, N, best_param, x_index=i)
        cuts_x.append(x_i)
        cuts_z.append(z_i)
        z_min = min(z_min, min(z_i))
        z_max = max(z_max, max(z_i))
    x, x_scaled, z = get_diagonal_cuts(problem, N, best_param)
    cuts_x = np.array(cuts_x)
    cuts_z = np.array(cuts_z)

    # Find where the diagonal cut exits the [-5, 5]^d range
    valid_idx = np.where((x >= -5) & (x <= 5))[0]
    x_low = x_scaled[valid_idx[0]]
    x_high = x_scaled[valid_idx[-1]]
    z_low = z[valid_idx[0]]
    z_high = z[valid_idx[-1]]
    z_min = min(z_min, z_low)
    z_max = max(z_max, z_high)
    x_min = np.min(cuts_x)

    # Plot 1: lin-lin axes
    fig, ax = plt.subplots()
    for i in range(min(dim, 5)):
        ax.plot(cuts_x[i], cuts_z[i], label=f'x{i + 1}', color=COLORS_CATEGORICAL[i])
    ax.plot(x_scaled, z, label='ones', linestyle='dashed', color='gray')
    ax.scatter([x_low, x_high], [z_low, z_high], marker='*', color='gray')
    ax.set_ylim([z_min - 0.02 * (z_max - z_min), z_max + 0.1 * (z_max - z_min)])
    ax.legend()
    ax.grid(ls='dotted')
    plt.suptitle(problem_name)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{plot_folder}/bbob_f{func:02}_d{dim:02}_i{inst:02}_cuts_lin_lin.png')
    else:
        plt.show()
    plt.close()

    # Plot 2: lin-log axes
    fig, ax = plt.subplots()
    for i in range(min(dim, 5)):
        ax.semilogy(cuts_x[i], cuts_z[i] - z_min, label=f'x{i + 1}', color=COLORS_CATEGORICAL[i])
    ax.semilogy(x_scaled, z - z_min, label='ones', linestyle='dashed', color='gray')
    ax.scatter([x_low, x_high], [z_low - z_min, z_high - z_min], marker='*', color='gray')
    ax.legend()
    ax.grid(ls='dotted')
    plt.suptitle(problem_name)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{plot_folder}/bbob_f{func:02}_d{dim:02}_i{inst:02}_cuts_lin_log.png')
    else:
        plt.show()
    plt.close()

    # Plot 3: log-log axes
    fig, ax = plt.subplots()
    for i in range(min(dim, 5)):
        x_opt = cuts_x[i][np.argmin(cuts_z[i])]
        ax.loglog(cuts_x[i][cuts_x[i] > x_opt] - x_opt, cuts_z[i][cuts_x[i] > x_opt] - z_min,
                  label=f'x{i + 1}', color=COLORS_CATEGORICAL[i])
        ax.loglog(x_opt - cuts_x[i][cuts_x[i] < x_opt], cuts_z[i][cuts_x[i] < x_opt] - z_min,
                  label=u'\u2212' + f'x{i + 1}', color=COLORS_CATEGORICAL[i], linestyle='dotted')
    ax.loglog(x_scaled[x_scaled > 0], z[x_scaled > 0] - z_min,
              label='ones', linestyle='dashed', color='gray')
    ax.loglog(- x_scaled[x_scaled < 0], z[x_scaled < 0] - z_min,
              label=u'\u2212' + 'ones', linestyle='dotted', color='gray')
    ax.legend()
    ax.grid(ls='dotted')
    plt.suptitle(problem_name)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{plot_folder}/bbob_f{func:02}_d{dim:02}_i{inst:02}_cuts_log_log.png')
    else:
        plt.show()
    plt.close()


def plot_all_functions(suite_name, dim, inst, N=101, plot_folder='plots_bbob', x_index=0, y_index=1, cmap='viridis',
                       cmap_lines='Greys', save_plots=True):

    suite = cocoex.Suite(suite_name, "", "")
    scale = 0.5
    fig, axes = plt.subplots(5, 5, figsize=(6.4 * 5 * scale, 4.8 * 5 * scale))

    for ax, func in zip(axes.flatten(), list(range(1, 10)) + [-1] + list(range(10, 25))):
        # Take care of the special case - there are only 4 functions in the second group
        if func == -1:
            ax.set_visible(False)
            continue

        problem = suite.get_problem_by_function_dimension_instance(func, dim, inst)
        best_param = best_parameter(problem)
        problem_name = f'f{func}'

        X, Y, Z = get_landscape(problem, num_intervals=N, best_param=best_param, x_index=x_index, y_index=y_index)

        # Heat map with level sets
        plot = ax.pcolormesh(X, Y, Z, cmap=cmap)
        ax.contour(X, Y, Z, cmap=cmap_lines)
        ax.scatter(best_param[x_index], best_param[y_index], c='white', marker='x', zorder=10)
        ax.set_xlabel(f'x{x_index + 1}')
        ax.set_ylabel(f'x{y_index + 1}', rotation=0)
        ax.set_ylim([-5, 5])
        ax.set_title(problem_name)
        plt.colorbar(plot, ax=ax)

    plt.tight_layout()
    if save_plots:
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plt.savefig(f'{plot_folder}/{suite_name}_all_d{dim}_i{inst:02}_heat.png')
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    plot_all_views(func=22, dim=5, inst=2)
    plot_all_views(func=22, dim=5, inst=3)
    exit(0)
    plot_all_functions(suite_name='bbob', dim=2, inst=1)
    plot_all_functions(suite_name='bbob-mixint', dim=5, inst=1, x_index=3, y_index=4)
    exit(0)
    plot_all_views(func=22, dim=5, inst=1)
    plot_all_views(func=22, dim=3, inst=1)
    plot_all_views(func=22, dim=2, inst=1)

