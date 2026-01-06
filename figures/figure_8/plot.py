import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
from scipy.optimize import curve_fit
import sys
import warnings

import graphics.utils as gr
import graphics.images as im
import list.utils as lis
import input_output.utils as io
import system.paths as paths
import system.utils as sysio
import text.utils as text


matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of


# add the path where to find the shared modules
module_path = paths.root_path + "/figures/modules/"
sys.path.append(module_path)

# Suppress the specific warning
warnings.filterwarnings(
    "ignore", message=".*Z contains NaN values.*", category=UserWarning)
# clean the matplotlib cache to load the correct version of definitions.tex
os.system("rm -rf ~/.matplotlib/tex.cache")

pplt.rc['grid'] = False  # disables default gridlines

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{newpxtext,newpxmath} "
        r"\usepackage{xcolor} "
        r"\usepackage{glossaries} "
        rf"\input{{{paths.definitions_path}}}"
    )
})

parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))


print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))
solution_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "mesh/solution/")
figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/nodal_values/")


fig = pplt.figure(figsize=(parameters['figure_size'][0], parameters['figure_size'][1]),
                  left=parameters['figure_margin_l'],
                  bottom=parameters['figure_margin_b'],
                  right=parameters['figure_margin_r'],
                  top=parameters['figure_margin_t'],
                  wspace=parameters['wspace'],
                  hspace=parameters['hspace'])

fig.add_subplot(1, 1, 1)


# the fitting function for the fit
def fitting_function(x, a, b):
    return a + b * x


'''
err = A * h^B
log err = log A + B log h
'''

data_time = pd.read_csv(os.path.join(solution_path, 'time.csv'))


def plot_panel(field_name, ax):

    ax.set_axis_off()
    ax.grid(False)
    ax.set_box_aspect(parameters['box_aspect'])

    data = {
        'f': data_time[field_name],
        ':0': data_time['num_cells_mesh']
    }

    data_x_for_fit = np.emath.logn(parameters['log_base'], np.array(data[':0']))[
        parameters['index_first_fitted_datum']:]
    data_y_for_fit = np.emath.logn(parameters['log_base'], np.array(data['f']))[
        parameters['index_first_fitted_datum']:]

    F, _ = gr.interpolate_1d_function(data,  parameters['n_bins'])
    [[x_min, x_max], [y_min, y_max]] = lis.min_max_coordinates(F)

    fit = curve_fit(
        fitting_function, data_x_for_fit, data_y_for_fit)

    fit_parameters = fit[0]
    fit_covariance = fit[1]

    print(
        f'{field_name}: \n\tb = {fit_parameters[1]} +- {np.sqrt(fit_covariance[1][1])}')

    if (x_min <= 0) or (x_max <= 0):
        text.print_text_color('Error: x_min / max are not positive!', 'red')

    # plot the interpolated curve which goes through the data points
    ax.plot(np.emath.logn(parameters['log_base'], np.array(data[':0'])),
            fitting_function(np.emath.logn(parameters['log_base'], np.array(data[':0'])), *fit_parameters), '-', color=parameters['line_color'], linewidth=parameters['curve_line_width'], label='')

    # plot the data points
    ax.plot(np.emath.logn(parameters['log_base'], np.array(data[':0'])), np.emath.logn(parameters['log_base'], np.array(
        data['f'])), 'x', color=parameters['dot_color'], linewidth=parameters['curve_line_width'], s=parameters['dot_size'])

    gr.plot_2d_axes(ax, [x_min, y_min], [x_max - x_min, y_max-y_min],
                    scale=['log', 'log'],
                    log_base=[parameters['log_base']] * 2,
                    axis_label=parameters['axis_label'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_length=parameters['tick_length'],
                    minor_tick_length=parameters['minor_tick_length'],
                    axis_origin=parameters['axis_origin'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    axis_label_angle=parameters['axis_label_angle'],
                    line_width=parameters['line_width'],
                    plot_label_offset=parameters['plot_label_offset']
                    )

    # ATTENTION: THE VALUES OF THE AXES MARGINS MAY HIDE SOME PARTS OF THE FIGURE, ADJUST THEM !
    gr.set_2d_axis_limits_margin(
        ax, [[x_min, x_max], [y_min, y_max]],
        margin=parameters['axis_margin'],
        log_base=parameters['log_base']
    )


def plot(fig):

    plot_panel('time', fig.axes[0])


plot(fig)


plt.savefig(figure_path + '_large.pdf')
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')
