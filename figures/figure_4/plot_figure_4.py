import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings


import graphics.graph as gr
import list.list as lis
import input_output.input_output as io
import system.paths as paths
import system.system_io as sysio

matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of


# add the path where to find the shared modules
module_path = paths.root_path + "/figures/modules/"
sys.path.append(module_path)

# Suppress the specific warning
warnings.filterwarnings("ignore", message=".*Z contains NaN values.*", category=UserWarning)
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


# CHANGE PARAMETERS HERE
print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))
solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
# solution_path = "/Users/michelecastellana/Documents/finite_elements/dynamics/lagrangian_approach/one_dimension/solution/"
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")
# mesh_path = "/Users/michelecastellana/Documents/finite_elements/generate_mesh/1d/line/solution/"
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure_4')
snapshot_path = os.path.join(solution_path, "snapshots/csv/nodal_values/")

parameters = io.read_parameters_from_csv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))   



x_min = 0.0
x_max = 1.0
# CHANGE PARAMETERS HERE



# labels of columns to read
columns_X = ["f:0","f:1","f:2",":0",":1",":2"]
columns_v = ["f:0","f:1","f:2",":0",":1",":2"]
columns_sigma = ["f",":0", ":1",":2"]
columns_nu = ["f",":0", ":1",":2"]
columns_psi = ["f",":0", ":1",":2"]

n_min, n_max = sysio.n_min_max('X_n_12_', snapshot_path)
# number_of_frames = sysio.count_v_files('X_n_12_', pfig.snapshot_path)
number_of_frames = n_max-n_min + 1  # +1 because the frames start from 0

# fork
# to plot the figure
# to plot the animation
'''
sigma_min_max = gr.min_max_files('sigma_n_12_', snapshot_path, columns_sigma[0], n_min, n_max, parameters['frame_stride'])
'''

fig = pplt.figure(figsize=(parameters['figure_size'][0], parameters['figure_size'][1]), left=parameters['figure_margin_l'], bottom=parameters['figure_margin_b'], right=parameters['figure_margin_r'], top=parameters['figure_margin_t'], wspace=0, hspace=0)


def plot_column(fig, n_file, sigma_min_max=None):
    
    n_snapshot = str(n_file)
    data_X = pd.read_csv(os.path.join(snapshot_path, 'X_n_12_' + n_snapshot + '.csv'), usecols=columns_X)
    data_nu = pd.read_csv(os.path.join(snapshot_path, 'nu_n_12_' + n_snapshot + '.csv'), usecols=columns_nu)
    data_psi = pd.read_csv(os.path.join(snapshot_path, 'psi_n_12_' + n_snapshot + '.csv'), usecols=columns_psi)
    data_sigma = pd.read_csv(os.path.join(snapshot_path, 'sigma_n_12_' + n_snapshot + '.csv'), usecols=columns_sigma)
    data_v = pd.read_csv(os.path.join(snapshot_path, 'v_n_' + n_snapshot + '.csv'), usecols=columns_v)

    # data_omega contains de values of \partial_1 X^alpha
    data_omega  = lis.data_omega(data_nu, data_psi)

    
    # print(f'data_sigma = {data_sigma}')

    # Check if we already have an axis, if not create one
    if len(fig.axes) == 0:
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.axes[0]  # Use the existing axis
    

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    X, t = gr.interpolate_curve(data_X, x_min, x_max, parameters['n_bins'])


    color_map_sigma = gr.cb.make_curve_colorbar(fig, t, data_sigma,
                                    parameters['color_bar_position'], parameters['sigma_color_bar_size'], parameters['sigma_color_bar_angle'], parameters["sigma_color_bar_label_pad"], 
                                    r'$\sigma \, [\newt/\met]$', parameters['font_size'], sigma_min_max)

    #plot X and sigma 
    gr.plot_curve_grid(ax, X, color_map_sigma, 'black', parameters['X_line_width'])



    # plot v
    X_v, Y_v, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, norm_v = gr.vp.interpolate_t_vector_field_2d_arc_length_gauge(data_X, data_omega, data_v, parameters['n_bins'])
    
    
    gr.vp.plot_1d_vector_field(ax, [X_v, Y_v], [V_x, V_y], 
                               parameters['shaft_length'], parameters['head_over_shaft_length'], parameters['head_angle'], 
                               parameters['line_width'], parameters['alpha'], 'color_from_map', 0)
    
    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min, norm_v_max, \
                        1, parameters['v_color_bar_position'], parameters['v_color_bar_size'], \
                        90, parameters['v_color_bar_label_pad'], r'$v \, []$', parameters['font_size'])


    gr.set_2d_axes_limits(ax, [x_min, parameters['y_min']], [x_max, parameters['y_max']], [0, 0])


    gr.plot_2d_axes_label(ax, [x_min, parameters['y_min']], [x_max-x_min, parameters['y_max']-parameters['y_min']], \
                          0.05, 0.05, 1, \
                          r'$X^1 \, [\met]$', r'$X^2 \, [\met]$', 0, 90, \
                          parameters['axis_label_offset'][0], parameters['axis_label_offset'][1], parameters['ticks_label_offset'][0], parameters['ticks_label_offset'][1], 'f', 'f', \
                          parameters['font_size'], parameters['font_size'], 0, r'', [0, 0])


# fork
# to plot the figure
# 
plot_column(fig, parameters['n_early_snapshot'])
# plot_column(fig, parameters['n_late_snapshot'])
# 
# to plot the animation
'''
plot_column(fig, parameters['n_early_snapshot'])
'''

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
