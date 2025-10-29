import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings

import list.column_labels as clab
import graphics.utils as gr
import graphics.vector_plot as vec
import input_output.utils as io
import system.paths as paths
import graphics.vector_plot as vp

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


# LOAD PARAMETERS FROM CSV

print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))

parameters = io.read_parameters_from_csv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))

solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")


# labels of columns to read
columns_line_vertices = [clab.label_start_x_column, clab.label_start_y_column, clab.label_start_z_column,
                         clab.label_end_x_column,
                         clab.label_end_y_column, clab.label_end_z_column]
columns_v = [clab.label_x_column, clab.label_y_column, clab.label_v_column + clab.label_x_column,
             clab.label_v_column + clab.label_y_column]
columns_theta_omega = ["theta", "omega"]

data_theta_omega = pd.read_csv(solution_path + 'theta_omega.csv', usecols=columns_theta_omega)


fig = pplt.figure(figsize=parameters['figure_size'], left=parameters['figure_margin_l'], 
                  bottom=parameters['figure_margin_b'], right=parameters['figure_margin_r'], 
                  top=parameters['figure_margin_t'], wspace=0, hspace=0)


def plot_column(fig, n_file):
    n_snapshot = str(n_file)
    data_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_v = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/def_v_n_' + n_snapshot + '.csv', usecols=columns_v)

    ax = fig.add_subplot(1, 1, 1)

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    gr.set_2d_axes_limits(ax, [0, 0], [parameters['L'], parameters['h']], [0, 0])

    gr.plot_2d_mesh(ax, data_line_vertices, 0.1, 'black', parameters['alpha_mesh'])

    X, Y, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, norm_v = vp.interpolate_2d_vector_field(data_v,
                                                                                                    [0, 0],
                                                                                                    [parameters['L'], parameters['h']],
                                                                                                    parameters['n_bins_v'],
                                                                                                    clab.label_x_column,
                                                                                                    clab.label_y_column,
                                                                                                    clab.label_v_column)

    # set to nan the values of the velocity vector field which lie within the elliipse at step 'n_file', where I read the rotation angle of the ellipse from data_theta_omega
    gr.set_inside_ellipse(X, Y, parameters['c'], parameters['a'], parameters['b'], data_theta_omega.loc[n_file-1, 'theta'], V_x, np.nan)
    gr.set_inside_ellipse(X, Y, parameters['c'], parameters['a'], parameters['b'], data_theta_omega.loc[n_file-1, 'theta'], V_y, np.nan)

    vec.plot_2d_vector_field(ax, [X, Y], [V_x, V_y], parameters['arrow_length'], 0.3, 30, 1, 1, 'color_from_map', 0)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min, norm_v_max, parameters['color_bar_position'], parameters['color_bar_size'], 
                        label=r'$z \, [\mic]$', 
                        font_size=parameters['colorbar_font_size'],
                        tick_length=parameters['colorbar_tick_length'],
                        label_pad=parameters['colorbar_label_offset'], 
                        tick_label_offset=parameters['colorbar_tick_label_offset'],
                        tick_label_angle=parameters['colorbar_tick_label_angle'],
                        line_width=parameters['colorbar_line_width'])

    gr.plot_2d_axes(ax, [0, 0], [parameters['L'], parameters['h']],     
                          tick_length=[0.05, 0.05], 
                          line_width=parameters['axis_line_width'], 
                          axis_label=[r'$x \, [\met]$', r'$y \, [\met]$'],
                          tick_label_format=['f', 'f'], 
                          font_size=[parameters['font_size'], parameters['font_size']],
                          tick_label_offset=parameters['tick_label_offset'],
                          axis_label_offset=parameters['axis_label_offset'],
                          axis_origin=parameters['axis_origin'])


# fork
# 1) to plot the figure
# plot_column(fig, parameters['n_early_snapshot'])
# plot_column(fig, parameters['n_late_snapshot'])

# 2) to plot the animation
plot_column(fig, parameters['n_early_snapshot'])

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
