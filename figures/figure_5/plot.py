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
import input_output.utils as io
import list.utils as lis
import system.paths as paths
import graphics.vector_plot as vec

matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of


figure_name = 'figure_5'

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

print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))
solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), figure_name)
snapshot_path = os.path.join(solution_path, "snapshots/csv/")
snapshot_nodal_values_path = os.path.join(snapshot_path, "nodal_values")

parameters = io.read_parameters_from_csv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))   


# CHANGE PARAMETERS HERE
L = 1
h = 1
T = 1e-3
number_of_frames = 1000

alpha_mesh = 1
n_ticks_colorbar = 3
margin = 0.2


n_ticks = 4


compression_density = 1000
compression_quality = 60
# CHANGE PARAMETERS HERE


# labels of columns to read
columns_line_vertices = [clab.label_start_x_column, clab.label_start_y_column, clab.label_start_z_column,
                         clab.label_end_x_column,
                         clab.label_end_y_column, clab.label_end_z_column]
columns_v = [clab.label_x_column, clab.label_y_column, clab.label_v_column + clab.label_x_column,
             clab.label_v_column + clab.label_y_column]

fig = pplt.figure(
    figsize=(parameters['figure_size'][0], parameters['figure_size'][1]), 
    left=parameters['figure_margin_l'], 
    bottom=parameters['figure_margin_b'], 
    right=parameters['figure_margin_r'], 
    top=parameters['figure_margin_t'], 
    wspace=0, hspace=0)


def plot_snapshot(fig, n_file, snapshot_label):
    n_snapshot = str(n_file)

    # load data
    # data_el_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_el_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_msh_line_vertices = pd.read_csv(os.path.join(snapshot_path, 'line_mesh_n_' + n_snapshot + '.csv'), usecols=columns_line_vertices)
    data_v = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'def_v_fl_n_' + n_snapshot + '.csv'), usecols=columns_v)
    data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + n_snapshot + '.csv'), usecols=columns_v)


    ax = fig.add_subplot(1, 1, 1)

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # plot snapshot label
    fig.text(0.55, 0.85, snapshot_label, fontsize=parameters['plot_label_font_size'], ha='center', va='center')

    gr.set_2d_axes_limits(ax, [0, 0], [L, h], [0, 0])


    # plot mesh under the membrane
    # gr.plot_2d_mesh(ax, data_el_line_vertices, 0.2, 'red', alpha_mesh)
    gr.plot_2d_mesh(ax, data_msh_line_vertices, parameters['plot_line_width'], 'black', alpha_mesh)

    
    # here X_ref, Y_ref are the coordinates of the points in the reference configuration of the mesh
    X_ref, Y_ref, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, norm_v = vec.interpolate_2d_vector_field(data_v,
                                                                                                    [0, 0],
                                                                                                    [L, h],
                                                                                                    parameters['n_bins_v'],
                                                                                                    clab.label_x_column,
                                                                                                    clab.label_y_column,
                                                                                                    clab.label_v_column)
    
    print(f'data_v = {data_v}\n Y={Y_ref}')

    
    X_ref, Y_ref, u_n_X, u_n_Y, grid_norm_u_n, norm_u_n_min, norm_u_n_max, norm_u_n = vec.interpolate_2d_vector_field(data_u_msh,
                                                                                                                    [0, 0],
                                                                                                                    [L, h],
                                                                                                                    parameters['n_bins_v'],
                                                                                                                    clab.label_x_column,
                                                                                                                    clab.label_y_column,
                                                                                                                    clab.label_v_column)

    #X, Y are the positions of the mesh nodes in the current configuration    
    X = np.array(lis.add_lists_of_lists(X_ref, u_n_X))
    Y = np.array(lis.add_lists_of_lists(Y_ref, u_n_Y))

    

    # plot velocity of fluid
    vec.plot_2d_vector_field(ax, [X_ref, Y_ref], [V_x, V_y], parameters['arrow_length'], 0.3, 30, 0.5, 1, 'color_from_map', 0)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min, norm_v_max, \
                        1, [0.05, 0.3], [0.01, 0.3], \
                        90, [-3.0, 0.5], r'$v \, [\met/\sec]$', parameters['colorbar_font_size'])
                        
    

    gr.plot_2d_axes_label(ax, [0, 0], [L, h], \
                          parameters['tick_length'], parameters['axis_line_width'], \
                          parameters['axis_labels'], parameters['axis_label_angle'], \
                          parameters['axis_label_offset'], parameters['tick_label_offset'], ['f', 'f'], \
                          parameters['axis_font_size'], parameters['plot_label_font_size'], 
                          0, r'', parameters['plot_label_offset'], margin=parameters['margin'], axis_origin=parameters['axis_origin'])



plot_snapshot(fig, parameters['n_late_snapshot'], rf'$t = \,$' + io.time_to_string(parameters['n_late_snapshot'] * T / number_of_frames, 's', 0))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {compression_density} {figure_path}_large.pdf -quality {compression_quality} -compress JPEG {figure_path}.pdf')

# pplt.show()
