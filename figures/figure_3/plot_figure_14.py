import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings

import column_labels as clab
import graph as gr
import input_output as io
import list as lis
import paths
import vector_plot as vec

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
# define the folder where to read the data
# solution_path = paths.root_path + "figures/figure_14/solution/"
solution_path = '/Users/michelecastellana/Desktop/elastic_obstacle_2/solution/'
snapshot_path = solution_path + "snapshots/csv/"

figure_name = 'figure_14'
L = 2.2
h = 0.41
c = [0.4, 0.2, 0]
a = 0.2
b = 0.1
r = 0.0125
T = 1.0
number_of_frames = 1000

alpha_mesh = 1
n_ticks_colorbar = 3
margin = 0.2

n_bins_v = [30, 20]
# n_bins_v = [3, 3]

n_ticks = 4
font_size = 8
n_early_snapshot = 1
n_late_snapshot = 10
arrow_length = 0.025

compression_density = 1000
compression_quality = 60
# CHANGE PARAMETERS HERE


# labels of columns to read
columns_line_vertices = [clab.label_start_x_column, clab.label_start_y_column, clab.label_start_z_column,
                         clab.label_end_x_column,
                         clab.label_end_y_column, clab.label_end_z_column]
columns_v = [clab.label_x_column, clab.label_y_column, clab.label_v_column + clab.label_x_column,
             clab.label_v_column + clab.label_y_column]

fig = pplt.figure(figsize=(5, 1.5), left=8, bottom=0, right=2, top=-1, wspace=0, hspace=0)


def plot_snapshot(fig, n_file, snapshot_label):
    n_snapshot = str(n_file)

    # load data
    data_el_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_el_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_msh_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_msh_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_v = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/def_v_n_' + n_snapshot + '.csv', usecols=columns_v)
    data_u_msh = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/u_msh_n_' + n_snapshot + '.csv', usecols=columns_v)

    ax = fig.add_subplot(1, 1, 1)

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # plot snapshot label
    fig.text(0.55, 0.85, snapshot_label, fontsize=8, ha='center', va='center')

    gr.set_2d_axes_limits(ax, [0, 0], [L, h], [0, 0])

    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_el_line_vertices, 0.2, 'red', alpha_mesh)
    gr.plot_2d_mesh(ax, data_msh_line_vertices, 0.05, 'black', alpha_mesh)


    X, Y, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, norm_v = gr.vp.interpolate_2d_vector_field(data_v,
                                                                                                    [0, 0],
                                                                                                    [L, h],
                                                                                                    n_bins_v,
                                                                                                    clab.label_x_column,
                                                                                                    clab.label_y_column,
                                                                                                    clab.label_v_column)

    X_u_msh, Y_u_msh, U_msh_x, U_msh_y, grid_norm_u_msh, norm_u_msh_min, norm_u_msh_max, norm_u_msh = gr.vp.interpolate_2d_vector_field(data_u_msh,
                                                                                                                                        [0, 0],
                                                                                                                                        [L, h],
                                                                                                                                        n_bins_v,
                                                                                                                                        clab.label_x_column,
                                                                                                                                        clab.label_y_column,
                                                                                                                                        clab.label_v_column)

    # set to nan the values of the velocity vector field which lie within the elliipse at step 'n_file', where I read the rotation angle of the ellipse from data_theta_omega
    # 1. obtain the coordinates of the points X, Y of the vector field V_x, V_y in the reference configuration of the mesh
    X_ref = np.array(lis.add_lists_of_lists(X, U_msh_x))
    Y_ref = np.array(lis.add_lists_of_lists(Y, U_msh_y))
    # 2. once the coordinates in the reference configuration are known, assess whether they fall within the elastic body by checking whether they fall wihin the ellipse
    gr.set_inside_ellipse(X_ref, Y_ref, c, a, b, 0, V_x, np.nan)
    gr.set_inside_ellipse(X_ref, Y_ref, c, a, b, 0, V_y, np.nan)

    # plot velocity of fluid
    vec.plot_2d_vector_field(ax, [X, Y], [V_x, V_y], arrow_length, 0.3, 30, 0.5, 1, 'color_from_map', 0)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min, norm_v_max, \
                        1, [0.05, 0.3], [0.01, 0.3], \
                        90, [-3.0, 0.5], r'$v \, [\met/\sec]$', font_size)

    gr.plot_2d_axes_label(ax, [0, 0], [L, h], \
                          0.05, 0.05, 0.3, \
                          r'$x \, [\met]$', r'$y \, [\met]$', 0, 90, \
                          0.1, 0.1, 0.3, 0.05, 'f', 'f', \
                          font_size, font_size, 0, r'', [0, 0])



plot_snapshot(fig, n_early_snapshot, rf'$t = \,$' + io.time_to_string(n_early_snapshot * T / number_of_frames, 's', 0))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_name + '_large.pdf')
os.system(f'magick -density {compression_density} {figure_name}_large.pdf -quality {compression_quality} -compress JPEG {figure_name}.pdf')

# pplt.show()
