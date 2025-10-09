import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings



import list.column_labels as clab
import graphics.utils as gr
import system.paths as paths
import graphics.vector_plot as vec

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
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure_1')
snapshot_path = os.path.join(solution_path, "snapshots/csv/")



# define the folder where to read the data
L = 2.2
h = 0.41
c = [0.25, 0.2, 0]
a = 0.1
b = 0.05

alpha_mesh = 1
n_ticks_colorbar = 3

n_bins_v = [40, 20]
# n_bins_v = [3, 3]

n_ticks = 4
font_size = 8
n_early_snapshot = 1
n_late_snapshot = 1358
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
columns_theta_omega = ["theta", "omega"]

data_theta_omega = pd.read_csv(solution_path + 'theta_omega.csv', usecols=columns_theta_omega)


fig = pplt.figure(figsize=(8, 2), left=10, bottom=0, right=10, top=0, wspace=0, hspace=0)


def plot_column(fig, n_file):
    n_snapshot = str(n_file)
    data_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_v = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/def_v_n_' + n_snapshot + '.csv', usecols=columns_v)

    ax = fig.add_subplot(1, 1, 1)

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    gr.set_2d_axes_limits(ax, [0, 0], [L, h], [0, 0])

    gr.plot_2d_mesh(ax, data_line_vertices, 0.1, 'black', alpha_mesh)

    X, Y, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, norm_v = gr.vp.interpolate_2d_vector_field(data_v,
                                                                                                    [0, 0],
                                                                                                    [L, h],
                                                                                                    n_bins_v,
                                                                                                    clab.label_x_column,
                                                                                                    clab.label_y_column,
                                                                                                    clab.label_v_column)

    # set to nan the values of the velocity vector field which lie within the elliipse at step 'n_file', where I read the rotation angle of the ellipse from data_theta_omega
    gr.set_inside_ellipse(X, Y, c, a, b, data_theta_omega.loc[n_file-1, 'theta'], V_x, np.nan)
    gr.set_inside_ellipse(X, Y, c, a, b, data_theta_omega.loc[n_file-1, 'theta'], V_y, np.nan)

    vec.plot_2d_vector_field(ax, [X, Y], [V_x, V_y], arrow_length, 0.3, 30, 1, 1, 'color_from_map', 0)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min, norm_v_max, \
                        1, [0.025, 0.2], [0.02, 0.4], \
                        90, [0, 0], r'$v \, []$', font_size)

    gr.plot_2d_axes_label(ax, [0, 0], [L, h], \
                          0.05, 0.05, 0.3, \
                          r'$x \, [\met]$', r'$y \, [\met]$', 0, 90, \
                          0.4, 0.1, 0.15, 0.05, 'f', 'f', \
                          font_size, font_size, 0, r'', [0, 0])


# fork:  to plot the figure
# plot_column(fig, n_early_snapshot)
# plot_column(fig, n_late_snapshot)

# fork : to plot the animation
plot_column(fig, n_early_snapshot)

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {compression_density} {figure_path}_large.pdf -quality {compression_quality} -compress JPEG {figure_path}.pdf')

# pplt.show()
