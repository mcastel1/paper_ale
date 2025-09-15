import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings


import list.column_labels as clab
import graphics.graph as gr
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
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure_4')
snapshot_path = os.path.join(solution_path, "snapshots/csv/nodal_values/")


# define the folder where to read the data
x_min = 0.0
x_max = 1.0
h = 1.0


n_ticks_colorbar = 3

n_bins = 100

n_ticks = 4
font_size = 8
n_early_snapshot = 1
n_late_snapshot = 10
compression_density = 1000
compression_quality = 60
# CHANGE PARAMETERS HERE


# labels of columns to read
columns_line_vertices = ["f:0","f:1","f:2",":0", ":1",":2"]


fig = pplt.figure(figsize=(4, 4), left=5, bottom=5, right=5, top=5, wspace=0, hspace=0)


def plot_column(fig, n_file):
    n_snapshot = str(n_file)
    data_X = pd.read_csv(os.path.join(snapshot_path, 'X_n_12_' + n_snapshot + '.csv'), usecols=columns_line_vertices)

    ax = fig.add_subplot(1, 1, 1)

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid


    X = gr.interpolate_curve(data_X, x_min, x_max, n_bins)


    print(f'X = {X}')

    plt.plot(X[:, 0], X[:, 1], 'b-', linewidth=2, label='Interpolated Curve')


    gr.set_2d_axes_limits(ax, [x_min, -h], [x_max, h], [0, 0])

    '''
    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min, norm_v_max, \
                        1, [0.025, 0.2], [0.02, 0.4], \
                        90, [0, 0], r'$v \, []$', font_size)
    '''

    gr.plot_2d_axes_label(ax, [x_min, -h], [x_max-x_min, 2*h], \
                          0.05, 0.05, 1, \
                          r'$X^1 \, []$', r'$X^2 \, []$', 0, 90, \
                          0.1, 0.1, 0.05, 0.05, 'f', 'f', \
                          font_size, font_size, 0, r'', [0, 0])


# fork:  to plot the figure
plot_column(fig, n_early_snapshot)
# plot_column(fig, n_late_snapshot)

# fork : to plot the animation
# plot_column(fig, n_early_snapshot)

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {compression_density} {figure_path}_large.pdf -quality {compression_quality} -compress JPEG {figure_path}.pdf')

# pplt.show()
