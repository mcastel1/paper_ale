import matplotlib.pyplot as plt
import os

# import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings

import list.column_labels as clab
import graphics.utils as gr
import system.paths as paths

# import vector_plot as vec

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
solution_path = paths.root_path + "figures/figure_13/solution/"
snapshot_path = solution_path + "snapshots/csv/"

figure_name = 'figure_13'
L = 1.0
h = 0.25

alpha_mesh = 1
n_ticks_colorbar = 3
margin = 0.2

n_bins_v = [40, 20]
# n_bins_v = [3, 3]

n_ticks = 4
font_size = 8
n_early_snapshot = 1
n_late_snapshot = 10
arrow_length = 0.025

compression_density = 1000
compression_quality = 60
# CHANGE PARAMETERS HERE

fig, ax = plt.subplots(figsize=(3, 2))

# fig = pplt.figure(figsize=(8, 2), left=10, bottom=0, right=10, top=0, wspace=0, hspace=0)



def plot_column(ax, n_file):
    n_snapshot = str(n_file)
    data_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_n_' + n_snapshot + '.csv')
    # data_v = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/def_v_n_' + n_snapshot + '.csv', usecols=columns_v)

    ax.set_axis_off()

    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    gr.set_2d_axes_limits(ax, [0, -h * ( 1 + margin)], [L * ( 1 + margin), h * ( 1 + margin)], [0, 0])

    gr.plot_2d_mesh(ax, data_line_vertices, 0.5, 'black', alpha_mesh)


# fork:  to plot the figure
# plot_column(ax, n_early_snapshot)
# plot_column(ax, n_late_snapshot)

# fork : to plot the animation
plot_column(ax, n_early_snapshot)

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_name + '_large.pdf')
# os.system(f'magick -density {compression_density} {figure_name}_large.pdf -quality {compression_quality} -compress JPEG {figure_name}.pdf')

# pplt.show()
