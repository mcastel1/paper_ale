import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings


import graphics.graph as gr
import input_output.input_output as io
import system.paths as paths

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

parameters = io.read_parameters_from_csv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))   

x_min = 0.0
x_max = 1.0
# CHANGE PARAMETERS HERE


# labels of columns to read
columns_X = ["f:0","f:1","f:2",":0", ":1",":2"]
columns_sigma = ["f",":0", ":1",":2"]


fig = pplt.figure(figsize=(6, 3), left=5, bottom=5, right=0, top=5, wspace=0, hspace=0)


def plot_column(fig, n_file):
    n_snapshot = str(n_file)
    data_X = pd.read_csv(os.path.join(snapshot_path, 'X_n_12_' + n_snapshot + '.csv'), usecols=columns_X)
    data_sigma = pd.read_csv(os.path.join(snapshot_path, 'sigma_n_12_' + n_snapshot + '.csv'), usecols=columns_sigma)
    
    sigma_min = np.min(data_sigma['f'])
    sigma_max = np.max(data_sigma['f'])

    print(f'data_sigma = {data_sigma}')

    # Check if we already have an axis, if not create one
    if len(fig.axes) == 0:
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.axes[0]  # Use the existing axis
    

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    X, t = gr.interpolate_curve(data_X, x_min, x_max, parameters['n_bins'])

    print(f'X = {X}')

    color_map = gr.cb.make_curve_colorbar(fig, t, data_sigma, sigma_min, sigma_max, 
                                    [0.1, 0.1], [0.01, 0.1], 90, [0,0], 
                                    r'$\sigma \, []$', parameters['font_size'],)

    gr.plot_curve_grid(ax, X, color_map)


    gr.set_2d_axes_limits(ax, [x_min, -parameters['X2_max']], [x_max, parameters['X2_max']], [0, 0])


    gr.plot_2d_axes_label(ax, [x_min, -parameters['X2_max']], [x_max-x_min, 2*parameters['X2_max']], \
                          0.05, 0.05, 1, \
                          r'$X^1 \, [\met]$', r'$X^2 \, [\met]$', 0, 90, \
                          0.1, 0.1, 0.05, 0.05, 'f', 'f', \
                          parameters['font_size'], parameters['font_size'], 0, r'', [0, 0])


# fork:  to plot the figure
# plot_column(fig, parameters['n_early_snapshot'])
# plot_column(fig, parameters['n_late_snapshot'])

# fork : to plot the animation
plot_column(fig, parameters['n_early_snapshot'])

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
