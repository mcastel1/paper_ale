import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import proplot as pplt
import warnings

import graphics.color_bar as cb
import constants.utils as const
import input_output.utils as io
import list.column_labels as clab
import graphics.utils as gr
import system.paths as paths

matplotlib.use(
    "Agg"
)  # use a non-interactive backend so that figures do not pop up during run in debug mode

# Suppress the specific warning
warnings.filterwarnings(
    "ignore", message=".*Z contains NaN values.*", category=UserWarning)
# clean the matplotlib cache to load the correct version of definitions.tex
os.system(" rm -rf ~/.matplotlib/tex.cache")

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{newpxtext,newpxmath} "
        r"\usepackage{xcolor} "
        r"\usepackage{glossaries} "
        rf"\input{{{paths.definitions_path}}}"
    )
})


parameters = io.read_parameters_from_csv_file(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.csv")
)
mesh_parameters = io.read_parameters_from_csv_file(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_parameters.csv")
)


# define the folder where to read the data
print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))
solution_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "solution/nodal_values/"
)
solution_ode_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "solution-ode"
)

mesh_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "mesh/solution/")
figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), parameters['figure_name'])


# labels of columns to read
data_line_vertices = pd.read_csv(os.path.join(mesh_path, "line_vertices.csv"))


fig = pplt.figure(figsize=np.array(parameters['figure_size']),
                  left=parameters['figure_margin'][0][0],
                  right=parameters['figure_margin'][0][1],
                  bottom=parameters['figure_margin'][1][1],
                  top=parameters['figure_margin'][1][0],
                  wspace=parameters['wspace'],
                  hspace=parameters['hspace'])

# create axes
# 3d axes
fig.add_subplot(1, 1, 1, projection="3d", auto_add_to_figure=False)

def plot_snapshot(fig, azimuth, altitude):

    # =============
    # mesh plot
    # =============

    ax = fig.axes[0]  # Use the existing axis

    # ax.set_box_aspect([parameters['L'], parameters['h'], z_max - z_min])
    gr.empty_panes(ax)
    ax.set_axis_off()
    ax.view_init(elev=altitude, azim=azimuth)

    gr.plot_mesh(ax, data_line_vertices,
                 parameters['mesh_line_width'], 'black', parameters['alpha_mesh'])


    gr.plot_3d_axes(ax, [-mesh_parameters['r'], -mesh_parameters['r'], -mesh_parameters['r']], [2*mesh_parameters['r'], 2*mesh_parameters['r'], 2*mesh_parameters['r']],
                    scale_factor=[1, 1, parameters['scale_factor_z']],
                    axis_origin=parameters['axis_origin_3d'],
                    axis_label=parameters['axis_label_3d'],
                    axis_label_offset=parameters['axis_label_offset_3d'],
                    tick_label_offset=parameters['tick_label_offset_3d'],
                    tick_label_format=parameters['tick_label_format_3d'],
                    tick_length=parameters['tick_length_3d'],
                    minor_tick_length=parameters['minor_tick_length_3d'],
                    n_minor_ticks=parameters['n_minor_ticks_3d'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width_3d'],
                    plot_label=r'$\textbf{B}$',
                    plot_label_position=parameters['plot_label_offset_3d'],
                    plot_label_font_size=parameters['plot_label_font_size'])


plot_snapshot(fig, parameters['azimuth'], parameters['altitude'])

plt.savefig(figure_path + "_large.pdf")
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf'
)
