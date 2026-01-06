import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import proplot as pplt
import warnings

import input_output.utils as io
import list.column_labels as clab
import graphics.utils as gr
import graphics.images as images
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


# READ PARAMETERS FROM CSV
parameters = io.read_parameters_from_csv_file(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.csv")
)

z_min = -0.5
z_max = 0


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


data_z_ode = pd.read_csv(os.path.join(solution_ode_path, "z_ode.csv"))


fig = pplt.figure(figsize=np.array(parameters['figure_size']),
                  left=parameters['figure_margin'][0][0],
                  right=parameters['figure_margin'][0][1],
                  bottom=parameters['figure_margin'][1][1],
                  top=parameters['figure_margin'][1][0],
                  wspace=parameters['wspace'],
                  hspace=parameters['hspace'])

# create axes
# 2d axes
fig.add_subplot(1, 1, 1)


def plot_snapshot(fig, azimuth, altitude):

    ################## 2d plots ################

    # =============
    # z_r subplot
    # =============

    ax = fig.axes[0]  # Use the existing axis
    ax.set_axis_off()

    data_z_ode["r"] = np.sqrt(
        data_z_ode[clab.label_x_column] ** 2 + data_z_ode[clab.label_y_column] ** 2)

    # Sort data by r for a proper plot
    data_z_ode_sorted = data_z_ode.sort_values(by="r")

    # Plot z vs r
    ax.plot(data_z_ode_sorted["r"], data_z_ode_sorted[clab.label_z_column], linestyle="-", linewidth=2, color="red",
            zorder=1,
            label=r' ')

    gr.plot_2d_axes(ax, [0, z_min], [parameters['R'], z_max - z_min],
                    margin=parameters['axis_margin_2d'],
                    tick_length=parameters['tick_length_2d'],
                    line_width=parameters['axis_line_width_2d'],
                    axis_label=parameters['axis_label_2d_z'],
                    axis_origin=parameters['axis_origin_2d'],
                    axis_label_offset=parameters['axis_label_offset_2d'],
                    tick_label_format=parameters['tick_label_format_2d'],
                    plot_label_font_size=parameters['plot_label_font_size'],
                    font_size=parameters['font_size_2d'],
                    tick_label_offset=parameters['tick_label_offset_2d'],
                    plot_label=r'$\textbf{C}$',
                    n_minor_ticks=parameters['n_minor_ticks_2d'],
                    minor_tick_length=parameters['minor_tick_length_2d'],
                    axis_label_angle=parameters['axis_label_angle_2d'],
                    plot_label_offset=parameters['plot_label_offset_2d'])


plot_snapshot(fig, parameters['azimuth'], parameters['altitude'])

plt.savefig(figure_path + "_large.pdf")
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf'
)
