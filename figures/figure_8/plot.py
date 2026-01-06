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
solution_parameters = io.read_parameters_from_csv_file(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "solution_parameters.csv")
)


# define the folder where to read the data
print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))
solution_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "solution/"
)

figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), parameters['figure_name'])


data_theta = pd.read_csv(os.path.join(solution_path, "theta_omega.csv"))

# add step number column
data_theta['t'] = (data_theta.index + 1) * \
    solution_parameters['print_out_stride']/solution_parameters['T']


fig = pplt.figure(figsize=np.array(parameters['figure_size']),
                  left=parameters['figure_margin'][0][0],
                  right=parameters['figure_margin'][0][1],
                  bottom=parameters['figure_margin'][1][1],
                  top=parameters['figure_margin'][1][0],
                  wspace=parameters['wspace'],
                  hspace=parameters['hspace'])

# create axes
fig.add_subplot(1, 1, 1)


def plot(fig):

    # =============
    # theta plot
    # =============

    ax = fig.axes[0]  # Use the existing axis
    ax.set_axis_off()

    # data_theta["r"] = np.sqrt(
    #     data_theta[clab.label_x_column] ** 2 + data_theta[clab.label_y_column] ** 2)

    # Sort data by r for a proper plot
    # data_z_ode_sorted = data_theta.sort_values(by="r")

    # Plot theta vs t
    ax.plot(data_theta['t'], data_theta['theta'],
            linestyle="-",
            linewidth=2,
            color="red",
            zorder=1,
            label=r' '
            )

    t_max = np.max(data_theta['t'])

    theta_min = np.min(data_theta['theta'])
    theta_max = np.max(data_theta['theta'])

    gr.plot_2d_axes(ax, [0, theta_min], [t_max - 0, theta_max-theta_min],
                    margin=parameters['axis_margin'],
                    tick_length=parameters['tick_length'],
                    line_width=parameters['axis_line_width'],
                    axis_label=parameters['axis_label'],
                    axis_origin=parameters['axis_origin'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    plot_label_font_size=parameters['plot_label_font_size'],
                    font_size=parameters['font_size'],
                    tick_label_offset=parameters['tick_label_offset'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length'],
                    axis_label_angle=parameters['axis_label_angle'],
                    plot_label_offset=parameters['plot_label_offset'])


plot(fig)

plt.savefig(figure_path + "_large.pdf")
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf'
)
