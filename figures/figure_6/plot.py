import matplotlib
from matplotlib.patches import Arc
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings

import calculus.geometry as geo
import calculus.utils as cal
import constants.utils as const
import graphics.utils as gr
import graphics.vector_plot as vp
import input_output.utils as io
import list.column_labels as clab
import list.utils as lis
import system.paths as paths
import system.utils as sys_utils

'''
Parameter meaning: 
- solution_stride: the stride with which data were saved as during the solution of the finite-element problem
- animation_stride: the stride with which frames will be read by animate.py to generate the animation
'''

matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of


# add the path where to find the shared modules
module_path = paths.root_path + "/figures/modules/"
sys.path.append(module_path)

# Suppress the specific warning
warnings.filterwarnings(
    "ignore", message=".*Z contains NaN values.*", category=UserWarning)
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

parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))

solution_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "mesh/solution/")
figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")

snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_path)
number_of_frames = snapshot_max - snapshot_min + 1


# labels of columns to read
columns_line_vertices = [clab.label_start_x_column, clab.label_start_y_column, clab.label_start_z_column,
                         clab.label_end_x_column,
                         clab.label_end_y_column, clab.label_end_z_column]
columns_theta_omega = ["theta", "omega"]

data_theta_omega = pd.read_csv(
    solution_path + 'theta_omega.csv', usecols=columns_theta_omega)


data_boundary_vertices_ellipse = pd.read_csv(os.path.join(
    mesh_path, 'boundary_points_id_' + str(parameters['ellipse_loop_id']) + '.csv'))


fig = pplt.figure(figsize=parameters['figure_size'], left=parameters['figure_margin_l'],
                  bottom=parameters['figure_margin_b'], right=parameters['figure_margin_r'],
                  top=parameters['figure_margin_t'], wspace=0, hspace=0)

# pre-create subplots and axes
fig.add_subplot(2, 1, 1)
fig.add_subplot(2, 1, 2)

data_line_vertices_ref = pd.read_csv(
    os.path.join(mesh_path, 'line_vertices.csv'))


def plot_snapshot(fig, n_file,
                  snapshot_label='',
                  norm_v_min_max=None,
                  sigma_min_max=None):

    n_snapshot = str(n_file)
    data_line_vertices_curr = pd.read_csv(
        solution_path + 'snapshots/csv/line_mesh_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_u = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/u_n_' +
                         n_snapshot + '.csv')

    # plot the ellipse focus
    focal_point_position = [
        parameters['c'][0] - np.sqrt(parameters['a']**2-parameters['b']**2), parameters['c'][1]]
    theta_1 = min(0, data_theta_omega.loc[int(
        n_file/parameters['solution_frame_stride']-1), 'theta'])
    theta_2 = max(0, data_theta_omega.loc[int(
        n_file/parameters['solution_frame_stride']-1), 'theta'])

    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1],
             snapshot_label, fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')

    # =============
    # reference configuration  subplot
    # =============

    ax = fig.axes[0]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)

    gr.plot_2d_mesh(ax, data_line_vertices_ref,
                    line_width=parameters['mesh_line_width_ref_plot'],
                    color='black',
                    alpha=parameters['alpha_mesh'],
                    zorder=1)

    # run through points in data_boundary_vertices_ellipse (reference configuration) and add to them [U_interp_x, U_interp_y] in order to obtain the boundary polygon in the current configuration
    data_ref_boundary_vertices_ellipse = []
    for _, row in data_boundary_vertices_ellipse.iterrows():
        data_ref_boundary_vertices_ellipse.append(
            [row[':0'], row[':1']]
        )

    # plot the boundary polygon in the current configuration
    poly = Polygon(data_ref_boundary_vertices_ellipse, fill=True,
                   linewidth=parameters['mesh_line_width_ref_plot'], edgecolor='red', facecolor='white', zorder=1)
    ax.add_patch(poly)

    # plot the focal point
    ax.scatter(focal_point_position[0], focal_point_position[1],
               color=parameters['ellipse_focal_point_color'], s=parameters['ellipse_focal_point_size'], zorder=2)

    #
    # 1) plot fixed axis
    ax.plot(
        [focal_point_position[0], focal_point_position[0] +
            parameters['ellipse_angle_axis_length']],
        [focal_point_position[1]] * 2,
        color=parameters['ellipse_angle_axis_color'],
        linewidth=parameters['mesh_line_width_curr_plot'],
        linestyle='--',
        zorder=const.high_z_order
    )

    gr.plot_2d_axes(ax, [0, 0], [parameters['L'], parameters['h']],
                    tick_length=parameters['tick_length'],
                    line_width=parameters['axis_line_width'],
                    axis_label=[r'$x \, [\met]$', r'$y \, [\met]$'],
                    tick_label_format=['f', 'f'],
                    font_size=[parameters['font_size'],
                               parameters['font_size']],
                    tick_label_offset=parameters['tick_label_offset'],
                    axis_label_offset=parameters['axis_label_offset'],
                    axis_origin=parameters['axis_origin'],
                    plot_label=parameters["ref_plot_panel_label"],
                    plot_label_offset=parameters['curr_plot_panel_label_position'],
                    plot_label_font_size=parameters['panel_label_font_size'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length'],
                    tick_label_angle=parameters['tick_label_angle'])

    # =============
    # current configuration subplot
    # =============

    ax = fig.axes[1]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)

    gr.plot_2d_mesh(ax, data_line_vertices_curr,
                    line_width=parameters['mesh_line_width_curr_plot'],
                    color='black',
                    alpha=parameters['alpha_mesh'])

    # plot the focal point
    ax.scatter(focal_point_position[0], focal_point_position[1],
               color=parameters['ellipse_focal_point_color'], s=parameters['ellipse_focal_point_size'])

    # plot the axes that define the angle theta
    # 1) plot fixed axis
    ax.plot(
        [focal_point_position[0], focal_point_position[0] +
            parameters['ellipse_angle_axis_length']],
        [focal_point_position[1]] * 2,
        color=parameters['ellipse_angle_axis_color'],
        linewidth=parameters['mesh_line_width_curr_plot'],
        linestyle='--',
        zorder=const.high_z_order
    )
    # 2) plot moving axis
    delta = np.dot(
        gr.R_2d(data_theta_omega.loc[int(
            n_file/parameters['solution_frame_stride']-1), 'theta']),
        [parameters['ellipse_angle_axis_length'], 0]
    )

    ax.plot(
        [focal_point_position[0], focal_point_position[0] + delta[0]],
        [focal_point_position[1], focal_point_position[1] + delta[1]],
        color=parameters['ellipse_angle_axis_color'],
        linewidth=parameters['mesh_line_width_curr_plot'],
        linestyle='--',
        zorder=const.high_z_order
    )

    theta_arc = Arc(focal_point_position,
                    theta1=const.rad_to_deg * theta_1,
                    theta2=const.rad_to_deg * theta_2,
                    width=parameters['ellipse_angle_axis_length'],
                    height=parameters['ellipse_angle_axis_length'],
                    color=parameters['ellipse_angle_axis_color'],
                    zorder=const.high_z_order
                    )

    ax.add_patch(theta_arc)

    gr.plot_2d_axes(ax, [0, 0], [parameters['L'], parameters['h']],
                    tick_length=parameters['tick_length'],
                    line_width=parameters['axis_line_width'],
                    axis_label=[r'$x \, [\met]$', r'$y \, [\met]$'],
                    tick_label_format=['f', 'f'],
                    font_size=[parameters['font_size'],
                               parameters['font_size']],
                    tick_label_offset=parameters['tick_label_offset'],
                    axis_label_offset=parameters['axis_label_offset'],
                    axis_origin=parameters['axis_origin'],
                    plot_label=parameters["curr_plot_panel_label"],
                    plot_label_offset=parameters['curr_plot_panel_label_position'],
                    plot_label_font_size=parameters['panel_label_font_size'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length'],
                    tick_label_angle=parameters['tick_label_angle'])


plot_snapshot(fig, snapshot_max)

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
