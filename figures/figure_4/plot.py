'''
copy data for this plot with

./copy_from_abacus.sh line_1/solution/snapshots/csv/  'X_n_12_*' 'v_n_*' 'w_n_*' 'sigma_n_12_*' 'nu_n_12_*' 'psi_n_12_*' ~/Documents/paper_ale/figures/figure_4 1 1000000 1000

'''
import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings

import calculus.utils as cal
import graphics.utils as gr
import graphics.vector_plot as vp
import list.utils as lis
import input_output.utils as io
import system.paths as paths
import system.utils as sys_utils

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
solution_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "solution/")
# solution_path = "/Users/michelecastellana/Documents/finite_elements/dynamics/lagrangian_approach/one_dimension/solution/"
mesh_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "mesh/solution/")
# mesh_path = "/Users/michelecastellana/Documents/finite_elements/generate_mesh/1d/line/solution/"
figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'figure_4')
snapshot_path = os.path.join(solution_path, "snapshots/csv/nodal_values/")

parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))

# labels of columns to read
columns_X = ["f:0", "f:1", "f:2", ":0", ":1", ":2"]
columns_v = ["f:0", "f:1", "f:2", ":0", ":1", ":2"]
columns_sigma = ["f", ":0", ":1", ":2"]
columns_nu = ["f", ":0", ":1", ":2"]
columns_psi = ["f", ":0", ":1", ":2"]

snapshot_min, snapshot_max = sys_utils.n_min_max('X_n_12_', snapshot_path)
# number_of_frames = sys_utils.count_v_files('X_n_12_', pfig.snapshot_path)
number_of_frames = snapshot_max - snapshot_min + \
    1  # +1 because the frames start from 0

snapshot_max_with_margin = snapshot_max - parameters['snapshot_max_margin']


fig = pplt.figure(
    figsize=(parameters['figure_size'][0], parameters['figure_size'][1]),
    left=parameters['figure_margin'][0][0],
    right=parameters['figure_margin'][0][1],
    top=parameters['figure_margin'][1][0],
    bottom=parameters['figure_margin'][1][1],
    wspace=parameters['wspace'],
    hspace=parameters['hspace'])


# pre-create subplots and axes
fig.add_subplot(3, 2, 1)
fig.add_subplot(3, 2, 2)
fig.add_subplot(3, 2, 3)
fig.add_subplot(3, 2, 4)
fig.add_subplot(3, 2, 5)
fig.add_subplot(3, 2, 6)

nu_colorbar_axis = fig.add_axes([parameters['nu_colorbar_position'][0],
                                parameters['nu_colorbar_position'][1],
                                parameters['nu_colorbar_size'][0],
                                parameters['nu_colorbar_size'][1]])

psi_colorbar_axis = fig.add_axes([parameters['psi_colorbar_position'][0],
                                  parameters['psi_colorbar_position'][1],
                                  parameters['psi_colorbar_size'][0],
                                  parameters['psi_colorbar_size'][1]])

v_colorbar_axis = fig.add_axes([parameters['v_colorbar_position'][0],
                                parameters['v_colorbar_position'][1],
                                parameters['v_colorbar_size'][0],
                                parameters['v_colorbar_size'][1]])


w_colorbar_axis = fig.add_axes([parameters['w_colorbar_position'][0],
                                parameters['w_colorbar_position'][1],
                                parameters['w_colorbar_size'][0],
                                parameters['w_colorbar_size'][1]])

sigma_colorbar_axis = fig.add_axes([parameters['sigma_colorbar_position'][0],
                                    parameters['sigma_colorbar_position'][1],
                                    parameters['sigma_colorbar_size'][0],
                                    parameters['sigma_colorbar_size'][1]])


def plot_snapshot(fig, n_file,
                  snapshot_label='',
                  X_min_max=None,
                  nu_min_max=None,
                  psi_min_max=None,
                  norm_v_min_max=None,
                  sigma_min_max=None,
                  w_min_max=None):

    n_snapshot = str(n_file)
    data_X = pd.read_csv(os.path.join(
        snapshot_path, 'X_n_12_' + n_snapshot + '.csv'), usecols=columns_X)
    data_nu = pd.read_csv(os.path.join(
        snapshot_path, 'nu_n_12_' + n_snapshot + '.csv'))
    data_psi = pd.read_csv(os.path.join(
        snapshot_path, 'psi_n_12_' + n_snapshot + '.csv'))
    data_sigma = pd.read_csv(os.path.join(
        snapshot_path, 'sigma_n_12_' + n_snapshot + '.csv'))
    data_w = pd.read_csv(os.path.join(
        snapshot_path, 'w_n_' + n_snapshot + '.csv'))
    data_v = pd.read_csv(os.path.join(
        snapshot_path, 'v_n_' + n_snapshot + '.csv'), usecols=columns_v)
    # data_omega contains de values of \partial_1 X^alpha
    data_omega = lis.data_omega(data_nu, data_psi)

    if X_min_max == None:
        X_min_max = [
            cal.min_max_file(os.path.join(
                snapshot_path, 'X_n_12_' + str(n_file) + '.csv'), column_name='f:0'),
            cal.min_max_file(os.path.join(
                snapshot_path, 'X_n_12_' + str(n_file) + '.csv'), column_name='f:1')
        ]
    if nu_min_max == None:
        nu_min_max = cal.min_max_file(os.path.join(
            snapshot_path, 'nu_n_12_' + str(n_file) + '.csv'))
    if psi_min_max == None:
        psi_min_max = cal.min_max_file(os.path.join(
            snapshot_path, 'psi_n_12_' + str(n_file) + '.csv'))
    if norm_v_min_max == None:
        norm_v_min_max = cal.norm_min_max_file(os.path.join(
            snapshot_path, 'v_n_' + str(n_file) + '.csv'))
    if sigma_min_max == None:
        sigma_min_max = cal.min_max_file(os.path.join(
            snapshot_path, 'sigma_n_12_' + str(n_file) + '.csv'))
    if w_min_max == None:
        w_min_max = cal.min_max_file(os.path.join(
            snapshot_path, 'w_n_' + str(n_file) + '.csv'))

    X_curr, t = gr.interpolate_curve(
        data_X, X_min_max[0][0], X_min_max[0][1], parameters['n_bins_X'])

    # plot snapshot label
    if snapshot_label != '':
        fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1],
                 snapshot_label, fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')

    # =============
    # u subplot
    # =============

    ax = fig.axes[0]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    # setting the axes limits here is necessary because some methods will be called before plot_2d_axes, and these methods will need the axes limits to be properly set to place, for example, text labels
    gr.set_2d_axes_limits(ax,
                          [X_min_max[0][0], X_min_max[1][0]],
                          [X_min_max[0][1] - X_min_max[0][0],
                              X_min_max[1][1] - X_min_max[1][0]],
                          axis_origin=parameters['axis_origin']
                          )
    # compute the vector field u and store it in U_x, U_y and its related coordinates X_u, Y_u in the current configuration
    U_x = []
    U_y = []
    X_u = []
    Y_u = []
    for _, row in data_X.iterrows():

        X_u.append(row[':0'])
        Y_u.append(0)

        U_x.append(row['f:0'] - row[':0'])
        U_y.append(row['f:1'] - 0)

    # Convert to numpy arrays
    X_u = np.array(X_u)
    Y_u = np.array(Y_u)
    U_x = np.array(U_x)
    U_y = np.array(U_y)

    # coordinates of the curve in the reference configuration
    X_ref = np.array(list(zip(X_u, Y_u)))

    # plot the vector field u
    vp.plot_1d_vector_field(ax, [X_u, Y_u], [U_x, U_y],
                            shaft_length=None,
                            head_length=parameters['u_arrow_head_length'],
                            head_angle=parameters['head_angle'],
                            line_width=parameters['u_arrow_line_width'],
                            alpha=parameters['alpha'],
                            color=parameters['u_arrow_color'],
                            threshold_arrow_length=parameters['threshold_arrow_length'],
                            legend='$\\vec{U}$',
                            legend_font_size=8,
                            legend_arrow_length=0.15,
                            legend_text_arrow_space=0.2,
                            legend_head_over_shaft_length=parameters['legend_head_over_shaft_length'],
                            legend_position=[-0.525, 0.71],
                            z_order=1)

    # plot X_curr
    gr.plot_curve_grid(ax, X_curr,
                       line_color='green',
                       legend='$\\text{Current}$',
                       legend_position=[-0.55, 0.9],
                       legend_inner_location='upper left',
                       line_width=parameters['X_curr_line_width'],
                       z_order=0
                       )

    # plot X_ref
    gr.plot_curve_grid(ax, X_ref,
                       line_color='red',
                       legend='$\\text{Reference}$',
                       legend_position=[-0.55, 1],
                       legend_inner_location='upper left',
                       line_width=parameters['X_ref_line_width'],
                       z_order=0
                       )

    gr.plot_2d_axes(ax,
                    [X_min_max[0][0], X_min_max[1][0]],
                    [X_min_max[0][1] - X_min_max[0][0],
                     X_min_max[1][1] - X_min_max[1][0]],
                    axis_origin=parameters['axis_origin'],
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width'],
                    tick_length=parameters['tick_length'],
                    plot_label=parameters['u_plot_label'],
                    plot_label_offset=parameters['plot_label_offset'],
                    plot_label_font_size=parameters['plot_label_font_size'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                    )

    # =============
    # nu subplot
    # =============

    ax = fig.axes[1]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    # setting the axes limits here is necessary because some methods will be called before plot_2d_axes, and these methods will need the axes limits to be properly set to place, for example, text labels
    gr.set_2d_axes_limits(ax,
                          [X_min_max[0][0], X_min_max[1][0]],
                          [X_min_max[0][1] - X_min_max[0][0],
                              X_min_max[1][1] - X_min_max[1][0]],
                          axis_origin=parameters['axis_origin']
                          )

    color_map_nu = gr.cb.make_curve_colorbar(fig, t, data_nu, parameters['nu_colorbar_position'], parameters['nu_colorbar_size'],
                                             min_max=nu_min_max,
                                             tick_label_angle=parameters['nu_colorbar_tick_label_angle'],
                                             label=r'$\nu$',
                                             font_size=parameters['color_map_font_size'],
                                             label_offset=parameters["colorbar_label_offset"],
                                             tick_label_offset=parameters['nu_colorbar_tick_label_offset'],
                                             tick_label_format=parameters['nu_colorbar_tick_label_format'],
                                             label_angle=parameters['nu_colorbar_label_angle'],
                                             line_width=parameters['colorbar_tick_line_width'],
                                             tick_length=parameters['nu_colorbar_tick_length'],
                                             axis=nu_colorbar_axis)

    # plot X_curr and w
    gr.plot_curve_grid(ax, X_curr,
                       color_map=color_map_nu,
                       line_color='black',
                       line_width=parameters['w_line_width']
                       )

    gr.plot_2d_axes(ax,
                    [X_min_max[0][0], X_min_max[1][0]],
                    [X_min_max[0][1] - X_min_max[0][0],
                        X_min_max[1][1] - X_min_max[1][0]],
                    axis_origin=parameters['axis_origin'],
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    font_size=parameters['font_size'],
                    plot_label=parameters['w_plot_label'],
                    plot_label_offset=parameters['plot_label_offset'],
                    plot_label_font_size=parameters['plot_label_font_size'],
                    line_width=parameters['axis_line_width'],
                    tick_length=parameters['tick_length'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                    )

    # =============
    # psi subplot
    # =============

    ax = fig.axes[2]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    # setting the axes limits here is necessary because some methods will be called before plot_2d_axes, and these methods will need the axes limits to be properly set to place, for example, text labels
    gr.set_2d_axes_limits(ax,
                          [X_min_max[0][0], X_min_max[1][0]],
                          [X_min_max[0][1] - X_min_max[0][0],
                              X_min_max[1][1] - X_min_max[1][0]],
                          axis_origin=parameters['axis_origin']
                          )

    color_map_psi = gr.cb.make_curve_colorbar(fig, t, data_psi, parameters['psi_colorbar_position'], parameters['psi_colorbar_size'],
                                              min_max=psi_min_max,
                                              tick_label_angle=parameters['psi_colorbar_tick_label_angle'],
                                              label=r'$\psi$',
                                              font_size=parameters['color_map_font_size'],
                                              label_offset=parameters["colorbar_label_offset"],
                                              tick_label_offset=parameters['psi_colorbar_tick_label_offset'],
                                              tick_label_format=parameters['psi_colorbar_tick_label_format'],
                                              label_angle=parameters['psi_colorbar_label_angle'],
                                              line_width=parameters['colorbar_tick_line_width'],
                                              tick_length=parameters['psi_colorbar_tick_length'],
                                              axis=psi_colorbar_axis)

    # plot X_curr and psi
    gr.plot_curve_grid(ax, X_curr,
                       color_map=color_map_psi,
                       line_color='black',
                       line_width=parameters['w_line_width']
                       )

    gr.plot_2d_axes(ax,
                    [X_min_max[0][0], X_min_max[1][0]],
                    [X_min_max[0][1] - X_min_max[0][0],
                        X_min_max[1][1] - X_min_max[1][0]],
                    axis_origin=parameters['axis_origin'],
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    font_size=parameters['font_size'],
                    plot_label=parameters['w_plot_label'],
                    plot_label_offset=parameters['plot_label_offset'],
                    plot_label_font_size=parameters['plot_label_font_size'],
                    line_width=parameters['axis_line_width'],
                    tick_length=parameters['tick_length'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                    )

    # =============
    # v subplot
    # =============

    ax = fig.axes[3]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    # setting the axes limits here is necessary because some methods will be called before plot_2d_axes, and these methods will need the axes limits to be properly set to place, for example, text labels
    gr.set_2d_axes_limits(ax,
                          [X_min_max[0][0], X_min_max[1][0]],
                          [X_min_max[0][1] - X_min_max[0][0],
                              X_min_max[1][1] - X_min_max[1][0]],
                          axis_origin=parameters['axis_origin']
                          )

    # plot X_curr
    gr.plot_curve_grid(ax, X_curr,
                       line_color='black',
                       line_width=parameters['X_dummy_line_width'],
                       alpha=parameters['alpha_X']
                       )

    # plot v
    X_v, Y_v, V_x, V_y, grid_norm_v, _, _, _ = vp.interpolate_t_vector_field_2d_arc_length_gauge(
        data_X, data_omega, data_v, parameters['n_bins_v'])

    vp.plot_1d_vector_field(ax, [X_v, Y_v], [V_x, V_y],
                            shaft_length=parameters['shaft_length'],
                            head_over_shaft_length=parameters['v_head_over_shaft_length'],
                            head_angle=parameters['head_angle'],
                            line_width=parameters['v_arrow_line_width'],
                            alpha=parameters['alpha'],
                            color='color_from_map',
                            threshold_arrow_length=parameters['threshold_arrow_length'])

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min_max[0], norm_v_min_max[1],
                        position=parameters['v_colorbar_position'],
                        size=parameters['v_colorbar_size'],
                        label_pad=parameters['colorbar_label_offset'],
                        label=r'$v \, [\met / \sec]$',
                        label_angle=parameters['v_colorbar_label_angle'],
                        font_size=parameters['color_map_font_size'],
                        tick_label_offset=parameters['v_colorbar_tick_label_offset'],
                        tick_label_angle=parameters['v_colorbar_tick_label_angle'],
                        axis=v_colorbar_axis,
                        tick_label_format=parameters['v_colorbar_tick_label_format'],
                        tick_length=parameters['v_colorbar_tick_length'],
                        line_width=parameters['colorbar_tick_line_width'])

    gr.plot_2d_axes(ax,
                    [X_min_max[0][0], X_min_max[1][0]],
                    [X_min_max[0][1] - X_min_max[0][0],
                        X_min_max[1][1] - X_min_max[1][0]],
                    axis_origin=parameters['axis_origin'],
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width'],
                    tick_length=parameters['tick_length'],
                    plot_label=parameters['v_plot_label'],
                    plot_label_offset=parameters['plot_label_offset'],
                    plot_label_font_size=parameters['plot_label_font_size'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                    )

    # =============
    # w subplot
    # =============

    ax = fig.axes[4]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    # setting the axes limits here is necessary because some methods will be called before plot_2d_axes, and these methods will need the axes limits to be properly set to place, for example, text labels
    gr.set_2d_axes_limits(ax,
                          [X_min_max[0][0], X_min_max[1][0]],
                          [X_min_max[0][1] - X_min_max[0][0],
                              X_min_max[1][1] - X_min_max[1][0]],
                          axis_origin=parameters['axis_origin']
                          )

    color_map_w = gr.cb.make_curve_colorbar(fig, t, data_w, parameters['w_colorbar_position'], parameters['w_colorbar_size'],
                                            min_max=w_min_max,
                                            tick_label_angle=parameters['w_colorbar_tick_label_angle'],
                                            label=r'$w \, [\met/\sec]$',
                                            font_size=parameters['color_map_font_size'],
                                            label_offset=parameters["colorbar_label_offset"],
                                            tick_label_offset=parameters['w_colorbar_tick_label_offset'],
                                            tick_label_format=parameters['w_colorbar_tick_label_format'],
                                            label_angle=parameters['w_colorbar_label_angle'],
                                            line_width=parameters['colorbar_tick_line_width'],
                                            tick_length=parameters['w_colorbar_tick_length'],
                                            axis=w_colorbar_axis)

    # plot X_curr and w
    gr.plot_curve_grid(ax, X_curr,
                       color_map=color_map_w,
                       line_color='black',
                       line_width=parameters['w_line_width']
                       )

    gr.plot_2d_axes(ax,
                    [X_min_max[0][0], X_min_max[1][0]],
                    [X_min_max[0][1] - X_min_max[0][0],
                        X_min_max[1][1] - X_min_max[1][0]],
                    axis_origin=parameters['axis_origin'],
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    font_size=parameters['font_size'],
                    plot_label=parameters['w_plot_label'],
                    plot_label_offset=parameters['plot_label_offset'],
                    plot_label_font_size=parameters['plot_label_font_size'],
                    line_width=parameters['axis_line_width'],
                    tick_length=parameters['tick_length'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                    )

    # =============
    # sigma subplot
    # =============

    ax = fig.axes[5]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    # setting the axes limits here is necessary because some methods will be called before plot_2d_axes, and these methods will need the axes limits to be properly set to place, for example, text labels
    gr.set_2d_axes_limits(ax,
                          [X_min_max[0][0], X_min_max[1][0]],
                          [X_min_max[0][1] - X_min_max[0][0],
                              X_min_max[1][1] - X_min_max[1][0]],
                          axis_origin=parameters['axis_origin']
                          )

    color_map_sigma = gr.cb.make_curve_colorbar(fig, t, data_sigma, parameters['sigma_colorbar_position'], parameters['sigma_colorbar_size'],
                                                min_max=sigma_min_max,
                                                tick_label_angle=parameters['sigma_colorbar_tick_label_angle'],
                                                label=r'$\sigma \, [\newt/\met]$',
                                                font_size=parameters['color_map_font_size'],
                                                label_offset=parameters["colorbar_label_offset"],
                                                tick_label_format=parameters['sigma_colorbar_tick_label_format'],
                                                tick_label_offset=parameters['sigma_colorbar_tick_label_offset'],
                                                label_angle=parameters['sigma_colorbar_label_angle'],
                                                line_width=parameters['colorbar_tick_line_width'],
                                                tick_length=parameters['sigma_colorbar_tick_length'],
                                                axis=sigma_colorbar_axis)

    # plot X and sigma
    gr.plot_curve_grid(ax, X_curr,
                       color_map=color_map_sigma,
                       line_color='black',
                       line_width=parameters['sigma_line_width'])

    gr.plot_2d_axes(ax,
                    [X_min_max[0][0], X_min_max[1][0]],
                    [X_min_max[0][1] - X_min_max[0][0],
                        X_min_max[1][1] - X_min_max[1][0]],
                    axis_origin=parameters['axis_origin'],
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    plot_label=parameters['sigma_plot_label'],
                    plot_label_offset=parameters['plot_label_offset'],
                    plot_label_font_size=parameters['plot_label_font_size'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width'],
                    tick_length=parameters['tick_length'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                    )


plot_snapshot(fig,
              snapshot_max_with_margin,
              rf'$t = \,$' + io.time_to_string(snapshot_max_with_margin *
                                               parameters['T'] / number_of_frames, 'min_s', parameters['snapshot_label_decimals'])
              )


# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')
