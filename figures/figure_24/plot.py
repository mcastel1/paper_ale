import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import ultraplot as pplt
import sys
import warnings

import calculus.utils as cal
import constants.utils as const
import graphics.color_bar as cb
import graphics.utils as gr
import graphics.vector_plot as vp
import input_output.utils as io
import list.utils as lis
import preambles.utils as pre
import system.paths as paths
import system.utils as sys_utils

'''
you can copy the data from abacus with
    REMOTE_PATH="paper_ale_figure_5_pp"
    FIGURE_NAME="figure_24"
    cd /Users/michelecastellana/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME
    rm -rf solution mesh
    mkdir solution
    mkdir -p mesh/solution
    ../copy_from_abacus.sh $REMOTE_PATH/solution/snapshots/csv/  'line_mesh_n_*' 'u_n_*' 'U_n_12_*' 'X_n_12_*' 'X_ref_n_*'  'boundary_points_id_2_n_*' 'f_el_n_12_*' 'f_lap_n_12_*' 'f_vis_n_n_*' 'f_fl_n_12_*' ~/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME 1 14000 100
    mv $REMOTE_PATH/solution .
    rm -rf $REMOTE_PATH
    rsync -avr mcastel1@abacus:$REMOTE_PATH/solution/solution_metadata.csv /Users/michelecastellana/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME/solution
    rsync -avr mcastel1@abacus:$REMOTE_PATH/mesh/solution/mesh_metadata.csv /Users/michelecastellana/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME/mesh/solution


to copy the parameters to finite_elements:

    FIGURE_NAME="figure_24"
    cp ~/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME/solution_parameters.csv ~/Documents/finite_elements/fluid_structure_interaction/membrane/parameters_bc_square_no_circle_line_a.csv
    cp ~/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME/mesh_parameters.csv ~/Documents/finite_elements/generate_mesh/2d/square_no_circle/line/mesh_parameters.csv 
    

'''
matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of

# Show all rows and columns when printing a Pandas array
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))



preamble = pre.figure_preamble([
            paths.definitions_path, 
            paths.latex_module_definitions_path,
            paths.ale_definitions_path,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 
            '../../definitions.tex')
        ])

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
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": preamble,
    "text.usetex": True,
    "text.latex.preamble": preamble,
})

# 1. read solution from local folder
solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")

'''
# 2 read solution from external folder
solution_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/fluid_structure_interaction/membrane', "solution")
mesh_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/generate_mesh/2d/square_no_circle/line', "solution") 

'''
solution_parameters = io.read_parameters_from_csv_file(os.path.join(solution_path, 'solution_metadata.csv'))
mesh_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_path, 'mesh_metadata.csv'))


figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")
snapshot_nodal_values_path = os.path.join(snapshot_path, "nodal_values")

# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_path)
number_of_frames = snapshot_max - snapshot_min + 1


if parameters['mode'] == 'animation':
    # I am plotting the animaition -> select min and max as the global min and max across files

    # f_el
    data_f_el = pd.read_csv(os.path.join(snapshot_path, f'f_el_n_12_{parameters['colorbar_f_el_sample_snapshot']}.csv'))
    interior = (data_f_el.sort_values(':0'))['f'].iloc[parameters['f_el_margin']:-parameters['f_el_margin']]
    f_el_min_max = [interior.min(), interior.max()]


    # f_lap
    f_lap_min_max = cal.min_max_file(
                       os.path.join(solution_path + f'snapshots/csv/f_lap_n_12_{parameters['colorbar_f_lap_sample_snapshot']}.csv')
                       )

    # f_fl
    f_fl_min_max = cal.min_max_file(
                       os.path.join(solution_path + f'snapshots/csv/f_fl_n_12_{parameters["colorbar_f_fl_sample_snapshot"]}.csv')
                       )
   
    #f_vis
    f_vis_min_max = cal.min_max_file(
                       os.path.join(solution_path + f'snapshots/csv/f_vis_n_n_{parameters["colorbar_f_vis_sample_snapshot"]}.csv')
                       )



sys_utils.check_strides(parameters['frame_stride'], solution_parameters['print_out_stride'])

fig = pplt.figure(
    figsize=(parameters['figure_size'][0], parameters['figure_size'][1]),
    left=parameters['figure_margin_l'],
    bottom=parameters['figure_margin_b'],
    right=parameters['figure_margin_r'],
    top=parameters['figure_margin_t'],
    wspace=parameters['wspace'],
    hspace=parameters['hspace'])

# pre-create subplots and axes
fig.add_subplot(2, 2, 1)
fig.add_subplot(2, 2, 2)
fig.add_subplot(2, 2, 3)
fig.add_subplot(2, 2, 4)


f_el_colorbar_axis = fig.add_axes(const.default_axis_position_size)
cb.set_size(f_el_colorbar_axis, parameters['colorbar_size'])

f_lap_colorbar_axis = fig.add_axes(const.default_axis_position_size)
cb.set_size(f_lap_colorbar_axis, parameters['colorbar_size'])

f_vis_colorbar_axis = fig.add_axes(const.default_axis_position_size)
cb.set_size(f_vis_colorbar_axis, parameters['colorbar_size'])

f_fl_colorbar_axis = fig.add_axes(const.default_axis_position_size)
cb.set_size(f_fl_colorbar_axis, parameters['colorbar_size'])


# 
# compute the min and max of the axes
#
data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + str(snapshot_max) + '.csv'))

_, Y_msh_ref, _, u_msh_n_Y, _, _, _, _ = vp.interpolate_2d_vector_field(data_u_msh,
                                                                                        [0, 0],
                                                                                        [mesh_parameters['L'],
                                                                                            np.max(data_u_msh[':1'])],
                                                                                        parameters['n_bins_u'])

# Y are the positions of the mesh nodes in the current configuration
Y = np.array(lis.add_lists_of_lists(Y_msh_ref, u_msh_n_Y))

h = lis.min_max(Y)[1]



def plot_snapshot(fig, n_file,
                  snapshot_label='',
                  axis_min_max=None):

    n_file_string = str(n_file)

    global f_el_min_max, f_fl_min_max, f_vis_min_max, f_lap_min_max


    # load data
    # data_el_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_el_n_' + str(n_file) + '.csv')
    data_msh_line_vertices = pd.read_csv(os.path.join(
        snapshot_path, 'line_mesh_n_' + n_file_string + '.csv'))
    
    data_X_ref = pd.read_csv(os.path.join(snapshot_path, 'X_ref_n_' + n_file_string + '.csv'))
    data_X_ref = data_X_ref.sort_values(by=[":0"]).copy()

    data_U = pd.read_csv(os.path.join(snapshot_path, 'U_n_12_' + n_file_string + '.csv'))
    data_U = data_U.sort_values(by=[":0"]).copy()

    data_f_vis = pd.read_csv(os.path.join(snapshot_path, 'f_vis_n_n_' + n_file_string + '.csv'))
    data_f_vis = data_f_vis.sort_values(by=[":0"]).copy()

    data_f_fl = pd.read_csv(os.path.join(snapshot_path, 'f_fl_n_12_' + n_file_string + '.csv'))
    data_f_fl = data_f_fl.sort_values(by=[":0"]).copy()

    data_f_el = pd.read_csv(os.path.join(snapshot_path, 'f_el_n_12_' + n_file_string + '.csv'))
    data_f_el = data_f_el.sort_values(by=[":0"]).copy()

    data_f_lap = pd.read_csv(os.path.join(snapshot_path, 'f_lap_n_12_' + n_file_string + '.csv'))
    data_f_lap = data_f_lap.sort_values(by=[":0"]).copy()

  
    data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + n_file_string + '.csv'))

    # build `data_X_cur` from `data_X_ref` and `data_U`
    data_X_cur = data_X_ref.copy()
    data_X_cur[['f:0', 'f:1']] += data_U[['f:0', 'f:1']]
    data_X_cur = data_X_cur.sort_values(by=[":0"]).copy()

    # build `data_X_0` (initial shape of the membrane) from `data_X_ref` 
    data_X_0 = data_X_ref.copy()
    data_X_0['f:1'] = mesh_parameters['shape_coordinates'][0][1]
    data_X_0 = data_X_0.sort_values(by=[":0"]).copy()




    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1],
             snapshot_label, fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')

    if axis_min_max == None:

        # compute the min and max of the axes
        data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + str(n_file) + '.csv'))
        
        axis_min_max = [[0, mesh_parameters['L']], [0, np.max(data_u_msh[':1'] + data_u_msh['f:1'])]]




    # if f_el_min_max == None:
    #     # f_el_min_max = cal.min_max_file(os.path.join(snapshot_path, 'f_el_n_12_' + str(n_file) + '.csv'))
    #     interior = data_f_el.sort_values(':0')['f'].iloc[parameters['f_el_margin']:-parameters['f_el_margin']]
    #     f_el_min_max = [interior.min(), interior.max()]

    # if f_lap_min_max == None:
    #     f_lap_min_max = cal.min_max_file(os.path.join(snapshot_path, 'f_lap_n_12_' + str(n_file) + '.csv'))

    # if f_fl_min_max == None:
    #     f_fl_min_max = cal.min_max_file(os.path.join(snapshot_path, 'f_fl_n_12_' + str(n_file) + '.csv'))

    # if f_vis_min_max == None:
    #     f_vis_min_max = cal.min_max_file(os.path.join(snapshot_path, 'f_vis_n_n_' + str(n_file) + '.csv'))

    if parameters['mode'] == 'figure':
    # I am plotting the figure -> select min and max according to the current frame 

        # f_el
        interior = (data_f_el.sort_values(':0'))['f'].iloc[parameters['f_el_margin']:-parameters['f_el_margin']]
        f_el_min_max = [interior.min(), interior.max()]

        # f_lap
        f_lap_min_max = cal.min_max_file(os.path.join(snapshot_path, f'f_lap_n_12_{n_file}.csv'))

        # f_fl
        f_fl_min_max = cal.min_max_file(os.path.join(snapshot_path, f'f_fl_n_12_{n_file}.csv'))

        # f_vis
        f_vis_min_max = cal.min_max_file(os.path.join(snapshot_path, f'f_vis_n_n_{n_file}.csv'))
   





    X_cur, text = gr.interpolate_curve(
        data_X_cur, data_X_cur[':0'].min(), data_X_cur[':0'].max(), parameters['n_bins_X'])
    
    X_0, _ = gr.interpolate_curve(
        data_X_0, data_X_0[':0'].min(), data_X_0[':0'].max(), parameters['n_bins_X'])
    

    # =============
    # f_el subplot
    # =============

    ax = fig.axes[0]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax,[0, 0], [mesh_parameters['L'], h])



    color_map_f_el = gr.cb.make_curve_colorbar(fig, text, data_f_el,
                                             min_max=f_el_min_max,
                                             tick_label_angle=parameters['f_el_colorbar_tick_label_angle'],
                                             label=parameters['f_el_colorbar_axis_label'],
                                             font_size=parameters['colorbar_font_size'],
                                             tick_label_offset=parameters['colorbar_tick_label_offset'],
                                             label_angle=parameters['f_el_colorbar_label_angle'],
                                             tick_length=parameters['colorbar_tick_length'],
                                             label_offset=parameters['colorbar_axis_label_offset'],
                                             tick_n_bins=parameters['colorbar_tick_n_bins'],
                                             tick_label_format=parameters['f_el_colorbar_tick_label_format'],
                                             axis=f_el_colorbar_axis)

    # plot X and nu
    gr.plot_curve_grid(ax, X_cur,
                       color_map=color_map_f_el,
                       line_color='black',
                       line_width=parameters['f_el_line_width'])

    # plot X_0
    gr.plot_curve_grid(ax, X_0,
                       line_color=parameters['X_0_color'],
                       legend=parameters['X_0_legend'],
                       line_width=parameters['X_line_width'],
                       alpha=parameters['X_0_alpha'],
                       z_order=0
                       )

    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    line_width=parameters['plot_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh_low'],
                    zorder=parameters['mesh_zorder'])

    gr.plot_2d_axes(
        ax, [0, 0], [mesh_parameters['L'], h],
        tick_length=parameters['tick_length'],
        tick_n_bins=parameters['tick_n_bins'],
        line_width=parameters['axis_line_width'],
        axis_label=parameters['axis_label_cur'],
        axis_label_angle=parameters['axis_label_angle'],
        tick_label_angle=parameters['tick_label_angle'],
        axis_label_offset=parameters['axis_label_offset'],
        tick_label_offset=parameters['tick_label_offset'],
        tick_label_format=['f', 'f'],
        font_size=parameters['axis_font_size'],
        plot_label=parameters["f_el_panel_label"],
        plot_label_offset=parameters['panel_label_offset'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order,
        colorbar_axis=f_el_colorbar_axis,
        colorbar_axis_offset=parameters['colorbar_offset'])

    
    # =============
    # f_lap subplot
    # =============

    ax = fig.axes[1]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax,[0, 0], [mesh_parameters['L'], h])

    color_map_f_lap = gr.cb.make_curve_colorbar(fig, text, data_f_lap,
                                              min_max=f_lap_min_max,
                                              tick_label_angle=parameters['f_lap_colorbar_tick_label_angle'],
                                              label=parameters['f_lap_colorbar_axis_label'],
                                              font_size=parameters['colorbar_font_size'],
                                              tick_label_offset=parameters['colorbar_tick_label_offset'],
                                              label_angle=parameters['f_lap_colorbar_label_angle'],
                                              tick_length=parameters['colorbar_tick_length'],
                                              label_offset=parameters['colorbar_axis_label_offset'],
                                              tick_n_bins=parameters['colorbar_tick_n_bins'],
                                              tick_label_format=parameters['f_lap_colorbar_tick_label_format'],
                                              axis=f_lap_colorbar_axis)

    # plot X and psi
    gr.plot_curve_grid(ax, X_cur,
                       color_map=color_map_f_lap,
                       line_color='black',
                       line_width=parameters['f_lap_line_width'])

    # plot X_0
    gr.plot_curve_grid(ax, X_0,
                       line_color=parameters['X_0_color'],
                       legend=parameters['X_0_legend'],
                       line_width=parameters['X_line_width'],
                       alpha=parameters['X_0_alpha'],
                       z_order=0
                       )

    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    line_width=parameters['plot_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh_low'],
                    zorder=parameters['mesh_zorder'])

    gr.plot_2d_axes(
        ax, [0, 0], [mesh_parameters['L'], h],
        tick_length=parameters['tick_length'],
        tick_n_bins=parameters['tick_n_bins'],
        line_width=parameters['axis_line_width'],
        axis_label=parameters['axis_label_cur'],
        axis_label_angle=parameters['axis_label_angle'],
        tick_label_angle=parameters['tick_label_angle'],
        axis_label_offset=parameters['axis_label_offset'],
        tick_label_offset=parameters['tick_label_offset'],
        tick_label_format=['f', 'f'],
        font_size=parameters['axis_font_size'],
        plot_label=parameters["f_lap_panel_label"],
        plot_label_offset=parameters['panel_label_offset'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order,
        colorbar_axis=f_lap_colorbar_axis,
        colorbar_axis_offset=parameters['colorbar_offset'])


    
    # =============
    # f_eta subplot
    # =============

    ax = fig.axes[2]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax, [0, 0], [mesh_parameters['L'], h])

    color_map_w = gr.cb.make_curve_colorbar(fig, text, data_f_vis,
                                            min_max=f_vis_min_max,
                                            tick_label_angle=parameters['f_vis_colorbar_tick_label_angle'],
                                            label=parameters['f_vis_colorbar_axis_label'],
                                            font_size=parameters['colorbar_font_size'],
                                            tick_label_offset=parameters['colorbar_tick_label_offset'],
                                            label_angle=parameters['f_vis_colorbar_label_angle'],
                                            tick_length=parameters['colorbar_tick_length'],
                                            label_offset=parameters['colorbar_axis_label_offset'],
                                            tick_n_bins=parameters['colorbar_tick_n_bins'],
                                            tick_label_format=parameters['f_vis_colorbar_tick_label_format'],
                                            axis=f_vis_colorbar_axis)

    # plot X and w
    gr.plot_curve_grid(ax, X_cur,
                       color_map=color_map_w,
                       line_color='black',
                       line_width=parameters['f_vis_line_width'])

    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    line_width=parameters['plot_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh_low'],
                    zorder=parameters['mesh_zorder'])

    # plot X_0
    gr.plot_curve_grid(ax, X_0,
                       line_color=parameters['X_0_color'],
                       legend=parameters['X_0_legend'],
                       line_width=parameters['X_line_width'],
                       alpha=parameters['X_0_alpha'],
                       z_order=0
                       )

    gr.plot_2d_axes(
        ax, [0, 0], [mesh_parameters['L'], h],
        tick_length=parameters['tick_length'],
        tick_n_bins=parameters['tick_n_bins'],
        line_width=parameters['axis_line_width'],
        axis_label=parameters['axis_label_cur'],
        axis_label_angle=parameters['axis_label_angle'],
        tick_label_angle=parameters['tick_label_angle'],
        axis_label_offset=parameters['axis_label_offset'],
        tick_label_offset=parameters['tick_label_offset'],
        tick_label_format=['f', 'f'],
        font_size=parameters['axis_font_size'],
        plot_label=parameters["f_vis_panel_label"],
        plot_label_offset=parameters['panel_label_offset'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order,
        colorbar_axis=f_vis_colorbar_axis,
        colorbar_axis_offset=parameters['colorbar_offset'])

    
    # =============
    # f_fl subplot
    # =============

    ax = fig.axes[3]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax, [0, 0], [mesh_parameters['L'], h])

    color_map_f_fl = gr.cb.make_curve_colorbar(fig, text, data_f_fl,
                                                min_max=f_fl_min_max,
                                                tick_label_angle=parameters['f_fl_colorbar_tick_label_angle'],
                                                label=parameters['f_fl_colorbar_axis_label'],
                                                font_size=parameters['colorbar_font_size'],
                                                tick_label_offset=parameters['colorbar_tick_label_offset'],
                                                tick_label_format=parameters['f_fl_colorbar_tick_label_format'],
                                                label_angle=parameters['f_fl_colorbar_label_angle'],
                                                tick_length=parameters['colorbar_tick_length'],
                                                label_offset=parameters['colorbar_axis_label_offset'],
                                                tick_n_bins=parameters['colorbar_tick_n_bins'],
                                                axis=f_fl_colorbar_axis)

    # plot X and sigma
    gr.plot_curve_grid(ax, X_cur,
                       color_map=color_map_f_fl,
                       line_color='black',
                       line_width=parameters['f_fl_line_width'])

    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    line_width=parameters['plot_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh_low'],
                    zorder=parameters['mesh_zorder'])

    # plot X_0
    gr.plot_curve_grid(ax, X_0,
                       line_color=parameters['X_0_color'],
                       legend=parameters['X_0_legend'],
                       line_width=parameters['X_line_width'],
                       alpha=parameters['X_0_alpha'],
                       z_order=0
                       )

    gr.plot_2d_axes(
        ax, [0, 0], [mesh_parameters['L'], h],
        tick_length=parameters['tick_length'],
        tick_n_bins=parameters['tick_n_bins'],
        line_width=parameters['axis_line_width'],
        axis_label=parameters['axis_label_cur'],
        axis_label_angle=parameters['axis_label_angle'],
        tick_label_angle=parameters['tick_label_angle'],
        axis_label_offset=parameters['axis_label_offset'],
        tick_label_offset=parameters['tick_label_offset'],
        tick_label_format=['f', 'f'],
        font_size=parameters['axis_font_size'],
        plot_label=parameters["f_fl_panel_label"],
        plot_label_offset=parameters['panel_label_offset'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order,
        colorbar_axis=f_fl_colorbar_axis,
        colorbar_axis_offset=parameters['colorbar_offset'])

    
     


plot_snapshot(fig, snapshot_max,
              snapshot_label=rf'$t = \,$' + io.time_to_string(snapshot_max * solution_parameters['T'] / solution_parameters['N'], 's', 0))
# plot_snapshot(fig, parameters['snapshot_to_plot'],
#               snapshot_label=rf'$t = \,$' + io.time_to_string(parameters['snapshot_to_plot'] * solution_parameters['T'] / solution_parameters['N'], 'min_s', 0))


if parameters['mode'] == 'figure':

    plt.savefig(figure_path + '_large.pdf')
    os.system(
        f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

