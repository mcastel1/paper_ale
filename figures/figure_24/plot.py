import matplotlib
from matplotlib.axes import Axes as MplAxes
from matplotlib.font_manager import FontProperties
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
import list.column_labels as clab
import graphics.utils as gr
import graphics.vector_plot as vp
import input_output.utils as io
import list.utils as lis
import preambles.utils as pre
import system.paths as paths
import system.utils as sys_utils

'''
you can copy the data from abacus with
    REMOTE_PATH="membrane_paper_ale_fig_5_etax10"
    FIGURE_NAME="figure_24"
    cd /Users/michelecastellana/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME
    rm -rf solution mesh
    mkdir solution
    mkdir -p mesh/solution
    ../copy_from_abacus.sh $REMOTE_PATH/solution/snapshots/csv/  'line_mesh_n_*' 'u_n_*' 'U_n_12_*' 'X_n_12_*' 'X_ref_n_*' 'v_n_*' 'w_n_*' 'sigma_n_12_*' 'nu_n_12_*' 'psi_n_12_*' 'def_v_fl_n_*' 'v_fl_n_*'  'sigma_fl_n_*'  'def_sigma_fl_n_*' 'boundary_points_id_2_n_*' 'sigma_n_12_*'  ~/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME 1 14000 100
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

#  read solution from local folder
solution_1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution_1/")
solution_2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution_2/")
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")


solution_1_parameters = io.read_parameters_from_csv_file(os.path.join(solution_1_path, 'solution_metadata.csv'))
solution_2_parameters = io.read_parameters_from_csv_file(os.path.join(solution_2_path, 'solution_metadata.csv'))
mesh_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_path, 'mesh_metadata.csv'))


figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])
snapshot_1_path = os.path.join(solution_1_path, "snapshots/csv/")
snapshot_1_nodal_values_path = os.path.join(snapshot_1_path, "nodal_values")

snapshot_2_path = os.path.join(solution_2_path, "snapshots/csv/")
snapshot_2_nodal_values_path = os.path.join(snapshot_2_path, "nodal_values")

# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_1_path)
number_of_frames = snapshot_max - snapshot_min + 1



sys_utils.check_strides(parameters['frame_stride'], solution_1_parameters['print_out_stride'])

fig = pplt.figure(
    figsize=(parameters['figure_size'][0], parameters['figure_size'][1]),
    left=parameters['figure_margin_l'],
    bottom=parameters['figure_margin_b'],
    right=parameters['figure_margin_r'],
    top=parameters['figure_margin_t'],
    wspace=parameters['wspace'],
    hspace=parameters['hspace'])

# pre-create subplots and axes
fig.add_subplot(1, 2, 1)
fig.add_subplot(1, 2, 2)





# 
# compute the min and max of the axes
#
data_u_msh = pd.read_csv(os.path.join(snapshot_1_nodal_values_path, 'u_n_' + str(snapshot_max) + '.csv'))

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
                  axis_min_max=None,
                  nu_min_max=None,
                  psi_min_max=None,
                  norm_v_fl_min_max=None,
                  sigma_fl_min_max=None,
                  norm_v_min_max=None,
                  w_min_max=None,
                  sigma_min_max=None):

    n_file_string = str(n_file)

    # load data
    # data_el_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_el_n_' + str(n_file) + '.csv')
    data_msh_line_vertices_1 = pd.read_csv(os.path.join(snapshot_1_path, 'line_mesh_n_' + n_file_string + '.csv'))
    
    data_X_ref_1 = pd.read_csv(os.path.join(snapshot_1_path, 'X_ref_n_' + n_file_string + '.csv'))
    data_X_ref_1 = data_X_ref_1.sort_values(by=[":0"]).copy()

    data_U_1 = pd.read_csv(os.path.join(snapshot_1_path, 'U_n_12_' + n_file_string + '.csv'))
    data_U_1 = data_U_1.sort_values(by=[":0"]).copy()

    data_msh_line_vertices_2 = pd.read_csv(os.path.join(snapshot_2_path, 'line_mesh_n_' + n_file_string + '.csv'))
    
    data_X_ref_2 = pd.read_csv(os.path.join(snapshot_2_path, 'X_ref_n_' + n_file_string + '.csv'))
    data_X_ref_2 = data_X_ref_2.sort_values(by=[":0"]).copy()

    data_U_2 = pd.read_csv(os.path.join(snapshot_2_path, 'U_n_12_' + n_file_string + '.csv'))
    data_U_2 = data_U_1.sort_values(by=[":0"]).copy()

   
    data_u_msh_1 = pd.read_csv(os.path.join(snapshot_1_nodal_values_path, 'u_n_' + n_file_string + '.csv'))
    data_u_msh_2 = pd.read_csv(os.path.join(snapshot_2_nodal_values_path, 'u_n_' + n_file_string + '.csv'))


    # build `data_X_cur` from `data_X_ref` and `data_U`
    data_X_cur_1 = data_X_ref_1.copy()
    data_X_cur_1[['f:0', 'f:1']] += data_U_1[['f:0', 'f:1']]
    data_X_cur_1 = data_X_cur_1.sort_values(by=[":0"]).copy()

    # build `data_X_0` (initial shape of the membrane) from `data_X_ref` 
    data_X_0_1 = data_X_ref_1.copy()
    data_X_0_1['f:1'] = mesh_parameters['shape_coordinates'][0][1]
    data_X_0_1 = data_X_0_1.sort_values(by=[":0"]).copy()

    # build `data_X_cur` from `data_X_ref` and `data_U`
    data_X_cur_2 = data_X_ref_2.copy()
    data_X_cur_2[['f:0', 'f:1']] += data_U_2[['f:0', 'f:1']]
    data_X_cur_2 = data_X_cur_2.sort_values(by=[":0"]).copy()

    # build `data_X_0` (initial shape of the membrane) from `data_X_ref` 
    data_X_0_2 = data_X_ref_2.copy()
    data_X_0_2['f:1'] = mesh_parameters['shape_coordinates'][0][1]
    data_X_0_2 = data_X_0_2.sort_values(by=[":0"]).copy()




    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1],
             snapshot_label, fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')

    if axis_min_max == None:

        # compute the min and max of the axes
        #
        data_u_msh_1 = pd.read_csv(os.path.join(snapshot_1_nodal_values_path, 'u_n_' + str(n_file) + '.csv'))

        
        
        axis_min_max = [[0, mesh_parameters['L']], [0, np.max(data_u_msh_1[':1'] + data_u_msh_1['f:1'])]]

        #




      
   
    # =============
    # plot for solution_1
    # =============

    ax = fig.axes[0]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax,[0, 0], [mesh_parameters['L'], h])

    # compute the vector field u and store it in U_x, U_y and its related coordinates X_U, Y_U in the current configuration
    # X_U, Y_U, U_x, U_y = geo.u_1d(data_X_ref, data_U)

    X_cur_1, _ = gr.interpolate_curve(
            data_X_cur_1, data_X_cur_1[':0'].min(), data_X_cur_1[':0'].max(), parameters['n_bins_X'])
        
    X_0_1, _ = gr.interpolate_curve(
            data_X_0_1, data_X_0_1[':0'].min(), data_X_0_1[':0'].max(), parameters['n_bins_X'])
    
    
    X_sym_1 = np.array([[mesh_parameters['L'], - h * parameters['X_sym_margin']], [mesh_parameters['L'], h + h*parameters['X_sym_margin']]])
      



    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices_1,
                    line_width=parameters['mesh_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh_high'],
                    zorder=parameters['mesh_zorder'])
    # plot X_0
    gr.plot_curve_grid(ax, X_0_1,
                       line_color=parameters['X_0_color'],
                       legend=parameters['X_0_legend'],
                       line_width=parameters['X_line_width'],
                       z_order=1
                       )
    
    # plot X_cur
    gr.plot_curve_grid(ax, X_cur_1,
                       line_color=parameters['X_cur_color'],
                       legend=parameters['X_cur_legend'],
                       line_width=parameters['X_line_width'],
                       line_style=parameters['X_cur_line_style'],
                       z_order=1
                       )


    # plot symmetry axis
    gr.plot_curve_grid(ax, X_sym_1,
                       line_color=parameters['X_sym_color'],
                       legend=parameters['X_sym_legend'],
                       line_width=parameters['X_line_width'],
                       line_style=parameters['X_sym_line_style'],
                       alpha=parameters['X_sym_alpha'],
                       z_order=0
                       )


    

    # Create custom legend handles
    handles, labels = ax.get_legend_handles_labels()

    with plt.rc_context({'text.usetex': False}):
        # disable tex otherwise the legend and the line sample will be displayed far from each other

        MplAxes.legend(
            ax,
            handles=handles,
            labels=labels,
            loc='center',
            bbox_to_anchor=np.array(parameters['X_legend_position']),
            frameon=False,
            handlelength=parameters['legend_line_length'],
            handletextpad=parameters['legend_text_horizontal_pad'],
            prop=FontProperties(size=parameters['legend_font_size']),
            labelspacing=parameters['label_spacing'],
            columnspacing=parameters['column_spacing'],
            ncols=2
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
        plot_label=parameters["u_panel_label"],
        plot_label_offset=parameters['panel_label_offset'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order)



    # =============
    # plot for solution_2
    # =============

    ax = fig.axes[1]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax,[0, 0], [mesh_parameters['L'], h])

    X_cur_2, _ = gr.interpolate_curve(
            data_X_cur_2, data_X_cur_2[':0'].min(), data_X_cur_2[':0'].max(), parameters['n_bins_X'])
        
    X_0_2, _ = gr.interpolate_curve(
            data_X_0_2, data_X_0_2[':0'].min(), data_X_0_2[':0'].max(), parameters['n_bins_X'])
    
    
    X_sym_2 = np.array([[mesh_parameters['L'], - h * parameters['X_sym_margin']], [mesh_parameters['L'], h + h*parameters['X_sym_margin']]])
      


    # compute the vector field u and store it in U_x, U_y and its related coordinates X_U, Y_U in the current configuration
    # X_U, Y_U, U_x, U_y = geo.u_1d(data_X_ref, data_U)

    
  
    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices_2,
                    line_width=parameters['mesh_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh_high'],
                    zorder=parameters['mesh_zorder'])
    # plot X_0
    gr.plot_curve_grid(ax, X_0_2,
                        line_color=parameters['X_0_color'],
                        legend=parameters['X_0_legend'],
                        line_width=parameters['X_line_width'],
                        z_order=1
                        )
    
    # plot X_cur
    gr.plot_curve_grid(ax, X_cur_2,
                        line_color=parameters['X_cur_color'],
                        legend=parameters['X_cur_legend'],
                        line_width=parameters['X_line_width'],
                        line_style=parameters['X_cur_line_style'],
                        z_order=1
                        )


    # plot symmetry axis
    gr.plot_curve_grid(ax, X_sym_2,
                        line_color=parameters['X_sym_color'],
                        legend=parameters['X_sym_legend'],
                        line_width=parameters['X_line_width'],
                        line_style=parameters['X_sym_line_style'],
                        alpha=parameters['X_sym_alpha'],
                        z_order=0
                        )


    

    # Create custom legend handles
    handles, labels = ax.get_legend_handles_labels()

    with plt.rc_context({'text.usetex': False}):
        # disable tex otherwise the legend and the line sample will be displayed far from each other

        MplAxes.legend(
            ax,
            handles=handles,
            labels=labels,
            loc='center',
            bbox_to_anchor=np.array(parameters['X_legend_position']),
            frameon=False,
            handlelength=parameters['legend_line_length'],
            handletextpad=parameters['legend_text_horizontal_pad'],
            prop=FontProperties(size=parameters['legend_font_size']),
            labelspacing=parameters['label_spacing'],
            columnspacing=parameters['column_spacing'],
            ncols=2
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
        plot_label=parameters["u_panel_label"],
        plot_label_offset=parameters['panel_label_offset'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order)

     


# plot_snapshot(fig, snapshot_max,
#               snapshot_label=rf'$t = \,$' + io.time_to_string(snapshot_max * solution_parameters['T'] / solution_parameters['N'], 's', 0))
plot_snapshot(fig, parameters['snapshot_to_plot'],
              snapshot_label=rf'$t = \,$' + io.time_to_string(parameters['snapshot_to_plot'] * solution_1_parameters['T'] / solution_1_parameters['N'], 'min_s', 0))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
