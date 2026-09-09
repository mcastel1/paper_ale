import matplotlib
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings

import calculus.utils as cal
import constants.utils as const
import list.column_labels as clab
import graphics.utils as gr
import graphics.vector_plot as vp
import input_output.utils as io
import list.utils as lis
import system.paths as paths
import system.utils as sys_utils
import graphics.vector_plot as vec

'''
you can copy the data from abacus with
./copy_from_abacus.sh membrane_1/solution/snapshots/csv/  'line_mesh_n_*' 'u_n_*' 'X_n_12_*' 'v_n_*' 'w_n_*' 'sigma_n_12_*' 'nu_n_12_*' 'psi_n_12_*' 'def_v_fl_n_*' 'v_fl_n_*'  'sigma_fl_n_*'  'def_sigma_fl_n_*'  ~/Documents/work/manuscripts/paper_ale/figures/figure_22 1 1000000 10


to copy the parameters to finite_elements:
cp ~/Documents/work/manuscripts/paper_ale/figures/figure_22/mesh_parameters.csv ~/Documents/finite_elements/generate_mesh/2d/square_no_circle/line/mesh_parameters.csv
cp ~/Documents/work/manuscripts/paper_ale/figures/figure_22/mesh_parameters.csv ~/Documents/finite_elements/generate_mesh/2d/square_no_circle/line/mesh_parameters.csv
'''

matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of

# Show all rows and columns when printing a Pandas array
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))
solution_parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'solution_parameters.csv'))


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
        r"\usepackage{bm} "
        r"\usepackage{newpxtext,newpxmath} "
        r"\usepackage{xcolor} "
        r"\usepackage{glossaries} "
        rf"\input{{{paths.definitions_path}}}"
        rf"\input{{{os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../definitions.tex')}}}"
    )
})

print("Current working directory:", os.getcwd())
print("root_path:", os.path.dirname(os.path.abspath(__file__)))
'''

# 1. read solution from local folder
solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")

'''

# 2 read solutio from external folder
solution_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/fluid_structure_interaction/membrane/', "solution/")
mesh_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/generate_mesh/2d/square_no_circle/line/', "solution/")
 



figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")
snapshot_nodal_values_path = os.path.join(snapshot_path, "nodal_values")

# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_path)
number_of_frames = snapshot_max - snapshot_min + 1


data_ref_boundary_vertices_sub_mesh_1 = pd.read_csv(os.path.join(
    mesh_path, 'mesh_0', 'boundary_points_id_' + str(parameters['sub_mesh_1_id']) + '.csv'))


fig = pplt.figure(
    figsize=(parameters['figure_size'][0], parameters['figure_size'][1]),
    left=parameters['figure_margin_l'],
    bottom=parameters['figure_margin_b'],
    right=parameters['figure_margin_r'],
    top=parameters['figure_margin_t'],
    wspace=parameters['wspace'],
    hspace=parameters['hspace'])

# pre-create subplots and axes
fig.add_subplot(1, 1, 1)

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
    data_msh_line_vertices = pd.read_csv(os.path.join(snapshot_path, 'line_mesh_n_' + n_file_string + '.csv'))
    # data_X = pd.read_csv(os.path.join(snapshot_path, 'X_n_12_' + n_file_string + '.csv'))
    
    data_nu = pd.read_csv(os.path.join(snapshot_path, 'nu_n_12_' + n_file_string + '.csv'))
    data_psi = pd.read_csv(os.path.join(snapshot_path, 'psi_n_12_' + n_file_string + '.csv'))
    data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + n_file_string + '.csv'))

    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1],
             snapshot_label, fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')

    if axis_min_max == None:

        # compute the min and max of the axes
        #
        data_u_msh = pd.read_csv(os.path.join(
            snapshot_nodal_values_path, 'u_n_' + str(n_file) + '.csv'))

        X_msh_ref, Y_msh_ref, u_msh_n_X, u_msh_n_Y, _, _, _, _ = vp.interpolate_2d_vector_field(data_u_msh,
                                                                                                [0, 0],
                                                                                                [parameters['L'],
                                                                                                    parameters['h']],
                                                                                                parameters['n_bins_v_fl'])

        # X, Y are the positions of the mesh nodes in the current configuration
        X = np.array(lis.add_lists_of_lists(X_msh_ref, u_msh_n_X))
        Y = np.array(lis.add_lists_of_lists(Y_msh_ref, u_msh_n_Y))

        # compute the min-max of the snapshot
        axis_min_max = [lis.min_max(X), lis.min_max(Y)]
        #

    if nu_min_max == None:
        nu_min_max = cal.min_max_file(os.path.join(snapshot_path, 'nu_n_12_' + str(n_file) + '.csv'))
    if psi_min_max == None:
        psi_min_max = cal.min_max_file(os.path.join(snapshot_path, 'psi_n_12_' + str(n_file) + '.csv'))
    if norm_v_min_max == None:
        norm_v_min_max = cal.norm_min_max_file(os.path.join(snapshot_path, 'v_n_' + str(n_file) + '.csv'), scalar=True)
    if sigma_min_max == None:
        sigma_min_max = cal.min_max_file(os.path.join(snapshot_path, 'sigma_n_12_' + str(n_file) + '.csv'))
    if w_min_max == None:
        w_min_max = cal.min_max_file(os.path.join(snapshot_path, 'w_n_' + str(n_file) + '.csv'))

    # X_curr, t = gr.interpolate_curve(
    #     data_X, axis_min_max[0][0], axis_min_max[0][1], parameters['n_bins_X'])

    X_msh_ref, Y_msh_ref, u_msh_n_X, u_msh_n_Y, _, _, _, _ = vec.interpolate_2d_vector_field(data_u_msh,
                                                                                             [0, 0],
                                                                                             [parameters['L'],
                                                                                                 parameters['h']],
                                                                                             parameters['n_bins_v_fl'],
                                                                                             clab.label_x_column,
                                                                                             clab.label_y_column,
                                                                                             clab.label_v_column)
    
    # =============
    # u subplot
    # =============

    ax = fig.axes[0]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax,[0, 0], [parameters['L'], parameters['h']])

 

    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    line_width=parameters['plot_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh'],
                    zorder=parameters['mesh_zorder'])

 
    gr.plot_2d_axes(
        ax, [0, 0], [parameters['L'], parameters['h']],
        tick_length=parameters['tick_length'],
        line_width=parameters['axis_line_width'],
        axis_label=parameters['axis_label'],
        axis_label_angle=parameters['axis_label_angle'],
        axis_label_offset=parameters['axis_label_offset'],
        tick_label_offset=parameters['tick_label_offset'],
        tick_label_format=['f', 'f'],
        font_size=parameters['axis_font_size'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order)

    
     


plot_snapshot(fig, snapshot_max,
              snapshot_label=rf'$t = \,$' + io.time_to_string(snapshot_max * solution_parameters['T'] / solution_parameters['N'], 's', 1))
# plot_snapshot(fig, parameters['snapshot_to_plot'],
#               snapshot_label=rf'$t = \,$' + io.time_to_string(parameters['snapshot_to_plot'] * solution_parameters['T'] / solution_parameters['N'], 'min_s', 0))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
