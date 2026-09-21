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
import graphics.color_bar as cb
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
    REMOTE_PATH="membrane_phi_2"
    FIGURE_NAME="figure_23"
    cd /Users/michelecastellana/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME
    rm -rf solution
    mkdir solution
    ../copy_from_abacus.sh $REMOTE_PATH/solution/snapshots/csv/  'line_mesh_n_*' 'u_n_*' 'U_n_12_*' 'X_n_12_*' 'X_ref_n_*' 'v_n_*' 'w_n_*' 'sigma_n_12_*' 'nu_n_12_*' 'psi_n_12_*' 'def_v_fl_n_*' 'v_fl_n_*'  'sigma_fl_n_*'  'def_sigma_fl_n_*' 'boundary_points_id_2_n_*'  ~/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME 1 14000 100
    mv $REMOTE_PATH/solution .
    rm -rf $REMOTE_PATH
    rsync -avr mcastel1@abacus:membrane_phi_2/solution/solution_metadata.csv /Users/michelecastellana/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME/solution
    rsync -avr mcastel1@abacus:membrane_phi_2/mesh/solution/mesh_metadata.csv /Users/michelecastellana/Documents/work/manuscripts/paper_ale/figures/$FIGURE_NAME/mesh/solution


to copy the parameters to finite_elements:

    cp ~/Documents/work/manuscripts/paper_ale/figures/figure_5/solution_parameters.csv ~/Documents/finite_elements/fluid_structure_interaction/membrane/parameters_bc_square_no_circle_line_a.csv
    cp ~/Documents/work/manuscripts/paper_ale/figures/figure_5/mesh_parameters.csv ~/Documents/finite_elements/generate_mesh/2d/square_no_circle/line/mesh_parameters.csv 
    cp ~/Documents/work/manuscripts/paper_ale/figures/figure_5/variational_problem_membrane_bc_square_no_circle_line_a.py ~/Documents/finite_elements/fluid_structure_interaction/membrane

'''
matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of

# Show all rows and columns when printing a Pandas array
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))


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
        r"\usepackage{graphicx} "
        r"\usepackage{tikz} "
        rf"\input{{{paths.definitions_path}}}"
        rf"\input{{{os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../definitions.tex')}}}"
    )
})



'''
# 1. read solution from local folder
solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")
'''

# 2 read solution from external folder
solution_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/fluid_structure_interaction/membrane', "solution")
mesh_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/generate_mesh/2d/square_no_circle/line', "solution") 


solution_parameters = io.read_parameters_from_csv_file(os.path.join(solution_path, 'solution_metadata.csv'))
mesh_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_path, 'mesh_metadata.csv'))


figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")
snapshot_nodal_values_path = os.path.join(snapshot_path, "nodal_values")

# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_path)
number_of_frames = snapshot_max - snapshot_min + 1


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
fig.add_subplot(1, 2, 1)
fig.add_subplot(1, 2, 2)



v_fl_colorbar_axis = fig.add_axes(const.default_axis_position_size)
cb.set_size(v_fl_colorbar_axis, parameters['colorbar_size'])

sigma_fl_colorbar_axis = fig.add_axes(const.default_axis_position_size)
cb.set_size(sigma_fl_colorbar_axis, parameters['colorbar_size'])


'''
plot a masking polygon that hides the arrows of v_fl which result from the interpolation and lie outside the mesh in the current configuration
Input values:
    * Mandatory:
        - 'ax': the axis where the polygon will be drawn
        - 'axis_min_max': the bounds of the X values in the current configuration
        - 'data_u_msh': the data for the mesh displacement field
    * Optional:
        - 'margin': a margin, measured as relative to axis_min_max[1][1] - axis_mim_max[1][0] which is used to expand the region on top
'''


# 
# compute the min and max of the axes
#
data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + str(snapshot_max) + '.csv'))

_, Y_msh_ref, _, u_msh_n_Y, _, _, _, _ = vp.interpolate_2d_vector_field(data_u_msh,
                                                                                        [0, 0],
                                                                                        [mesh_parameters['L'],
                                                                                            np.max(data_u_msh[':1'])],
                                                                                        parameters['n_bins_v_fl'])

# Y are the positions of the mesh nodes in the current configuration
Y = np.array(lis.add_lists_of_lists(Y_msh_ref, u_msh_n_Y))

h = lis.min_max(Y)[1]

# 

def draw_masking_area(ax, axis_min_max, data_u_msh, data_ref_boundary_vertices_mesh_1,
                    margin=[0]*2):

    # 1. interpolate the mesh displacement field and construct the sequence of segments of the line corresponding to sub_mesh_1 by adding to the line in the reference configuration the displacement field

    U_interp_x, U_interp_y = vp.interpolating_function_2d_vector_field(data_u_msh)
    h_step = axis_min_max[1][1]

    data_def_boundary_vertices_mesh_1 = []
    for _, row in data_ref_boundary_vertices_mesh_1.iterrows():
        data_def_boundary_vertices_mesh_1.append(
            np.add(
                [row[':0'], row[':1']],
                [U_interp_x(row[':0'], row[':1']),
                 U_interp_y(row[':0'], row[':1'])]
            )
        )

    # 2.  add to the sequence of lines above the top-left and top-right and bottom-right extremal points of the region to cover
    # 2.1 two points at the bottom-right corner
    data_def_boundary_vertices_mesh_1.insert(0, (
        mesh_parameters['L'] + U_interp_x(mesh_parameters['L'], h_step),
        h_step + U_interp_y(mesh_parameters['L'], h_step)
    )
    )
    data_def_boundary_vertices_mesh_1.insert(0, (
        mesh_parameters['L'] + U_interp_x(mesh_parameters['L'], h_step) +
        margin[0] * (axis_min_max[0][1] - axis_min_max[0][0]),
        h_step + U_interp_y(mesh_parameters['L'], h_step)
    )
    )

    # 2.2 bottom-left point
    data_def_boundary_vertices_mesh_1.append(np.subtract(
        data_def_boundary_vertices_mesh_1[-1],
        (margin[0] * (axis_min_max[0]
                      [1] - axis_min_max[0][0]), 0)
    )
    )

    # 2.3 top-left point
    data_def_boundary_vertices_mesh_1.append((
        -margin[0] * (axis_min_max[0][1] - axis_min_max[0][0]),
        axis_min_max[1][1] + margin[1] *
        (axis_min_max[1][1] - axis_min_max[1][0])
    ))
    # 2.4 top-right point
    data_def_boundary_vertices_mesh_1.append((
        axis_min_max[0][1] + margin[0] *
        (axis_min_max[0][1] - axis_min_max[0][0]),
        axis_min_max[1][1] + margin[1] *
        (axis_min_max[1][1] - axis_min_max[1][0])
    ))

    # 3. plot the  polygon in order to hide the arrows
    poly = Polygon(data_def_boundary_vertices_mesh_1, fill=True,
                   linewidth=parameters['plot_line_width'], 
                   edgecolor='white', 
                   facecolor='white', 
                   zorder=const.high_z_order)
    ax.add_patch(poly)

    return data_def_boundary_vertices_mesh_1
    #


def plot_snapshot(fig, n_file,
                  snapshot_label='',
                  axis_min_max=None,
                  norm_v_fl_min_max=None,
                  sigma_fl_min_max=None):

    n_file_string = str(n_file)

    # load data
    # data_el_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_el_n_' + str(n_file) + '.csv')
    data_msh_line_vertices = pd.read_csv(os.path.join(
        snapshot_path, 'line_mesh_n_' + n_file_string + '.csv'))
    
    data_X_ref = pd.read_csv(os.path.join(snapshot_path, 'X_ref_n_' + n_file_string + '.csv'))
    data_X_ref = data_X_ref.sort_values(by=[":0"]).copy()

    data_U = pd.read_csv(os.path.join(snapshot_path, 'U_n_12_' + n_file_string + '.csv'))
    data_U = data_U.sort_values(by=[":0"]).copy()

    data_nu = pd.read_csv(os.path.join(snapshot_path, 'nu_n_12_' + n_file_string + '.csv'))
    data_nu = data_nu.sort_values(by=[":0"]).copy()

    data_psi = pd.read_csv(os.path.join(snapshot_path, 'psi_n_12_' + n_file_string + '.csv'))
    data_psi = data_psi.sort_values(by=[":0"]).copy()

    data_v_fl = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'def_v_fl_n_' + n_file_string + '.csv'))
    data_sigma_fl = pd.read_csv(os.path.join(solution_path, 'snapshots/csv/nodal_values/def_sigma_fl_n_12_' + n_file_string + '.csv'))

    data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + n_file_string + '.csv'))

    data_ref_boundary_vertices_mesh_1 = pd.read_csv(os.path.join(snapshot_path, 'boundary_points_id_' + str(mesh_parameters['mesh_1_id']) + f'_n_{n_file_string}.csv'))

    # data_omega contains de values of \partial_1 X^alpha
    data_omega = lis.data_omega(data_nu, data_psi)
    data_omega = data_omega.sort_values(by=[":0"]).copy()


    # build `data_X_cur` from `data_X_ref` and `data_U`
    data_X_cur = data_X_ref.copy()
    data_X_cur[['f:0', 'f:1']] += data_U[['f:0', 'f:1']]
    data_X_cur = data_X_cur.sort_values(by=[":0"]).copy()



    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1],
             snapshot_label, fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')

    if axis_min_max == None:

        # compute the min and max of the axes
        #
        data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + str(n_file) + '.csv'))

        X_msh_ref, Y_msh_ref, u_msh_n_X, u_msh_n_Y, _, _, _, _ = vp.interpolate_2d_vector_field(data_u_msh,
                                                                                                [0, 0],
                                                                                                [mesh_parameters['L'],
                                                                                                    np.max(data_u_msh[':1'])],
                                                                                                parameters['n_bins_v_fl'])

        # X, Y are the positions of the mesh nodes in the current configuration
        X = np.array(lis.add_lists_of_lists(X_msh_ref, u_msh_n_X))
        Y = np.array(lis.add_lists_of_lists(Y_msh_ref, u_msh_n_Y))

        # compute the min-max of the snapshot
        axis_min_max = [lis.min_max(X), lis.min_max(Y)]
        #


    X_msh_ref, Y_msh_ref, u_msh_n_X, u_msh_n_Y, _, _, _, _ = vec.interpolate_2d_vector_field(data_u_msh,
                                                                                             [0, 0],
                                                                                             [mesh_parameters['L'],
                                                                                                 np.max(data_u_msh[':1'])],
                                                                                             parameters['n_bins_v_fl'],
                                                                                             clab.label_x_column,
                                                                                             clab.label_y_column,
                                                                                             clab.label_v_column)
    
  
    
    # =============
    # v_fl subplot
    # =============

    ax = fig.axes[0]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax,[0, 0], [mesh_parameters['L'], h])

    # here X, Y are the coordinates of the points in the current configuration of the mesh: I interpolate def_v_fl in the rectangle delimited by axis_min_max. In some parts of this rectangle, def_v_fl is not defined and the interpolated points will be set to nan -> This is good because these points are the points outside \Omega and the vector field of v_fl will not be plotted there because its value is nan
    # here I use interpolate_2d_vector_field_layer because the values of the vector field vary very suddenly close to the bottom and right edge of the mesh, so I treat them with one-dimensional interpolation
    X, Y, V_x, V_y, grid_norm_v, norm_v_fl_min, norm_v_fl_max, _ = vec.interpolate_2d_vector_field_layer(
        data_v_fl,
        [axis_min_max[0][0], axis_min_max[1][0]],
        [axis_min_max[0][1], axis_min_max[1][1]],
        parameters['n_bins_v_fl'],
        right_edge_x=mesh_parameters['L'])
    

    if norm_v_fl_min_max == None:
        norm_v_fl_min_max = [norm_v_fl_min, norm_v_fl_max]

    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    line_width=parameters['plot_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh'],
                    zorder=parameters['mesh_zorder'])

     
    # plot the area that masks arrows which lie outside the mesh in the current configuration
    data_def_boundary_vertices_mesh_1 = draw_masking_area(ax, 
                      axis_min_max, 
                      data_u_msh,
                      data_ref_boundary_vertices_mesh_1,
                      parameters['masking_area_margin']
                      )
    

    # set to nan the values of V_x and V_y which lie inside the masking region 
    vp.set_in_polygon(data_def_boundary_vertices_mesh_1,
                      [X, Y],
                      [V_x, V_y])
    
    

    # plot velocity of fluid
    vec.plot_2d_vector_field(ax, [X, Y], [
                             V_x, V_y], parameters['arrow_length'], 0.3, 30, 0.5, 1, 'color_from_map', 0)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_fl_min_max[0], norm_v_fl_min_max[1],
                        label_pad=parameters['colorbar_axis_label_offset'],
                        label_angle=parameters['v_fl_colorbar_label_angle'],
                        label=parameters['v_fl_colorbar_axis_label'],
                        font_size=parameters['colorbar_font_size'],
                        tick_label_angle=parameters['v_fl_colorbar_tick_label_angle'],
                        tick_label_offset=parameters['v_fl_colorbar_tick_label_offset'],
                        line_width=parameters['v_fl_colorbar_tick_line_width'],
                        tick_length=parameters['colorbar_tick_length'],
                        axis=v_fl_colorbar_axis)

    gr.plot_2d_axes(
        ax, [0, 0], [mesh_parameters['L'], h],
        tick_length=parameters['tick_length'],
        line_width=parameters['axis_line_width'],
        axis_label=parameters['axis_label_cur'],
        axis_label_angle=parameters['axis_label_angle'],
        axis_label_offset=parameters['axis_label_offset'],
        tick_label_offset=parameters['tick_label_offset'],
        tick_label_format=['f', 'f'],
        font_size=parameters['axis_font_size'],
        plot_label=parameters["v_fl_panel_label"],
        plot_label_offset=parameters['panel_label_offset'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order,
        colorbar_axis=v_fl_colorbar_axis,
        colorbar_axis_offset=parameters['colorbar_offset'])

    
    # =============
    # sigma_fl subplot
    # =============

    ax = fig.axes[1]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax,
                       [0, 0], [mesh_parameters['L'], h])

    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    line_width=parameters['plot_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh'],
                    zorder=parameters['mesh_zorder'])

    _, _, Z_sigma_fl, _, _, _ = gr.interpolate_surface(data_sigma_fl, [axis_min_max[0][0], axis_min_max[1][0]], [
                                                       axis_min_max[0][1], axis_min_max[1][1]], parameters['n_bins_sigma_fl'])

    if sigma_fl_min_max == None:
        sigma_fl_min, sigma_fl_max, _ = cal.min_max_scalar_field(Z_sigma_fl)
        sigma_fl_min_max = [sigma_fl_min, sigma_fl_max]

    # plot the area that masks arrows which lie outside the mesh in the current configuration
    data_def_boundary_vertices_mesh_1 = draw_masking_area(ax, 
                      axis_min_max, 
                      data_u_msh,
                      data_ref_boundary_vertices_mesh_1,
                      parameters['masking_area_margin']
                      )

    contour_plot = ax.imshow(Z_sigma_fl.T,
                             origin='lower',
                             cmap=gr.cb.color_map_type,
                             aspect='equal',
                             extent=[axis_min_max[0][0], axis_min_max[0]
                                     [1], axis_min_max[1][0], axis_min_max[1][1]],
                             vmin=sigma_fl_min_max[0], vmax=sigma_fl_min_max[1],
                             interpolation='bilinear',
                             zorder=0
                             )

    gr.cb.make_colorbar(
        figure=fig,
        grid_values=Z_sigma_fl,
        min_value=sigma_fl_min_max[0],
        max_value=sigma_fl_min_max[1],
        label_pad=parameters['colorbar_axis_label_offset'],
        tick_label_offset=parameters['sigma_fl_colorbar_tick_label_offset'],
        line_width=parameters['sigma_fl_colorbar_tick_line_width'],
        tick_length=parameters['colorbar_tick_length'],
        tick_label_angle=parameters['sigma_fl_colorbar_tick_label_angle'],
        label=parameters['sigma_fl_colorbar_axis_label'],
        font_size=parameters['colorbar_font_size'],
        mappable=contour_plot,
        axis=sigma_fl_colorbar_axis
    )

    gr.plot_2d_axes(
        ax, [0, 0], [mesh_parameters['L'], h],
        tick_length=parameters['tick_length'],
        line_width=parameters['axis_line_width'],
        axis_label=parameters['axis_label_cur'],
        axis_label_angle=parameters['axis_label_angle'],
        axis_label_offset=parameters['axis_label_offset'],
        tick_label_offset=parameters['tick_label_offset'],
        tick_label_format=['f', 'f'],
        font_size=parameters['axis_font_size'],
        plot_label=parameters["sigma_fl_panel_label"],
        plot_label_offset=parameters['panel_label_offset'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order,
        colorbar_axis=sigma_fl_colorbar_axis,
        colorbar_axis_offset=parameters['colorbar_offset']
    )
    
    
 

    
     


plot_snapshot(fig, snapshot_max,
              snapshot_label=rf'$t = \,$' + io.time_to_string(snapshot_max * solution_parameters['T'] / solution_parameters['N'], 's', 1))
# plot_snapshot(fig, parameters['snapshot_to_plot'],
#               snapshot_label=rf'$t = \,$' + io.time_to_string(parameters['snapshot_to_plot'] * solution_parameters['T'] / solution_parameters['N'], 'min_s', 0))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
