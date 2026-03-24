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

matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of


'''
- to copy from local

OUTPUT_PATH="/Users/michelecastellana/Documents/work/manuscripts/paper_ale/figures/figure_11/"
rm -rf $OUTPUT_PATH/solution
cp ~/Documents/finite_elements/generate_mesh/2d/square/polygon/mesh_parameters.csv $OUTPUT_PATH
cp -r /Users/michelecastellana/Documents/finite_elements/fluid_structure_interaction/rigid_obstacle/remesh/solution $OUTPUT_PATH

- to copy from abacus

INPUT_PATH="remesh_3"
OUTPUT_PATH="/Users/michelecastellana/Documents/work/manuscripts/paper_ale/figures/figure_11/"
STRIDE="10"
cd $OUTPUT_PATH/../
rm -rf $OUTPUT_PATH/solution
./copy_from_abacus.sh $INPUT_PATH/solution/snapshots/csv 'line_mesh_n_*' 'def_v_n_*' 'u_n_*' 'def_sigma_n_12_*' 'boundary_points_id_6_*' $OUTPUT_PATH 0 100000 $STRIDE; mv $OUTPUT_PATH/$INPUT_PATH/solution $OUTPUT_PATH/; rm -rf $OUTPUT_PATH/$INPUT_PATH
rsync -avr mcastel1@abacus:$INPUT_PATH/mesh/mesh_parameters.csv $OUTPUT_PATH
rsync -avr mcastel1@abacus:$INPUT_PATH/solution/solution_metadata.csv $OUTPUT_PATH/solution
rsync -avr mcastel1@abacus:$INPUT_PATH/solution/theta_omega.csv $OUTPUT_PATH/solution
rsync -avr mcastel1@abacus:$INPUT_PATH/solution/remesh.csv $OUTPUT_PATH/solution

'''

# add the path where to find the shared modules
module_path = paths.root_path + "/figures/modules/"
sys.path.append(module_path)

# Suppress the specific warning
warnings.filterwarnings(
    "ignore", message=".*Z contains NaN values.*", category=UserWarning)
# clean the matplotlib cache to load the correct version of definitions.tex
# os.system("rm -rf ~/.matplotlib/tex.cache")

pplt.rc['grid'] = False  # disables default gridlines

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{newpxtext,newpxmath} "
        r"\usepackage{xcolor} "
        r"\usepackage{glossaries} "
        rf"\input{{{paths.definitions_path}}}"
        rf"\input{{{os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../definitions.tex')}}}"
    )
})




print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))

parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))
mesh_parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'mesh_parameters.csv'))
solution_parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'solution', 'solution_metadata.csv'))


if parameters['animation_frame_stride'] % solution_parameters['print_out_stride'] != 0:
    raise RuntimeError(f'Error: Animation frame stride is not a multiple of print out stride ! \n animation frame stride = {parameters["animation_frame_stride"]} \n print out stride = {solution_parameters["print_out_stride"]} \nAborting...')

solution_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "solution/")
figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")

snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_path)
number_of_frames = snapshot_max - snapshot_min + 1


# labels of columns to read
data_theta_omega = pd.read_csv(os.path.join(solution_path, 'theta_omega.csv'))
data_remesh = pd.read_csv(os.path.join(solution_path, 'remesh.csv'))

fig = pplt.figure(figsize=parameters['figure_size'], left=parameters['figure_margin_l'],
                  bottom=parameters['figure_margin_b'], right=parameters['figure_margin_r'],
                  top=parameters['figure_margin_t'], wspace=0, hspace=0)

# pre-create subplots and axes
fig.add_subplot(1, 1, 1)
# fig.add_subplot(2, 1, 2)

_colorbar_initialized = False


v_colorbar_axis = fig.add_axes([parameters['v_colorbar_position'][0],
                                parameters['v_colorbar_position'][1],
                                parameters['v_colorbar_size'][0],
                                parameters['v_colorbar_size'][1]])

sigma_colorbar_axis = fig.add_axes([parameters['sigma_colorbar_position'][0],
                                    parameters['sigma_colorbar_position'][1],
                                    parameters['sigma_colorbar_size'][0],
                                    parameters['sigma_colorbar_size'][1]])


def plot_snapshot(fig, n_file,
                  snapshot_label='',
                  norm_v_min_max=None,
                  sigma_min_max=None):


    global _colorbar_initialized

    n_snapshot = str(n_file)


    data_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_n_' + n_snapshot + '.csv')
    data_boundary_vertices_ellipse = pd.read_csv(os.path.join(
    snapshot_path, 'boundary_points_id_' + str(parameters['ellipse_loop_id']) + f'_n_{n_snapshot}.csv'))

    data_v = pd.read_csv(os.path.join(solution_path, 'snapshots/csv/nodal_values/def_v_n_' + n_snapshot + '.csv'))
    data_sigma = pd.read_csv(os.path.join(solution_path, 'snapshots/csv/nodal_values/def_sigma_n_12_' + n_snapshot + '.csv'))
    data_u = pd.read_csv(os.path.join(solution_path, 'snapshots/csv/nodal_values/u_n_' + n_snapshot + '.csv'))

    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1],
             snapshot_label, fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')


    # plot the polygon of the boundary 'ellipse_loop_id'
    #
    # build two a vector field which interpolates the displacement field in data_u_msh
    U_interp_x, U_interp_y = vp.interpolating_function_2d_vector_field(data_u)


    # run through points in data_boundary_vertices_ellipse (reference configuration) and add to them [U_interp_x, U_interp_y] in order to obtain the boundary polygon in the current configuration
    data_def_boundary_vertices_ellipse = []
    for _, row in data_boundary_vertices_ellipse.iterrows():
        data_def_boundary_vertices_ellipse.append(
            np.add(
                [row[':0'], row[':1']],
                [U_interp_x(row[':0'], row[':1']),
                 U_interp_y(row[':0'], row[':1'])]
            )
        )

    # =============
    # v subplot
    # =============

    ax = fig.axes[0]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    gr.plot_2d_mesh(ax, data_line_vertices,
                    line_width=parameters['mesh_line_width_v_plot'],
                    color='black',
                    alpha=parameters['alpha_mesh'])
    
    

    X, Y, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, _ = vp.interpolate_2d_vector_field(data_v,
                                                                                            [0, 0],
                                                                                            [mesh_parameters['L'], mesh_parameters['h']],
                                                                                            parameters['n_bins_v'],
                                                                                            clab.label_x_column,
                                                                                            clab.label_y_column,
                                                                                            clab.label_v_column)
    
  
    if norm_v_min_max == None:
        norm_v_min_max = [norm_v_min, norm_v_max]


    # set to nan the values of V_x and V_y which lie inside the polygon defined by data_def_boundary_vertices_ellipse
    vp.set_in_polygon(data_def_boundary_vertices_ellipse,
                      [X, Y],
                      [V_x, V_y])


    # set to nan the values of V_x, V_y which are on the lines x[1] = 0 or x[1] = h, because these will be vectors with zero norm which are not meaningful
    mask = np.isclose(np.array(Y), 0) | np.isclose(np.array(Y), mesh_parameters['h'])
    V_x[mask] = np.nan
    V_y[mask] = np.nan

  
    vp.plot_2d_vector_field(ax, [X, Y], [V_x, V_y], parameters['arrow_length'], parameters['head_over_shaft_length'], 30, 1, 1, 'color_from_map', 0,
                            clip_on=False)
    
  
    if not _colorbar_initialized:

        gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min_max[0], norm_v_min_max[1], parameters['v_colorbar_position'], parameters['v_colorbar_size'],
                            label=parameters['v_colorbar_axis_label'],
                            font_size=parameters['v_colorbar_font_size'],
                            tick_length=parameters['v_colorbar_tick_length'],
                            label_pad=parameters['v_colorbar_label_offset'],
                            tick_label_offset=parameters['v_colorbar_tick_label_offset'],
                            tick_label_angle=parameters['v_colorbar_tick_label_angle'],
                            line_width=parameters['v_colorbar_line_width'],
                            tick_label_format=parameters['v_colorbar_tick_label_format'],
                            axis=v_colorbar_axis)
        
        _colorbar_initialized = True


    # plot the ellipse focal point
    pivot_point_position = mesh_parameters['c']
    theta_1 = min(0, data_theta_omega.loc[int(
        n_file/solution_parameters['print_out_stride']-1), 'theta'])
    theta_2 = max(0, data_theta_omega.loc[int(
        n_file/solution_parameters['print_out_stride']-1), 'theta'])

    ax.scatter(pivot_point_position[0], pivot_point_position[1],
               color=parameters['ellipse_focal_point_color'], s=parameters['ellipse_focal_point_size'])

    # plot the axes that define the angle theta
    # 1) plot fixed axis
    ax.plot(
        [pivot_point_position[0], pivot_point_position[0] +
            parameters['angle_axis_length']],
        [pivot_point_position[1]] * 2,
        color=parameters['ellipse_angle_axis_color'],
        linewidth=parameters['mesh_line_width_curr_plot'],
        linestyle='--',
        zorder=const.high_z_order
    )
    # 2) plot moving axis
    delta = np.dot(
        gr.R_2d(data_theta_omega.loc[int(
            n_file/solution_parameters['print_out_stride']-1), 'theta']),
        [parameters['angle_axis_length'], 0]
    )

    ax.plot(
        [pivot_point_position[0], pivot_point_position[0] + delta[0]],
        [pivot_point_position[1], pivot_point_position[1] + delta[1]],
        color=parameters['ellipse_angle_axis_color'],
        linewidth=parameters['mesh_line_width_curr_plot'],
        linestyle='--',
        zorder=const.high_z_order
    )

    theta_arc = Arc(pivot_point_position,
                    theta1=const.rad_to_deg * theta_1,
                    theta2=const.rad_to_deg * theta_2,
                    width=parameters['ellipse_angle_axis_length'],
                    height=parameters['ellipse_angle_axis_length'],
                    color=parameters['ellipse_angle_axis_color'],
                    linestyle=':',
                    linewidth=parameters['ellipse_angle_line_width'],
                    zorder=const.high_z_order
                    )

    ax.add_patch(theta_arc)


    gr.plot_2d_axes(ax, [0, 0], [mesh_parameters['L'], mesh_parameters['h']],
                    tick_length=parameters['tick_length'],
                    line_width=parameters['axis_line_width'],
                    axis_label=parameters['axis_label'],
                    tick_label_format=['f', 'f'],
                    font_size=[parameters['font_size'],
                               parameters['font_size']],
                    tick_label_offset=parameters['tick_label_offset'],
                    axis_label_offset=parameters['axis_label_offset'],
                    axis_origin=parameters['axis_origin'],
                    plot_label=parameters["v_plot_panel_label"],
                    plot_label_offset=parameters['panel_label_position'],
                    plot_label_font_size=parameters['panel_label_font_size'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length'],
                    tick_label_angle=parameters['tick_label_angle'],
                    colorbar_axis=v_colorbar_axis,
                    colorbar_axis_offset=parameters['colorbar_axis_offset'])

 
    '''
    # =============
    # sigma subplot
    # =============

    ax = fig.axes[1]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    gr.plot_2d_mesh(ax, data_line_vertices,
                    line_width=parameters['mesh_line_width_sigma_plot'],
                    color='black',
                    alpha=parameters['alpha_mesh'],
                    zorder=1)

    _, _, Z_sigma, _, _, _ = gr.interpolate_surface(
        data_sigma, [0, 0], [mesh_parameters['L'], mesh_parameters['h']], parameters['n_bins_sigma'])

    if sigma_min_max == None:
        sigma_min, sigma_max, _ = cal.min_max_scalar_field(Z_sigma)
        sigma_min_max = [sigma_min, sigma_max]





    # plot the boundary polygon in the current configuration
    poly = Polygon(data_def_boundary_vertices_ellipse, fill=True,
                   linewidth=parameters['mesh_line_width_sigma_plot'], edgecolor='red', facecolor='white', zorder=1)
    ax.add_patch(poly)
    #

    contour_plot = ax.imshow(Z_sigma.T,
                             origin='lower',
                             cmap=gr.cb.color_map_type,
                             aspect='equal',
                             extent=[0, mesh_parameters['L'], 0, mesh_parameters['h']],
                             vmin=sigma_min_max[0], vmax=sigma_min_max[1],
                             interpolation='bilinear',
                             zorder=0
                             )

    # Corrected make_colorbar call (remove 'location')
    gr.cb.make_colorbar(
        figure=fig,
        grid_values=Z_sigma,
        min_value=sigma_min_max[0],
        max_value=sigma_min_max[1],
        position=parameters['sigma_colorbar_position'],
        size=parameters['sigma_colorbar_size'],
        label_pad=parameters['sigma_colorbar_label_offset'],
        tick_label_offset=parameters['sigma_colorbar_tick_label_offset'],
        line_width=parameters['sigma_colorbar_tick_line_width'],
        tick_length=parameters['sigma_colorbar_tick_length'],
        tick_label_angle=parameters['sigma_colorbar_tick_label_angle'],
        label=parameters['sigma_colorbar_axis_label'],
        tick_label_format=parameters['sigma_colorbar_tick_label_format'],
        mappable=contour_plot,
        axis=sigma_colorbar_axis
    )

    gr.plot_2d_axes(ax, [0, 0], [mesh_parameters['L'], mesh_parameters['h']],
                    tick_length=parameters['tick_length'],
                    line_width=parameters['axis_line_width'],
                    axis_label=parameters['axis_label'],
                    tick_label_format=['f', 'f'],
                    font_size=[parameters['font_size'],
                               parameters['font_size']],
                    tick_label_offset=parameters['tick_label_offset'],
                    axis_label_offset=parameters['axis_label_offset'],
                    axis_origin=parameters['axis_origin'],
                    plot_label=parameters["sigma_plot_panel_label"],
                    plot_label_offset=parameters['panel_label_position'],
                    plot_label_font_size=parameters['panel_label_font_size'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length'],
                    tick_label_angle=parameters['tick_label_angle'],
                    colorbar_axis=sigma_colorbar_axis,
                    colorbar_axis_offset=parameters['colorbar_axis_offset'])
    '''




# plot_snapshot(fig, parameters['snapshot_to_plot'],
#               snapshot_label=rf'$t = \,$' + io.time_to_string(parameters['snapshot_to_plot'] * solution_parameters['T'] / solution_parameters['num_steps'], 'min_s', 0))
plot_snapshot(fig, snapshot_max,
              snapshot_label=rf'$t = \,$' + io.time_to_string(parameters['snapshot_to_plot'] * solution_parameters['T'] / solution_parameters['num_steps'], 'min_s', 0))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
