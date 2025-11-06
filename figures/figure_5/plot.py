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
./copy_from_abacus.sh membrane_1/solution/snapshots/csv/  'line_mesh_n_*' 'u_n_*' 'X_n_12_*' 'v_n_*' 'w_n_*' 'sigma_n_12_*' 'nu_n_12_*' 'psi_n_12_*' 'def_v_fl_n_*' 'v_fl_n_*'  'sigma_fl_n_*'  'def_sigma_fl_n_*'  ~/Documents/paper_ale/figures/figure_5 1 1000000 30000
'''

matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of

# Show all rows and columns when printing a Pandas array
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

parameters = io.read_parameters_from_csv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))   


# add the path where to find the shared modules
module_path = paths.root_path + "/figures/modules/"
sys.path.append(module_path)

# Suppress the specific warning
warnings.filterwarnings("ignore", message=".*Z contains NaN values.*", category=UserWarning)
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
print("root_path:", os.path.dirname(os.path.abspath(__file__)))



solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")
snapshot_nodal_values_path = os.path.join(snapshot_path, "nodal_values")


# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_path)
number_of_frames = snapshot_max - snapshot_min + 1


       
data_ref_boundary_vertices_sub_mesh_1 = pd.read_csv(os.path.join(mesh_path, 'boundary_points_id_' + str(parameters['sub_mesh_1_id']) + '.csv'))    
      


fig = pplt.figure(
    figsize=(parameters['figure_size'][0], parameters['figure_size'][1]), 
    left=parameters['figure_margin_l'], 
    bottom=parameters['figure_margin_b'], 
    right=parameters['figure_margin_r'], 
    top=parameters['figure_margin_t'], 
    wspace=parameters['wspace'], 
    hspace=parameters['hspace'])

# pre-create subplots and axes
fig.add_subplot(2, 3, 1)
fig.add_subplot(2, 3, 2)
fig.add_subplot(2, 3, 4)
fig.add_subplot(2, 3, 5)
fig.add_subplot(2, 3, 6)

v_fl_colorbar_axis = fig.add_axes([parameters['v_fl_colorbar_position'][0], 
                           parameters['v_fl_colorbar_position'][1],
                           parameters['v_fl_colorbar_size'][0],
                           parameters['v_fl_colorbar_size'][1]])

sigma_fl_colorbar_axis = fig.add_axes([parameters['sigma_fl_colorbar_position'][0], 
                           parameters['sigma_fl_colorbar_position'][1],
                           parameters['sigma_fl_colorbar_size'][0],
                           parameters['sigma_fl_colorbar_size'][1]])

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
def draw_masking_area(ax, axis_min_max, data_u_msh,
                      margin=[0]*2):
    
    # 1)interpolate the mesh displacement field and construct the sequence of segments of the line corresponding to sub_mesh_1 by adding to the line in the reference configuration the displacement field 
    
    U_interp_x, U_interp_y = vp.interpolating_function_2d_vector_field(data_u_msh)

    data_def_boundary_vertices_sub_mesh_1 = []
    for index, row in data_ref_boundary_vertices_sub_mesh_1.iterrows():
        data_def_boundary_vertices_sub_mesh_1.append(
            np.add(
                [row[':0'], row[':1']], 
                [U_interp_x(row[':0'], row[':1']), U_interp_y(row[':0'], row[':1'])]
                )
            )
    
    #2) add to the sequence of lines above the top-left and top-right and bottom-right extremal points of the region to cover
    # two points at the bottom-right corner
    data_def_boundary_vertices_sub_mesh_1.insert(0, (
                                                        parameters['L'] + U_interp_x(parameters['L'], parameters['h']), 
                                                        parameters['h'] + U_interp_y(parameters['L'], parameters['h'])
                                                    )
                                                 )
    data_def_boundary_vertices_sub_mesh_1.insert(0, (
                                                        parameters['L'] + U_interp_x(parameters['L'], parameters['h']) + margin[0] * (axis_min_max[0][1] - axis_min_max[0][0]), 
                                                        parameters['h'] + U_interp_y(parameters['L'], parameters['h'])
                                                    )
                                                 )

    # bottom-left point
    data_def_boundary_vertices_sub_mesh_1.append(np.subtract(
                                                            data_def_boundary_vertices_sub_mesh_1[-1],
                                                            (margin[0] * (axis_min_max[0][1] - axis_min_max[0][0]), 0)
                                                            )
                                                )

    # top-left point
    data_def_boundary_vertices_sub_mesh_1.append((
                                                    -margin[0] * (axis_min_max[0][1] - axis_min_max[0][0]), 
                                                    axis_min_max[1][1] + margin[1] * (axis_min_max[1][1] - axis_min_max[1][0])
                                                ))
    # top-right point
    data_def_boundary_vertices_sub_mesh_1.append((
                                                    axis_min_max[0][1] + margin[0] * (axis_min_max[0][1] - axis_min_max[0][0]), 
                                                    axis_min_max[1][1] + margin[1] * (axis_min_max[1][1] - axis_min_max[1][0])
                                                ))
        
    #3) plot the  polygon in order to hide the arrows
    poly = Polygon(data_def_boundary_vertices_sub_mesh_1, fill=True, linewidth=parameters['plot_line_width'], edgecolor='red', facecolor='white', zorder=1)
    ax.add_patch(poly)
    # 

def plot_snapshot(fig, n_file, 
                  snapshot_label='',
                  axis_min_max=None,
                  norm_v_fl_min_max=None,
                  sigma_fl_min_max=None,
                  norm_v_min_max=None,
                  w_min_max=None,
                  sigma_min_max=None):
    
    n_file_string = str(n_file)
    

    # load data
    # data_el_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_el_n_' + str(n_file) + '.csv')
    data_msh_line_vertices = pd.read_csv(os.path.join(snapshot_path, 'line_mesh_n_' + n_file_string + '.csv'))
    data_X = pd.read_csv(os.path.join(snapshot_path, 'X_n_12_' + n_file_string + '.csv'))
    data_v_fl = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'def_v_fl_n_' + n_file_string + '.csv'))
    data_sigma_fl = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/def_sigma_fl_n_12_' + n_file_string + '.csv')
    data_w = pd.read_csv(os.path.join(snapshot_path, 'w_n_' + n_file_string + '.csv'))
    data_sigma = pd.read_csv(os.path.join(snapshot_path, 'sigma_n_12_' + n_file_string + '.csv'))
    data_v = pd.read_csv(os.path.join(snapshot_path, 'v_n_' + n_file_string + '.csv'))
    data_nu = pd.read_csv(os.path.join(snapshot_path, 'nu_n_12_' + n_file_string + '.csv'))
    data_psi = pd.read_csv(os.path.join(snapshot_path, 'psi_n_12_' + n_file_string + '.csv'))
    data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + n_file_string + '.csv'))
    
    # data_omega contains de values of \partial_1 X^alpha
    data_omega  = lis.data_omega(data_nu, data_psi)
    
    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1], snapshot_label, fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')


    if axis_min_max == None:

        # compute the min and max of the axes
        # 
        data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + str(n_file) + '.csv'))

        X_ref, Y_ref, u_n_X, u_n_Y, _, _, _, _ = vp.interpolate_2d_vector_field(data_u_msh,
                                                                                [0, 0],
                                                                                [parameters['L'], parameters['h']],
                                                                                parameters['n_bins_v_fl'])
        
        #X, Y are the positions of the mesh nodes in the current configuration    
        X = np.array(lis.add_lists_of_lists(X_ref, u_n_X))
        Y = np.array(lis.add_lists_of_lists(Y_ref, u_n_Y))

        # compute the min-max of the snapshot
        axis_min_max = [lis.min_max(X),lis.min_max(Y)]
        # 
        
    X_t, t = gr.interpolate_curve(data_X, axis_min_max[0][0], axis_min_max[0][1], parameters['n_bins_X'])
    
    X_ref, Y_ref, u_n_X, u_n_Y, _, _, _, _ = vec.interpolate_2d_vector_field(data_u_msh,
                                                                                                                [0, 0],
                                                                                                                [parameters['L'], parameters['h']],
                                                                                                                parameters['n_bins_v_fl'],
                                                                                                                clab.label_x_column,
                                                                                                                clab.label_y_column,
                                                                                                                clab.label_v_column)

    
    # =============
    # v_fl subplot
    # =============    
    
    ax = fig.axes[0]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

        
    # here X, Y are the coordinates of the points in the current configuration of the mesh: I interpolate def_v_fl in the rectangle delimited by axis_min_max. In some parts of this rectangle, def_v_fl is not defined and the interpolated points will be set to nan -> This is good because these points are the points outside \Omega and the vector field of v_fl will not be plotted there because its value is nan
    X, Y, V_x, V_y, grid_norm_v, norm_v_fl_min, norm_v_fl_max, _ = vec.interpolate_2d_vector_field(data_v_fl,
                                                                                                    [axis_min_max[0][0], axis_min_max[1][0]],
                                                                                                    [axis_min_max[0][1], axis_min_max[1][1]],
                                                                                                    parameters['n_bins_v_fl'])
    

    if norm_v_fl_min_max == None:
        norm_v_fl_min_max = [norm_v_fl_min, norm_v_fl_max]



    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices, 
                    line_width=parameters['plot_line_width'], 
                    color='black', 
                    alpha=parameters['alpha_mesh'],
                    zorder=parameters['mesh_zorder'])
    
    # plot the area that masks arrows which lie outside the mesh in the current configuration
    draw_masking_area(ax, axis_min_max, data_u_msh, parameters['masking_area_margin'])

    # plot velocity of fluid
    vec.plot_2d_vector_field(ax, [X, Y], [V_x, V_y], parameters['arrow_length'], 0.3, 30, 0.5, 1, 'color_from_map', 0)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_fl_min_max[0], norm_v_fl_min_max[1], \
                        position=parameters['v_fl_colorbar_position'], 
                        size=parameters['v_fl_colorbar_size'], \
                        label_pad=parameters['v_fl_colorbar_axis_label_offset'], 
                        label_angle=parameters['v_fl_colorbar_label_angle'], 
                        label=parameters['v_fl_colorbar_axis_label'],
                        font_size=parameters['v_fl_colorbar_font_size'], 
                        tick_label_angle=parameters['v_fl_colorbar_tick_label_angle'],
                        tick_label_offset=parameters['v_fl_colorbar_tick_label_offset'],
                        line_width=parameters['v_fl_colorbar_tick_line_width'],
                        tick_length=parameters['v_fl_colorbar_tick_length'],
                        axis=v_fl_colorbar_axis)
                        
    

    # fig.text(parameters['v_fl_panel_label_position'][0], parameters['v_fl_panel_label_position'][1], rf'${parameters["v_fl_panel_label"]}$', fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')

    gr.plot_2d_axes(
                    ax, [0, 0], [parameters['L'], parameters['h']], 
                    tick_length=parameters['tick_length'], 
                    line_width=parameters['axis_line_width'], 
                    axis_label=parameters['axis_label'], 
                    axis_label_angle=parameters['axis_label_angle'], 
                    axis_label_offset=parameters['axis_label_offset'], 
                    tick_label_offset=parameters['tick_label_offset'], 
                    tick_label_format=['f', 'f'], \
                    font_size=parameters['axis_font_size'], 
                    plot_label=parameters["v_fl_panel_label"],
                    plot_label_offset=parameters['panel_label_offset'], 
                    axis_origin=parameters['axis_origin'], 
                    axis_bounds=axis_min_max, 
                    margin=parameters['axis_margin'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length'],
                    z_order=const.high_z_order)
    
    
    # =============
    # sigma_fl subplot
    # =============   
    
    ax = fig.axes[1]  # Use the existing axis
    
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    
    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices, 
                    line_width=parameters['plot_line_width'], 
                    color='black', 
                    alpha=parameters['alpha_mesh'],
                    zorder=parameters['mesh_zorder'])
    
    X_sigma_fl, Y_sigma_fl, Z_sigma_fl, _, _, _ = gr.interpolate_surface(data_sigma_fl, [axis_min_max[0][0], axis_min_max[1][0]], [axis_min_max[0][1], axis_min_max[1][1]], parameters['n_bins_sigma_fl'])


    if sigma_fl_min_max == None:
        sigma_fl_min, sigma_fl_max, _ = cal.min_max_scalar_field(Z_sigma_fl)
        sigma_fl_min_max = [sigma_fl_min, sigma_fl_max]
        
    # plot the area that masks arrows which lie outside the mesh in the current configuration
    draw_masking_area(ax, axis_min_max, data_u_msh, parameters['masking_area_margin'])
        
    contour_plot = ax.imshow(Z_sigma_fl.T, 
                    origin='lower', 
                    cmap=gr.cb.color_map_type, 
                    aspect='equal', 
                    extent=[axis_min_max[0][0], axis_min_max[0][1], axis_min_max[1][0], axis_min_max[1][1]],
                    vmin=sigma_fl_min_max[0], vmax=sigma_fl_min_max[1],
                    interpolation='bilinear',
                    zorder=0
                    )
    
    gr.cb.make_colorbar(
        figure=fig,
        grid_values=Z_sigma_fl,
        min_value=sigma_fl_min_max[0],
        max_value=sigma_fl_min_max[1],
        position=parameters['sigma_fl_colorbar_position'],
        size=parameters['sigma_fl_colorbar_size'],
        label_pad=parameters['sigma_fl_colorbar_axis_label_offset'],
        tick_label_offset=parameters['sigma_fl_colorbar_tick_label_offset'],
        line_width=parameters['sigma_fl_colorbar_tick_line_width'],
        tick_length=parameters['sigma_fl_colorbar_tick_length'],
        tick_label_angle=parameters['sigma_fl_colorbar_tick_label_angle'],
        label=parameters['sigma_fl_colorbar_axis_label'],
        font_size=parameters['sigma_fl_colorbar_font_size'], 
        mappable = contour_plot,
        axis=sigma_fl_colorbar_axis
    )
    
    gr.plot_2d_axes(
                ax, [0, 0], [parameters['L'], parameters['h']], 
                tick_length=parameters['tick_length'], 
                line_width=parameters['axis_line_width'], 
                axis_label=parameters['axis_label'], 
                axis_label_angle=parameters['axis_label_angle'], 
                axis_label_offset=parameters['axis_label_offset'], 
                tick_label_offset=parameters['tick_label_offset'], 
                tick_label_format=['f', 'f'], \
                font_size=parameters['axis_font_size'], 
                plot_label=parameters["sigma_fl_panel_label"],
                plot_label_offset=parameters['panel_label_offset'],
                axis_origin=parameters['axis_origin'], 
                axis_bounds=axis_min_max, 
                margin=parameters['axis_margin'],
                n_minor_ticks=parameters['n_minor_ticks'],
                minor_tick_length=parameters['minor_tick_length'],
                z_order=const.high_z_order)
    
    
    # =============
    # v subplot
    # =============   
    
    ax = fig.axes[2]  # Use the existing axis
    
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    
    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices, 
                    line_width=parameters['plot_line_width'], 
                    color='black', 
                    alpha=parameters['alpha_mesh'],
                    zorder=parameters['mesh_zorder'])
    
    # plot v
    X_v, Y_v, V_x, V_y, grid_norm_v, _, _, _ = vp.interpolate_t_vector_field_2d_arc_length_gauge(data_X, data_omega, data_v, parameters['n_bins_v'])
    
    if norm_v_min_max == None:
        norm_v_min_max=cal.norm_min_max_file(os.path.join(snapshot_path, 'v_n_' + str(n_file) + '.csv'))
    
    vp.plot_1d_vector_field(ax, [X_v, Y_v], [V_x, V_y], 
                        shaft_length=parameters['shaft_length'], 
                        head_over_shaft_length=parameters['head_over_shaft_length'], 
                        head_angle=parameters['head_angle'], 
                        line_width=parameters['arrow_line_width'], 
                        alpha=parameters['alpha'], 
                        color='color_from_map', 
                        threshold_arrow_length=parameters['threshold_arrow_length_v'])
    
    
    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min_max[0], norm_v_min_max[1], \
                        position=parameters['v_colorbar_position'], 
                        size=parameters['v_colorbar_size'], 
                        label_pad=parameters['v_colorbar_axis_label_offset'], 
                        label=parameters['v_colorbar_axis_label'], 
                        label_angle=parameters['v_colorbar_label_angle'],
                        font_size=parameters['v_colorbar_font_size'],
                        tick_label_offset=parameters['v_colorbar_tick_label_offset'],
                        tick_label_angle=parameters['v_colorbar_tick_label_angle'],
                        axis=v_colorbar_axis,
                        tick_length=parameters['v_colorbar_tick_length'],
                        line_width=parameters['v_colorbar_tick_line_width'])
    
    
    gr.plot_2d_axes(
            ax, [0, 0], [parameters['L'], parameters['h']], 
            tick_length=parameters['tick_length'], 
            line_width=parameters['axis_line_width'], 
            axis_label=parameters['axis_label'], 
            axis_label_angle=parameters['axis_label_angle'], 
            axis_label_offset=parameters['axis_label_offset'], 
            tick_label_offset=parameters['tick_label_offset'], 
            tick_label_format=['f', 'f'], \
            font_size=parameters['axis_font_size'], 
            plot_label=parameters["v_panel_label"],
            plot_label_offset=parameters['panel_label_offset'], 
            axis_origin=parameters['axis_origin'], 
            axis_bounds=axis_min_max, 
            margin=parameters['axis_margin'],
            n_minor_ticks=parameters['n_minor_ticks'],
            minor_tick_length=parameters['minor_tick_length'],
            z_order=const.high_z_order)

    
    # =============
    # w subplot
    # =============   
    
    ax = fig.axes[3]  # Use the existing axis
    
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    
    color_map_w = gr.cb.make_curve_colorbar(fig, t, data_w, parameters['w_colorbar_position'], parameters['w_colorbar_size'], 
                                        min_max=w_min_max,
                                        tick_label_angle=parameters['w_colorbar_tick_label_angle'], 
                                        label=parameters['w_colorbar_axis_label'],
                                        font_size=parameters['w_colorbar_font_size'], 
                                        tick_label_offset=parameters['w_colorbar_tick_label_offset'],
                                        label_angle=parameters['w_colorbar_label_angle'],
                                        tick_length=parameters['w_colorbar_tick_length'],
                                        label_offset=parameters['w_colorbar_axis_label_offset'],
                                        axis=w_colorbar_axis)
    
    #plot X and w
    gr.plot_curve_grid(ax, X_t, 
                       color_map=color_map_w, 
                       line_color='black', 
                       line_width=parameters['w_line_width'])
    
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
                tick_label_format=['f', 'f'], \
                font_size=parameters['axis_font_size'], 
                plot_label=parameters["w_panel_label"],
                plot_label_offset=parameters['panel_label_offset'], 
                axis_origin=parameters['axis_origin'], 
                axis_bounds=axis_min_max, 
                margin=parameters['axis_margin'],
                n_minor_ticks=parameters['n_minor_ticks'],
                minor_tick_length=parameters['minor_tick_length'],
                z_order=const.high_z_order)
 
    # =============
    # sigma subplot
    # =============   
    
    ax = fig.axes[4]  # Use the existing axis
    
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    
    color_map_sigma = gr.cb.make_curve_colorbar(fig, t, data_sigma, parameters['sigma_colorbar_position'], parameters['sigma_colorbar_size'], 
                                        min_max=sigma_min_max,
                                        tick_label_angle=parameters['sigma_colorbar_tick_label_angle'], 
                                        label=parameters['sigma_colorbar_axis_label'],
                                        font_size=parameters['sigma_colorbar_font_size'], 
                                        tick_label_offset=parameters['sigma_colorbar_tick_label_offset'],
                                        label_angle=parameters['sigma_colorbar_label_angle'],
                                        tick_length=parameters['sigma_colorbar_tick_length'],
                                        label_offset=parameters['sigma_colorbar_axis_label_offset'],
                                        axis=sigma_colorbar_axis)
    
    #plot X and sigma
    gr.plot_curve_grid(ax, X_t, 
                       color_map=color_map_sigma, 
                       line_color='black', 
                       line_width=parameters['sigma_line_width'])
    
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
                tick_label_format=['f', 'f'], \
                font_size=parameters['axis_font_size'], 
                plot_label=parameters["sigma_panel_label"],
                plot_label_offset=parameters['panel_label_offset'], 
                axis_origin=parameters['axis_origin'], 
                axis_bounds=axis_min_max, 
                margin=parameters['axis_margin'],
                n_minor_ticks=parameters['n_minor_ticks'],
                minor_tick_length=parameters['minor_tick_length'],
                z_order=const.high_z_order)



plot_snapshot(fig, snapshot_min, 
              snapshot_label=rf'$t = \,$' + io.time_to_string(snapshot_min * parameters['T'] / number_of_frames, 's', 1))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
