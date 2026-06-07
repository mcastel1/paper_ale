import matplotlib
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import time
import warnings

import calculus.utils as cal
import graphics.utils as gr
import graphics.vector_plot as vp
import list.column_labels as clab
import input_output.utils as io
import list.utils as lis
import system.paths as paths
import system.utils as sys_utils
import graphics.vector_plot as vec



'''
to copy files for this figure from abacus do :

 ./copy_from_abacus.sh surface_tension_1/solution/snapshots/csv 'boundary_points_id_7_n_*' 'line_mesh_n_*'   'line_mesh_0_n_*'  'def_v_n_*' 'u_n_*' 'u_0_n_*' 'def_sigma_n_*' 'def_mu_n_*'  ~/Desktop 0 100000 10

'''

matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of


parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))


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
        r"\usepackage{bm} "
        r"\usepackage{glossaries} "
        rf"\input{{{paths.definitions_path}}}"
        rf"\input{{{os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../definitions.tex')}}}"
    )
})

# define the folder where to read the data
'''# 1. read data from local folder 
solution_path = os.path.join(os.path.dirname( os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname( os.path.abspath(__file__)), "mesh/solution/")
sub_mesh_1_path = os.path.join(os.path.dirname( os.path.abspath(__file__)), "mesh/solution/sub_meshes/out/")
snapshot_path = os.path.join(solution_path, 'snapshots/csv/')
'''
# 2 read data from external folder
path = '/Users/michelecastellana/Documents/finite_elements/fluid_structure_interaction/elastic_obstacle/monolithic/surface_tension'
solution_path = os.path.join(path, "solution/")
mesh_path = "/Users/michelecastellana/Documents/finite_elements/generate_mesh/2d/square/shape_line/solution/"
sub_mesh_1_path = os.path.join(mesh_path, "sub_meshes/out")
snapshot_path = os.path.join(solution_path, 'snapshots/csv/')


solution_parameters = io.read_parameters_from_csv_file(os.path.join(path, 'parameters_bc_square_shape_line_a.csv'))
mesh_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_path, 'mesh_metadata.csv'))
mesh_0_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_path, 'mesh_0/mesh_metadata.csv'))

if parameters['frame_stride'] % solution_parameters['print_out_stride'] != 0:
    
    raise RuntimeError(f'Error: Animation frame stride is not a multiple of print out stride ! \n animation frame stride = {parameters["frame_stride"]} \n print out stride = {solution_parameters["print_out_stride"]} \nAborting...')

figure_path = os.path.join(os.path.dirname( os.path.abspath(__file__)), parameters['figure_name'])

# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max(
    'line_mesh_n_', snapshot_path)

number_of_frames = snapshot_max - snapshot_min + 1




# fork
# 2) to plot the animation: compute absolute min and max of norm v across  snapshots
# I compute mu_min_max from snapshots between snapshot_min + parameters['colorbar_mu_snapshot_min_offset'] and snapshot_max. I do not use snapshot_min because there is a tension shock at the first few steps of the dynamics that would yield a huge negative value of mu and an odd colorbars. I restrict the min max to the values outside the shape because values in the shape are large and may make the color code not clear. 
mu_min_max = cal.min_max_files(
                'def_mu_n_',
                os.path.join(solution_path + 'snapshots/csv'),
                snapshot_min +
                    parameters['colorbar_mu_snapshot_min_offset'],
                snapshot_max,
                parameters['frame_stride'],
                tag=mesh_parameters['sub_mesh_0_1_id']
                 )


norm_v_min_max = cal.min_max_vector_field(snapshot_min,
                                          snapshot_max, parameters['frame_stride'],
                                          os.path.join(
                                              solution_path + 'snapshots/csv'), 'def_v_n_',
                                          parameters['n_bins_v'],
                                          [[0, 0], [mesh_parameters['L'], mesh_parameters['h']]]
                                          )








fig = pplt.figure(figsize=(parameters['figure_size'][0], parameters['figure_size'][1]),
                  left=parameters['figure_margin_l'],
                  bottom=parameters['figure_margin_b'],
                  right=parameters['figure_margin_r'],
                  top=parameters['figure_margin_t'],
                  wspace=parameters['wspace'],
                  hspace=parameters['hspace'])


# pre-create subplots and axes
fig.add_subplot(3, 1, 1)
fig.add_subplot(3, 1, 2)
fig.add_subplot(3, 1, 3)


mu_colorbar_axis = fig.add_axes([parameters['mu_colorbar_position'][0],
                                    parameters['mu_colorbar_position'][1],
                                    parameters['mu_colorbar_size'][0],
                                    parameters['mu_colorbar_size'][1]])

v_colorbar_axis = fig.add_axes([parameters['v_colorbar_position'][0],
                                parameters['v_colorbar_position'][1],
                                parameters['v_colorbar_size'][0],
                                parameters['v_colorbar_size'][1]])


mu_colorbar = None
v_colorbar = None


def plot_snapshot(fig, n_file, snapshot_label):
    n_snapshot = str(n_file)

    global mu_colorbar, v_colorbar


    start_time_plot_snapshot = time.time()


    start_time = time.time()

    # 1 load data
    # 1.1 load boundary points
    data_ref_boundary_vertices_shape = pd.read_csv(os.path.join(solution_path, f'snapshots/csv/boundary_points_id_{mesh_0_parameters["shape_id"]}_n_{n_snapshot}.csv'))

    
    # 1.2 load  mesh data for mesh deformed with u_n
    data_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_n_' + n_snapshot + '.csv')
    data_u_cur = pd.read_csv(solution_path + 'snapshots/csv/u_n_' + n_snapshot + '.csv')

    data_v_raw = pd.read_csv(solution_path + 'snapshots/csv/def_v_n_' + n_snapshot + '.csv')
    # select from data_v_raw only data which belong to sub_mesh_0_1
    data_v = data_v_raw[data_v_raw['tag'] == mesh_parameters['sub_mesh_0_1_id']]

    data_mu_raw = pd.read_csv(solution_path + 'snapshots/csv/def_mu_n_' + n_snapshot + '.csv')
    # select from data_mu_raw only data which belong to sub_mesh_0_1
    data_mu = data_mu_raw[data_mu_raw['tag'] == mesh_parameters['sub_mesh_0_1_id']]
    

    # 1.3 load  mesh data for mesh deformed with u_0
    data_line_vertices_0 = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_0_n_' + n_snapshot + '.csv')
    data_u_cur_0 = pd.read_csv(solution_path + 'snapshots/csv/u_0_n_' + n_snapshot + '.csv')


    stop_time = time.time()
    print(f"Time for block 1 = {stop_time - start_time:.2f} s", flush=True)

    # build two a vector field which interpolates the displacement field in data_u_msh
    U_interp_x, U_interp_y = vec.interpolating_function_2d_vector_field(data_u_cur)

        # 1) plot the polygon of the boundary 'ellipse_loop_id'

    # run through points in data_boundary_vertices_ellipse (reference configuration) and add to them [U_interp_x, U_interp_y] in order to obtain the boundary polygon in the current configuration
    data_cur_boundary_vertices_shape = []
    for _, row in data_ref_boundary_vertices_shape.iterrows():
        data_cur_boundary_vertices_shape.append(
            np.add(
                [row[':0'], row[':1']],
                [U_interp_x(row[':0'], row[':1']),
                 U_interp_y(row[':0'], row[':1'])]
            )
        )


    # =============
    # mesh + mu subplot for deformation with u_n
    # =============

    start_time = time.time()


    ax = fig.axes[0]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position']
             [1], snapshot_label, fontsize=8, ha='center', va='center')

    _, _, Z_mu, _, _, _ = gr.interpolate_surface(
        data_mu, [0, 0], [mesh_0_parameters['L'],
                             mesh_0_parameters['h']], parameters['n_bins_mu'],
        method='griddata',
        margin=parameters['dg_margin']
    )

    '''    # fork
        # 1) to plot the figure, I set mu_min_max to the min and max for the current frame
        #
        mu_min, mu_max, _ = cal.min_max_scalar_field(Z_mu)
        mu_min_max = [mu_min, mu_max]
        #
    '''




    stop_time = time.time()
    print(f"Time for block 2 = {stop_time - start_time:.2f} s", flush=True)

    start_time = time.time()


    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_line_vertices,
                    parameters['mesh_el_line_width'], parameters['mesh_color'], parameters['alpha_mesh'],
                    zorder=2)
    
    stop_time = time.time()
    print(f"Time for block 3 = {stop_time - start_time:.2f} s", flush=True)

    start_time = time.time()

    
    # plot the boundary partial_omega_circle_out in the current configuration
    partial_omega_circle_out_cur = Polygon(data_cur_boundary_vertices_shape, fill=True,
                                           linewidth=parameters['partial_omega_line_width'],
                                           edgecolor=parameters['partial_omega_circle_out_color'],
                                           linestyle='-.',
                                           zorder=1)

    ax.add_patch(partial_omega_circle_out_cur)

    vertices_array = np.array(data_cur_boundary_vertices_shape)
    center = vertices_array.mean(axis=0)

    scale_outer = 1.1  # 5% bigger
    scale_inner = 0.9  # 5% smaller

    outer = center + (vertices_array - center) * scale_outer
    inner = (center + (vertices_array - center) * scale_inner)[::-1]  # reversed → makes a hole


    verts = np.vstack([outer, outer[0:1], inner, inner[0:1]])
    N = len(mesh_parameters['shape_coordinates'])
    codes = ([Path.MOVETO] + [Path.LINETO]*(N-1) + [Path.CLOSEPOLY] +
         [Path.MOVETO] + [Path.LINETO]*(N-1) + [Path.CLOSEPOLY])
    
    band_patch = PathPatch(Path(verts, codes), facecolor='none', edgecolor='none')
    ax.add_patch(band_patch)

    contour_plot = ax.imshow(
        Z_mu.T,
        origin='lower',
        cmap=gr.cb.color_map_type,
        aspect='equal',
        extent=[0, mesh_0_parameters['L'], 0, mesh_0_parameters['h']],
        vmin=mu_min_max[0], vmax=mu_min_max[1],
        interpolation='bilinear',
        zorder=0
    )
    

    # contour_plot.set_clip_path(partial_omega_circle_out_cur)
    contour_plot.set_clip_path(band_patch)


    stop_time = time.time()
    print(f"Time for block 4 = {stop_time - start_time:.2f} s", flush=True)

    start_time = time.time()



    if mu_colorbar is None:
        
        # first frame: create with real data, axis already positioned by ProPlot
        mu_colorbar, _ = gr.cb.make_colorbar(
            figure=fig,
            grid_values=Z_mu,
            min_value=mu_min_max[0],
            max_value=mu_min_max[1],
            position=parameters['mu_colorbar_position'],
            size=parameters['mu_colorbar_size'],
            label_pad=parameters['mu_colorbar_label_offset'],
            tick_label_offset=parameters['mu_colorbar_tick_label_offset'],
            line_width=parameters['mu_colorbar_tick_line_width'],
            tick_length=parameters['mu_colorbar_tick_length'],
            tick_label_angle=parameters['mu_colorbar_tick_label_angle'],
            label=parameters['mu_colorbar_axis_label'],
            axis=mu_colorbar_axis
        )
    

    stop_time = time.time()
    print(f"Time for block 5 = {stop_time - start_time:.2f} s", flush=True)

    start_time = time.time()


    gr.plot_2d_axes(ax, [0, 0], [mesh_0_parameters['L'], mesh_0_parameters['h']],
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    tick_label_angle=parameters['tick_label_angle'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width'],
                    axis_origin=parameters['axis_origin'],
                    tick_length=parameters['tick_length'],
                    plot_label=parameters["mu_plot_panel_label"],
                    plot_label_offset=parameters['panel_label_position'],
                    plot_label_font_size=parameters['panel_label_font_size'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                    )



    stop_time = time.time()
    print(f"Time for block 6 = {stop_time - start_time:.2f} s", flush=True)

    stop_time_plot_snapshot = time.time()
    print(f"Time for plot_snapshot = {stop_time_plot_snapshot- start_time_plot_snapshot:.2f} s", flush=True)


    '''
    # =============
    # v subplot
    # =============

    ax = fig.axes[1]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position']
             [1], snapshot_label, fontsize=8, ha='center', va='center')


    # 1) plot the polygon of the boundary 'ellipse_loop_id'
    #


    # plot the boundary partial_omega_circle_out in the current configuration
    partial_omega_circle_out_cur = Polygon(data_cur_boundary_vertices_shape, fill=True,
                                           linewidth=parameters['partial_omega_line_width'],
                                           edgecolor=parameters['partial_omega_circle_out_color'],
                                           linestyle='-.',
                                           zorder=1,
                                           facecolor=parameters['partial_omega_circle_fill_color'])

    ax.add_patch(partial_omega_circle_out_cur)

    
    X, Y, V_x, V_y, grid_norm_v, _, _, _ = vec.interpolate_2d_vector_field(
        data_v,
        [0, 0],
        [mesh_parameters['L'], mesh_parameters['h']],
        parameters['n_bins_v']
    )
    
    # set to nan the values of V_x V_y lying in the shape
    vp.set_in_polygon(data_cur_boundary_vertices_shape,
                      [X, Y],
                      [V_x, V_y])
    
    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_line_vertices,
                    parameters['mesh_el_line_width'], parameters['mesh_color'], parameters['alpha_mesh'],
                    zorder=2)
    
    
    # fork
    # 1) to plot the figure, I set norm_v_min_max to the min and max for the current frame

    _, _, U_msh_x, U_msh_y, _, _, _, _ = vec.interpolate_2d_vector_field(data_u_cur,
                                                                         [0, 0],
                                                                         [mesh_parameters['L'], mesh_parameters['h']],
                                                                         parameters['n_bins_v'],
                                                                         clab.label_x_column,
                                                                         clab.label_y_column,
                                                                         clab.label_v_column)

    # set to nan the values of the velocity vector field which lie within the elliipse at step 'n_file', where I read the rotation angle of the ellipse from data_theta_omega
    # 1. obtain the coordinates of the points X, Y of the vector field V_x, V_y in the reference configuration of the mesh
    X_ref = np.array(lis.substract_lists_of_lists(X, U_msh_x))
    Y_ref = np.array(lis.substract_lists_of_lists(Y, U_msh_y))
    # 2. once the coordinates in the reference configuration are known, assess whether they fall within the elastic body by checking whether they fall wihin the ellipse
    # gr.set_inside_ellipse(
    #     X_ref, Y_ref, mesh_parameters['c'], mesh_parameters['a'], mesh_parameters['b'], 0, V_x, np.nan)
    # gr.set_inside_ellipse(
    #     X_ref, Y_ref, mesh_parameters['c'], mesh_parameters['a'], mesh_parameters['b'], 0, V_y, np.nan)

    # norm_v_min_max = [norm_v_min, norm_v_max]

    # plot velocity of fluid
    vec.plot_2d_vector_field(ax, [X, Y], [V_x, V_y],
                             parameters['shaft_length'],
                             parameters['head_over_shaft_length'],
                             parameters['arrow_head_angle'],
                             parameters['arrow_line_width'],
                             1,
                             'color_from_map',
                             0)
    


    


    if v_colorbar is None:

        # first frame: create with real data, axis already positioned by ProPlot
        v_colorbar, _ = gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min_max[0], norm_v_min_max[1],
                        parameters['v_colorbar_position'], parameters['v_colorbar_size'],
                        label_pad=parameters['v_colorbar_label_offset'],
                        label=parameters['v_colorbar_axis_label'],
                        font_size=parameters['color_map_font_size'],
                        tick_label_offset=parameters['v_colorbar_tick_label_offset'],
                        tick_label_angle=parameters['v_colorbar_tick_label_angle'],
                        tick_length=parameters['v_colorbar_tick_length'],
                        line_width=parameters['v_colorbar_line_width'],
                        axis=v_colorbar_axis
                        )
    
    
    

    gr.plot_2d_axes(ax, [0, 0], [mesh_parameters['L'], mesh_parameters['h']],
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    tick_label_angle=parameters['tick_label_angle'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width'],
                    axis_origin=parameters['axis_origin'],
                    tick_length=parameters['tick_length'],
                    plot_label=parameters["v_plot_panel_label"],
                    plot_label_offset=parameters['panel_label_position'],
                    plot_label_font_size=parameters['panel_label_font_size'],                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                    )
    


    # =============
    # mesh subplot for deformation with u_0
    # =============

    ax = fig.axes[2]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position']
             [1], snapshot_label, fontsize=8, ha='center', va='center')



    # 1) plot the polygon of the boundary 'ellipse_loop_id'
    #
    # build two a vector field which interpolates the displacement field in data_u_msh
    U_interp_x_0, U_interp_y_0 = vec.interpolating_function_2d_vector_field(data_u_cur_0)


    # run through points in data_boundary_vertices_ellipse (reference configuration) and add to them [U_interp_x_0, U_interp_y_0] in order to obtain the boundary polygon in the current configuration
    data_cur_boundary_vertices_shape_0 = []
    for _, row in data_ref_boundary_vertices_shape.iterrows():
        data_cur_boundary_vertices_shape_0.append(
            np.add(
                [row[':0'], row[':1']],
                [U_interp_x_0(row[':0'], row[':1']),
                 U_interp_y_0(row[':0'], row[':1'])]
            )
        )



    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_line_vertices_0,
                    parameters['mesh_el_line_width'], parameters['mesh_color'], parameters['alpha_mesh'],
                    zorder=2)
    

    
    # plot the boundary partial_omega_circle_out in the current configuration
    partial_omega_circle_out_cur_0 = Polygon(data_cur_boundary_vertices_shape_0, fill=True,
                                           linewidth=parameters['partial_omega_0_line_width'],
                                           edgecolor=parameters['partial_omega_circle_out_color'],
                                           linestyle='-.',
                                           zorder=1,
                                           facecolor=parameters['partial_omega_circle_fill_color'])

    ax.add_patch(partial_omega_circle_out_cur_0)


    gr.plot_2d_axes(ax, [0, 0], [mesh_parameters['L'], mesh_parameters['h']],
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    tick_label_angle=parameters['tick_label_angle'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width'],
                    axis_origin=parameters['axis_origin'],
                    tick_length=parameters['tick_length'],
                    plot_label=parameters["u_0_plot_panel_label"],
                    plot_label_offset=parameters['panel_label_position'],
                    plot_label_font_size=parameters['panel_label_font_size'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                    )

    '''




plot_snapshot(fig, snapshot_max, rf'$n = \,$ { snapshot_max}')
# plot_snapshot(fig, parameters['snapshot_to_plot'], rf'$t = \,$' + io.time_to_string(parameters['snapshot_to_plot'] *
#               solution_parameters['T'] / solution_parameters['num_steps'], 'min_s', parameters['n_decimals_snapshot_label']))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')
