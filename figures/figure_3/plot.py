import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings

import calculus.utils as cal
import list.column_labels as clab
import graphics.utils as gr
import input_output.utils as io
import list.utils as lis
import system.paths as paths
import system.utils as sys_utils
import graphics.vector_plot as vec

matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of


parameters = io.read_parameters_from_csv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))   




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

# define the folder where to read the data
solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
snapshot_path = os.path.join(solution_path, 'snapshots/csv/')
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])


# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_msh_n_', snapshot_path)

number_of_frames = snapshot_max - snapshot_min + 1

# labels of columns to read
columns_line_vertices = [clab.label_start_x_column, clab.label_start_y_column, clab.label_start_z_column,
                         clab.label_end_x_column,
                         clab.label_end_y_column, clab.label_end_z_column]
columns_v = [clab.label_x_column, clab.label_y_column, clab.label_v_column + clab.label_x_column,
             clab.label_v_column + clab.label_y_column]



# fork
# 2) to plot the animation: compute absolute min and max of norm v across  snapshots
'''
norm_v_min_max = cal.min_max_vector_field(snapshot_min, 
                         snapshot_max, parameters['frame_stride'], 
                         os.path.join(solution_path + 'snapshots/csv/nodal_values'), 'def_v_n_', 
                         parameters['n_bins_v'],
                         [[0, 0],[parameters['L'], parameters['h']]]
                        )

sigma_min_max = cal.min_max_files(
                'def_sigma_n_12_', 
                os.path.join(solution_path + 'snapshots/csv/nodal_values'),
                snapshot_min, 
                snapshot_max, 
                parameters['frame_stride']
                 )
'''





fig = pplt.figure(figsize=(parameters['figure_size'][0], parameters['figure_size'][1]), 
                  left=parameters['figure_margin_l'], 
                  bottom=parameters['figure_margin_b'], 
                  right=parameters['figure_margin_r'], 
                  top=parameters['figure_margin_t'], 
                  wspace=0, hspace=0)


def plot_snapshot(fig, n_file, snapshot_label):
    n_snapshot = str(n_file)

    # load data
    data_el_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_el_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_msh_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_msh_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_v = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/def_v_n_' + n_snapshot + '.csv', usecols=columns_v)
    data_u_msh = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/u_msh_n_' + n_snapshot + '.csv', usecols=columns_v)
    data_sigma = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/def_sigma_n_12_' + n_snapshot + '.csv')


    
    # =============
    # v subplot
    # =============    
    
    # Check if we already have an axis, if not create one
    if len(fig.axes) == 0:
        ax = fig.add_subplot(2, 1, 1)
    else:
        ax = fig.axes[0]  # Use the existing axis
        
        
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1], snapshot_label, fontsize=8, ha='center', va='center')

    # gr.set_2d_axes_limits(ax, [0, 0], [parameters['L'], parameters['h']], [0, 0])

    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_el_line_vertices, parameters['mesh_el_line_width'], 'red', parameters['alpha_mesh'])
    gr.plot_2d_mesh(ax, data_msh_line_vertices, parameters['mesh_msh_line_width'], 'black', parameters['alpha_mesh'])


    X, Y, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, _ = vec.interpolate_2d_vector_field(
                                                                                                data_v,
                                                                                                [0, 0],
                                                                                                [parameters['L'], parameters['h']],
                                                                                                parameters['n_bins_v']
                                                                                            )
    

    # fork
    # 1) to plot the figure, I set norm_v_min_max to the min and max for the current frame
    #    
    norm_v_min_max= [norm_v_min, norm_v_max]     
    # 


    _, _, U_msh_x, U_msh_y, _, _, _, _ = vec.interpolate_2d_vector_field(data_u_msh,
                                                                        [0, 0],
                                                                        [parameters['L'], parameters['h']],
                                                                        parameters['n_bins_v'],
                                                                        clab.label_x_column,
                                                                        clab.label_y_column,
                                                                        clab.label_v_column)

    # set to nan the values of the velocity vector field which lie within the elliipse at step 'n_file', where I read the rotation angle of the ellipse from data_theta_omega
    # 1. obtain the coordinates of the points X, Y of the vector field V_x, V_y in the reference configuration of the mesh
    X_ref = np.array(lis.substract_lists_of_lists(X, U_msh_x))
    Y_ref = np.array(lis.substract_lists_of_lists(Y, U_msh_y))
    # 2. once the coordinates in the reference configuration are known, assess whether they fall within the elastic body by checking whether they fall wihin the ellipse
    gr.set_inside_ellipse(X_ref, Y_ref, parameters['c'], parameters['a'], parameters['b'], 0, V_x, np.nan)
    gr.set_inside_ellipse(X_ref, Y_ref, parameters['c'], parameters['a'], parameters['b'], 0, V_y, np.nan)

    # plot velocity of fluid
    vec.plot_2d_vector_field(ax, [X, Y], [V_x, V_y], parameters['shaft_length'], parameters['head_over_shaft_length'], parameters['arrow_head_angle'], parameters['arrow_line_width'], 1, 'color_from_map', 0)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min_max[0], norm_v_min_max[1], \
                        parameters['v_colorbar_position'], parameters['v_colorbar_size'], 
                        label_pad=parameters['v_colorbar_label_offset'], 
                        label=r'$v \, [\met/\sec]$', 
                        font_size=parameters['color_map_font_size'],
                        tick_label_offset=parameters['v_colorbar_tick_label_offset'],
                        tick_label_angle=parameters['v_colorbar_tick_label_angle'],
                        tick_length=parameters['v_colorbar_tick_length'],
                        line_width=parameters['v_colorbar_line_width']
                    )

    gr.plot_2d_axes(ax, [0, 0], [parameters['L'], parameters['h']], \
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width'],
                    axis_origin=parameters['axis_origin'],
                    tick_length=parameters['tick_length']
                )
    
    
        
    # =============
    # sigma subplot
    # =============

    ax = fig.add_subplot(2, 1, 2)
        
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    
    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_el_line_vertices, parameters['mesh_el_line_width'], 'red', parameters['alpha_mesh'])
    gr.plot_2d_mesh(ax, data_msh_line_vertices, parameters['mesh_msh_line_width'], 'black', parameters['alpha_mesh'])
    
    X_sigma, Y_sigma, Z_sigma, _, _, _ = gr.interpolate_surface(data_sigma, [0, 0], [parameters['L'], parameters['h']], parameters['n_bins_sigma'])
    
    # fork
    # 1) to plot the figure, I set sigma_min_max to the min and max for the current frame
    # 
    sigma_min, sigma_max, _ = cal.min_max_scalar_field(Z_sigma)
    sigma_min_max = [sigma_min, sigma_max]
    # 

    contour_plot = ax.imshow(
                                Z_sigma.T, 
                                origin='lower', 
                                cmap=gr.cb.color_map_type, 
                                aspect='equal', 
                                extent=[0, parameters['L'], 0, parameters['h']],
                                vmin=sigma_min_max[0], vmax=sigma_min_max[1]
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
        label=r"$\sigma \, [\pas \, \met]$",
        mappable = contour_plot
    )
    
    
    
    gr.plot_2d_axes(ax, [0, 0], [parameters['L'], parameters['h']], \
                axis_label=parameters['axis_label'],
                axis_label_angle=parameters['axis_label_angle'],
                axis_label_offset=parameters['axis_label_offset'],
                tick_label_offset=parameters['tick_label_offset'],
                tick_label_format=parameters['tick_label_format'],
                font_size=parameters['font_size'],
                line_width=parameters['axis_line_width'],
                axis_origin=parameters['axis_origin'],
                tick_length=parameters['tick_length']
            )

    



plot_snapshot(fig, snapshot_max, rf'$t = \,$' + io.time_to_string(snapshot_max * parameters['T'] / number_of_frames, 's', parameters['n_decimals_snapshot_label']))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

