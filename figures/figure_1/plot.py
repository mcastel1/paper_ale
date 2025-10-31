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
import graphics.vector_plot as vec
import input_output.utils as io
import system.paths as paths
import system.utils as sys_utils
import graphics.vector_plot as vp

matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of


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
print("Script location:", os.path.dirname(os.path.abspath(__file__)))

parameters = io.read_parameters_from_csv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))

solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")

snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_path)
number_of_frames = snapshot_max - snapshot_min + 1


# labels of columns to read
columns_line_vertices = [clab.label_start_x_column, clab.label_start_y_column, clab.label_start_z_column,
                         clab.label_end_x_column,
                         clab.label_end_y_column, clab.label_end_z_column]
columns_v = [clab.label_x_column, clab.label_y_column, clab.label_v_column + clab.label_x_column,
             clab.label_v_column + clab.label_y_column]
columns_theta_omega = ["theta", "omega"]

data_theta_omega = pd.read_csv(solution_path + 'theta_omega.csv', usecols=columns_theta_omega)


fig = pplt.figure(figsize=parameters['figure_size'], left=parameters['figure_margin_l'], 
                  bottom=parameters['figure_margin_b'], right=parameters['figure_margin_r'], 
                  top=parameters['figure_margin_t'], wspace=0, hspace=0)


# fork
# 2) to plot the animation: compute absolute min and max of norm v across  snapshots
'''
norm_v_min_max = cal.min_max_vector_field(parameters['n_first_frame'], 
                         number_of_frames, parameters['frame_stride'], 
                         os.path.join(solution_path + 'snapshots/csv/nodal_values'), 'def_v_n_', 
                         parameters['n_bins_v'],
                         [[0, 0],[parameters['L'], parameters['h']]]
                        )

'''

def plot_column(fig, n_file, snapshot_label):
    
    n_snapshot = str(n_file)
    data_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_v = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/def_v_n_' + n_snapshot + '.csv', usecols=columns_v)
    data_sigma = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/def_sigma_n_12_' + n_snapshot + '.csv')

    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1], snapshot_label, fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')


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

    
    gr.plot_2d_mesh(ax, data_line_vertices, parameters['mesh_line_width'], 'black', parameters['alpha_mesh'])

    X, Y, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, _ = vp.interpolate_2d_vector_field(data_v,
                                                                                                    [0, 0],
                                                                                                    [parameters['L'], parameters['h']],
                                                                                                    parameters['n_bins_v'],
                                                                                                    clab.label_x_column,
                                                                                                    clab.label_y_column,
                                                                                                    clab.label_v_column)
    # fork
    # 1) to plot the figure, I set norm_v_min_max to the min and max for the current frame
    # 
    norm_v_min_max = [norm_v_min, norm_v_max]
    # 


    # set to nan the values of the velocity vector field which lie within the elliipse at step 'n_file', where I read the rotation angle of the ellipse from data_theta_omega
    gr.set_inside_ellipse(X, Y, parameters['c'], parameters['a'], parameters['b'], data_theta_omega.loc[n_file-1, 'theta'], V_x, np.nan)
    gr.set_inside_ellipse(X, Y, parameters['c'], parameters['a'], parameters['b'], data_theta_omega.loc[n_file-1, 'theta'], V_y, np.nan)

    vec.plot_2d_vector_field(ax, [X, Y], [V_x, V_y], parameters['arrow_length'], 0.3, 30, 1, 1, 'color_from_map', 0,
                             clip_on=False)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min_max[0], norm_v_min_max[1], parameters['v_colorbar_position'], parameters['v_colorbar_size'], 
                        label=r'$v \, [\met/\sec]$', 
                        font_size=parameters['v_colorbar_font_size'],
                        tick_length=parameters['v_colorbar_tick_length'],
                        label_pad=parameters['v_colorbar_label_offset'], 
                        tick_label_offset=parameters['v_colorbar_tick_label_offset'],
                        tick_label_angle=parameters['v_colorbar_tick_label_angle'],
                        line_width=parameters['v_colorbar_line_width'])

    gr.plot_2d_axes(ax, [0, 0], [parameters['L'], parameters['h']],     
                          tick_length=parameters['tick_length'], 
                          line_width=parameters['axis_line_width'], 
                          axis_label=[r'$x \, [\met]$', r'$y \, [\met]$'],
                          tick_label_format=['f', 'f'], 
                          font_size=[parameters['font_size'], parameters['font_size']],
                          tick_label_offset=parameters['tick_label_offset'],
                          axis_label_offset=parameters['axis_label_offset'],
                          axis_origin=parameters['axis_origin'],
                          n_minor_ticks=parameters['n_minor_ticks'])
    
    
    
    # =============
    # sigma subplot
    # =============
    
    ax = fig.add_subplot(2, 1, 2)
        
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    
    # fork
    # 1) to plot the figure: compute sigma_min and sigma_max for the current snapshot
    sigma_min, sigma_max = gr.min_max_files('def_sigma_n_12_', 
                                    os.path.join(solution_path + 'snapshots/csv/nodal_values') + '/',
                                    clab.label_sigma_column, 
                                    n_file,
                                    n_file, 
                                    1)

    
    gr.plot_2d_mesh(ax, data_line_vertices, parameters['mesh_line_width'], 'black', parameters['alpha_mesh'])
    
    X_sigma, Y_sigma, Z_sigma = gr.interpolate_surface(data_sigma, [0, 0], [parameters['L'], parameters['h']], sigma_min, parameters['n_bins_sigma'], 1)
    
    color_map_sigma = gr.cb.make_colorbar(fig, Z_sigma, sigma_min, sigma_max, 
                                          parameters['sigma_colorbar_position'], parameters['sigma_color_bar_size'], 
                                          label_pad=parameters['sigma_colorbar_label_pad'],
                                          label=r'$\sigma \, [\textrm{Pa} \, \met]$', 
                                          font_size=parameters['sigma_colorbar_font_size'],
                                          line_width=parameters['sigma_colorbar_tick_line_width'],
                                          tick_length=parameters['sigma_colorbar_tick_length'],
                                          tick_label_offset=parameters['sigma_colorbar_tick_label_offset'],
                                          tick_label_angle=parameters['sigma_colorbar_tick_label_angle'])

    gr.plot_2d_axes(ax, [0, 0], [parameters['L'], parameters['h']],     
                          tick_length=parameters['tick_length'], 
                          line_width=parameters['axis_line_width'], 
                          axis_label=[r'$x \, [\met]$', r'$y \, [\met]$'],
                          tick_label_format=['f', 'f'], 
                          font_size=[parameters['font_size'], parameters['font_size']],
                          tick_label_offset=parameters['tick_label_offset'],
                          axis_label_offset=parameters['axis_label_offset'],
                          axis_origin=parameters['axis_origin'],
                          n_minor_ticks=parameters['n_minor_ticks'])


plot_column(fig, parameters['n_snapshot_to_plot'], rf'$t = \,$' + io.time_to_string(parameters['n_snapshot_to_plot'] * parameters['T'] / number_of_frames, 's', 1))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
