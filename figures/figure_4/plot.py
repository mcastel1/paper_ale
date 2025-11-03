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
solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
# solution_path = "/Users/michelecastellana/Documents/finite_elements/dynamics/lagrangian_approach/one_dimension/solution/"
mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")
# mesh_path = "/Users/michelecastellana/Documents/finite_elements/generate_mesh/1d/line/solution/"
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure_4')
snapshot_path = os.path.join(solution_path, "snapshots/csv/nodal_values/")

parameters = io.read_parameters_from_csv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))   


snapshot_min, snapshot_max = sys_utils.n_min_max('sigma_n_12_', snapshot_path)

# add a margin to the last snapshot in order not to plot the very last one
snapshot_max=snapshot_max-parameters['snapshot_max_margin']

# labels of columns to read
columns_X = ["f:0","f:1","f:2",":0",":1",":2"]
columns_v = ["f:0","f:1","f:2",":0",":1",":2"]
columns_sigma = ["f",":0", ":1",":2"]
columns_nu = ["f",":0", ":1",":2"]
columns_psi = ["f",":0", ":1",":2"]

n_min, n_max = sys_utils.n_min_max('X_n_12_', snapshot_path)
# number_of_frames = sys_utils.count_v_files('X_n_12_', pfig.snapshot_path)
number_of_frames = n_max-n_min + 1  # +1 because the frames start from 0



fig = pplt.figure(
    figsize=(parameters['figure_size'][0], parameters['figure_size'][1]), 
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
                  norm_v_min_max=None,
                  sigma_min_max=None,
                  w_min_max=None):
    
    n_snapshot = str(n_file)
    data_X = pd.read_csv(os.path.join(snapshot_path, 'X_n_12_' + n_snapshot + '.csv'), usecols=columns_X)
    data_nu = pd.read_csv(os.path.join(snapshot_path, 'nu_n_12_' + n_snapshot + '.csv'))
    data_psi = pd.read_csv(os.path.join(snapshot_path, 'psi_n_12_' + n_snapshot + '.csv'))
    data_sigma = pd.read_csv(os.path.join(snapshot_path, 'sigma_n_12_' + n_snapshot + '.csv'))
    data_w = pd.read_csv(os.path.join(snapshot_path, 'w_n_' + n_snapshot + '.csv'))
    data_v = pd.read_csv(os.path.join(snapshot_path, 'v_n_' + n_snapshot + '.csv'), usecols=columns_v)
    # data_omega contains de values of \partial_1 X^alpha
    data_omega  = lis.data_omega(data_nu, data_psi)
    

    if X_min_max == None:
        X_min_max = [
            cal.min_max_file(os.path.join(snapshot_path, 'X_n_12_' + str(n_file) + '.csv'), column_name='f:0'),
            cal.min_max_file(os.path.join(snapshot_path, 'X_n_12_' + str(n_file) + '.csv'), column_name='f:1')
            ]
    if norm_v_min_max == None:
        norm_v_min_max=cal.norm_min_max_file(os.path.join(snapshot_path, 'v_n_' + str(n_file) + '.csv'))
    if sigma_min_max == None:
        sigma_min_max = cal.min_max_file(os.path.join(snapshot_path, 'sigma_n_12_' + str(n_file) + '.csv'))
    if w_min_max == None:
        w_min_max = cal.min_max_file(os.path.join(snapshot_path, 'w_n_' + str(n_file) + '.csv'))
    
    
    X, t = gr.interpolate_curve(data_X, X_min_max[0][0], X_min_max[0][1], parameters['n_bins_X'])



    # plot snapshot label
    if snapshot_label != '':
        fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1], snapshot_label, fontsize=parameters['snapshot_label_font_size'], ha='center', va='center')

   
    
    # =============
    # v subplot
    # =============   

    ax = fig.axes[0]  # Use the existing axis
    
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    #plot X and sigma 
    gr.plot_curve_grid(ax, X,  
                       line_color='black', 
                       line_width=parameters['X_line_width'],
                       alpha=parameters['alpha_X']
                       )



    # plot v
    X_v, Y_v, V_x, V_y, grid_norm_v, _, _, _ = vp.interpolate_t_vector_field_2d_arc_length_gauge(data_X, data_omega, data_v, parameters['n_bins_v'])
    
    
    vp.plot_1d_vector_field(ax, [X_v, Y_v], [V_x, V_y], 
                            shaft_length=parameters['shaft_length'], 
                            head_over_shaft_length=parameters['head_over_shaft_length'], 
                            head_angle=parameters['head_angle'], 
                            line_width=parameters['arrow_line_width'], 
                            alpha=parameters['alpha'], 
                            color='color_from_map', 
                            threshold_arrow_length=parameters['threshold_arrow_length'])

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min_max[0], norm_v_min_max[1], \
                        position=parameters['v_colorbar_position'], 
                        size=parameters['v_colorbar_size'], 
                        label_pad=parameters['v_colorbar_label_offset'], 
                        label=r'$v \, [\met / \sec]$', 
                        label_angle=parameters['v_colorbar_label_angle'],
                        font_size=parameters['color_map_font_size'],
                        tick_label_offset=parameters['v_colorbar_tick_label_offset'],
                        tick_label_angle=parameters['v_colorbar_tick_label_angle'],
                        axis=v_colorbar_axis,
                        tick_length=parameters['v_colorbar_tick_length'],
                        line_width=parameters['v_colorbar_tick_line_width'])

    gr.plot_2d_axes(ax, 
                    [X_min_max[0][0], X_min_max[1][0]], 
                    [X_min_max[0][1] - X_min_max[0][0], X_min_max[1][1] - X_min_max[1][0]],
                    axis_origin=parameters['axis_origin'],
                    axis_label=parameters['axis_label'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width'],
                    tick_length=parameters['tick_length'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                )
    
    
    # =============
    # w subplot
    # =============   
    
    ax = fig.axes[1]  # Use the existing axis
    
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    
    color_map_w = gr.cb.make_curve_colorbar(fig, t, data_w, parameters['w_colorbar_position'], parameters['w_colorbar_size'], 
                                        min_max=w_min_max,
                                        tick_label_angle=parameters['w_colorbar_tick_label_angle'], 
                                        label=r'$w \, [\newt/\met]$', 
                                        font_size=parameters['color_map_font_size'], 
                                        label_offset=parameters["w_colorbar_label_offset"], 
                                        tick_label_offset=parameters['w_colorbar_tick_label_offset'],
                                        label_angle=parameters['w_colorbar_label_angle'],
                                        line_width=parameters['w_colorbar_tick_line_width'],
                                        tick_length=parameters['w_colorbar_tick_length'],
                                        axis=w_colorbar_axis)
    

    
    #plot X and w
    gr.plot_curve_grid(ax, X, 
                       color_map=color_map_w, 
                       line_color='black', 
                       line_width=parameters['w_line_width'])
    
    gr.plot_2d_axes(ax, 
                [X_min_max[0][0], X_min_max[1][0]], 
                [X_min_max[0][1] - X_min_max[0][0], X_min_max[1][1] - X_min_max[1][0]],
                axis_origin=parameters['axis_origin'],
                axis_label=parameters['axis_label'],
                axis_label_angle=parameters['axis_label_angle'],
                axis_label_offset=parameters['axis_label_offset'],
                tick_label_offset=parameters['tick_label_offset'],
                tick_label_format=parameters['tick_label_format'],
                font_size=parameters['font_size'],
                line_width=parameters['axis_line_width'],
                tick_length=parameters['tick_length'],
                n_minor_ticks=parameters['n_minor_ticks'],
                minor_tick_length=parameters['minor_tick_length']
            )
    
    
    # =============
    # sigma subplot
    # =============   
    
    ax = fig.axes[2]  # Use the existing axis
    
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid
    
    color_map_sigma = gr.cb.make_curve_colorbar(fig, t, data_sigma, parameters['sigma_colorbar_position'], parameters['sigma_colorbar_size'], 
                                            min_max=sigma_min_max,
                                            tick_label_angle=parameters['sigma_colorbar_tick_label_angle'], 
                                            label=r'$\sigma \, [\newt/\met]$', 
                                            font_size=parameters['color_map_font_size'], 
                                            label_offset=parameters["sigma_colorbar_label_offset"], 
                                            tick_label_offset=parameters['sigma_colorbar_tick_label_offset'],
                                            label_angle=parameters['sigma_colorbar_label_angle'],
                                            line_width=parameters['sigma_colorbar_tick_line_width'],
                                            tick_length=parameters['sigma_colorbar_tick_length'],
                                            axis=sigma_colorbar_axis)
    
    #plot X and sigma 
    gr.plot_curve_grid(ax, X, 
                       color_map=color_map_sigma, 
                       line_color='black', 
                       line_width=parameters['sigma_line_width'])
    
    gr.plot_2d_axes(ax, 
                [X_min_max[0][0], X_min_max[1][0]], 
                [X_min_max[0][1] - X_min_max[0][0], X_min_max[1][1] - X_min_max[1][0]],
                axis_origin=parameters['axis_origin'],
                axis_label=parameters['axis_label'],
                axis_label_angle=parameters['axis_label_angle'],
                axis_label_offset=parameters['axis_label_offset'],
                tick_label_offset=parameters['tick_label_offset'],
                tick_label_format=parameters['tick_label_format'],
                font_size=parameters['font_size'],
                line_width=parameters['axis_line_width'],
                tick_length=parameters['tick_length'],
                n_minor_ticks=parameters['n_minor_ticks'],
                minor_tick_length=parameters['minor_tick_length']
            )




plot_snapshot(fig, 
              snapshot_max, 
              rf'$t = \,$' + io.time_to_string(snapshot_max * parameters['T'] / number_of_frames, 's', 1)
              )


# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
