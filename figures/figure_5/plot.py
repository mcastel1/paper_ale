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
import graphics.vector_plot as vp
import input_output.utils as io
import list.utils as lis
import system.paths as paths
import system.utils as sys_utils
import graphics.vector_plot as vec

'''
you can copy the data from abacus with 
./copy_from_abacus.sh membrane_1/solution/snapshots/csv/  'line_mesh_n_*' 'u_n_*' 'X_n_12_*' 'v_n_*' 'w_n_*' 'sigma_n_12_*' 'nu_n_12_*' 'psi_n_12_*' 'def_v_fl_n_*' ~/Documents/paper_ale/figures/figure_5 1 1000000 10 
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
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")
snapshot_nodal_values_path = os.path.join(snapshot_path, "nodal_values")


# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_path)
number_of_frames = snapshot_max - snapshot_min + 1



# labels of columns to read
columns_line_vertices = [clab.label_start_x_column, clab.label_start_y_column, clab.label_start_z_column,
                         clab.label_end_x_column,
                         clab.label_end_y_column, clab.label_end_z_column]
columns_v = [clab.label_x_column, clab.label_y_column, clab.label_v_column + clab.label_x_column,
             clab.label_v_column + clab.label_y_column]

fig = pplt.figure(
    figsize=(parameters['figure_size'][0], parameters['figure_size'][1]), 
    left=parameters['figure_margin_l'], 
    bottom=parameters['figure_margin_b'], 
    right=parameters['figure_margin_r'], 
    top=parameters['figure_margin_t'], 
    wspace=parameters['wspace'], 
    hspace=parameters['hspace'])

# pre-create subplots and axes
fig.add_subplot(2, 1, 1)
fig.add_subplot(2, 1, 2)

v_colorbar_axis = fig.add_axes([parameters['v_colorbar_position'][0], 
                           parameters['v_colorbar_position'][1],
                           parameters['v_colorbar_size'][0],
                           parameters['v_colorbar_size'][1]])

w_colorbar_axis = fig.add_axes([parameters['w_colorbar_position'][0], 
                           parameters['w_colorbar_position'][1],
                           parameters['w_colorbar_size'][0],
                           parameters['w_colorbar_size'][1]])


           
            

def plot_snapshot(fig, n_file, 
                  snapshot_label='',
                  axis_min_max=None,
                  norm_v_min_max=None,
                  w_min_max=None):
    

    # load data
    # data_el_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_el_n_' + str(n_file) + '.csv', usecols=columns_line_vertices)
    data_msh_line_vertices = pd.read_csv(os.path.join(snapshot_path, 'line_mesh_n_' + str(n_file) + '.csv'), usecols=columns_line_vertices)
    data_v = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'def_v_fl_n_' + str(n_file) + '.csv'), usecols=columns_v)
    data_w = pd.read_csv(os.path.join(snapshot_path, 'w_n_' + str(n_file) + '.csv'))
    data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + str(n_file) + '.csv'), usecols=columns_v)


    if axis_min_max == None:

        # compute the min and max of the axes
        # 
        data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + str(n_file) + '.csv'))
        data_X = pd.read_csv(os.path.join(snapshot_path, 'X_n_12_' + str(n_file) + '.csv'))

        X_ref, Y_ref, u_n_X, u_n_Y, _, _, _, _ = vp.interpolate_2d_vector_field(data_u_msh,
                                                                                [0, 0],
                                                                                [parameters['L'], parameters['h']],
                                                                                parameters['n_bins_v'])
        
        #X, Y are the positions of the mesh nodes in the current configuration    
        X = np.array(lis.add_lists_of_lists(X_ref, u_n_X))
        Y = np.array(lis.add_lists_of_lists(Y_ref, u_n_Y))

        # compute the min-max of the snapshot
        axis_min_max = [lis.min_max(X),lis.min_max(Y)]
        # 
        
    X_t, t = gr.interpolate_curve(data_X, axis_min_max[0][0], axis_min_max[0][1], parameters['n_bins_X'])

    
    # =============
    # v subplot
    # =============    
    
    ax = fig.axes[0]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # plot snapshot label
    fig.text(parameters['snapshot_label_position'][0], parameters['snapshot_label_position'][1], snapshot_label, fontsize=parameters['plot_label_font_size'], ha='center', va='center')

        
    # here X_ref, Y_ref are the coordinates of the points in the reference configuration of the mesh
    X_ref, Y_ref, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, _ = vec.interpolate_2d_vector_field(data_v,
                                                                                                    [0, 0],
                                                                                                    [parameters['L'], parameters['h']],
                                                                                                    parameters['n_bins_v'],
                                                                                                    clab.label_x_column,
                                                                                                    clab.label_y_column,
                                                                                                    clab.label_v_column)
    

    if norm_v_min_max == None:
        norm_v_min_max = [norm_v_min, norm_v_max]

    
    X_ref, Y_ref, u_n_X, u_n_Y, _, _, _, _ = vec.interpolate_2d_vector_field(data_u_msh,
                                                                                                                    [0, 0],
                                                                                                                    [parameters['L'], parameters['h']],
                                                                                                                    parameters['n_bins_v'],
                                                                                                                    clab.label_x_column,
                                                                                                                    clab.label_y_column,
                                                                                                                    clab.label_v_column)

    #X, Y are the positions of the mesh nodes in the current configuration    
    X = np.array(lis.add_lists_of_lists(X_ref, u_n_X))
    Y = np.array(lis.add_lists_of_lists(Y_ref, u_n_Y))



    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices, parameters['plot_line_width'], 'black', parameters['alpha_mesh'])

    # plot velocity of fluid
    vec.plot_2d_vector_field(ax, [X, Y], [V_x, V_y], parameters['arrow_length'], 0.3, 30, 0.5, 1, 'color_from_map', 0)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min_max[0], norm_v_min_max[1], \
                        position=parameters['v_colorbar_position'], 
                        size=parameters['v_colorbar_size'], \
                        label_pad=parameters['v_colorbar_axis_label_offset'], 
                        label=parameters['v_colorbar_axis_label'],
                        font_size=parameters['v_colorbar_font_size'], 
                        tick_label_angle=parameters['v_colorbar_tick_label_angle'],
                        tick_label_offset=parameters['v_colorbar_tick_label_offset'],
                        line_width=parameters['v_colorbar_line_width'],
                        tick_length=parameters['v_colorbar_tick_length'],
                        axis=v_colorbar_axis)
                        
    

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
                    plot_label_font_size=parameters['plot_label_font_size'], 
                    plot_label_offset=parameters['plot_label_offset'], 
                    axis_origin=parameters['axis_origin'], 
                    axis_bounds=axis_min_max, 
                    margin=parameters['axis_margin'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length'])
    
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
                                        font_size=parameters['w_colorbar_font_size'], 
                                        label_offset=parameters["w_colorbar_label_offset"], 
                                        tick_label_offset=parameters['w_colorbar_tick_label_offset'],
                                        label_angle=parameters['w_colorbar_label_angle'],
                                        line_width=parameters['w_colorbar_tick_line_width'],
                                        tick_length=parameters['w_colorbar_tick_length'],
                                        axis=w_colorbar_axis)
    
    #plot X and w
    gr.plot_curve_grid(ax, X_t, 
                       color_map=color_map_w, 
                       line_color='black', 
                       line_width=parameters['w_line_width'])
 



plot_snapshot(fig, snapshot_max, 
              snapshot_label=rf'$t = \,$' + io.time_to_string(snapshot_max * parameters['T'] / number_of_frames, 's', 1))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
