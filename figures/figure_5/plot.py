import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings

import list.column_labels as clab
import graphics.utils as gr
import input_output.utils as io
import list.utils as lis
import system.paths as paths
import system.utils as sys_utils
import graphics.vector_plot as vec

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
# 1) choose the path as the path where this code is located 
# 
root_path = os.path.dirname(os.path.abspath(__file__))
mesh_path = os.path.join(root_path, "mesh/solution/")
# 
# 2) choose the  path as an external one
'''
root_path = '/Users/michelecastellana/Documents/finite_elements/fluid_structure_interaction/membrane'
mesh_path = '/Users/michelecastellana/Documents/finite_elements/generate_mesh/2d/square_no_circle/line/solution'
'''

solution_path = os.path.join(root_path, "solution/")
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")
snapshot_nodal_values_path = os.path.join(snapshot_path, "nodal_values")


# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_path)



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
    wspace=0, hspace=0)


# fork: 2) to plot the animation
# compute absolute min and max of the axes across all snapshots
'''
# initialize the values of axis_min_max
axis_min_max = [[np.inf,-np.inf],[np.inf,-np.inf]]

# run through all snapshots
for n_snapshot in range(snapshot_min, snapshot_max, parameters['frame_stride']):

    data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + str(n_snapshot) + '.csv'), usecols=columns_v)

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

    # compute the min-max of the snapshot
    axis_min_max_snapshot = [lis.min_max(X),lis.min_max(Y)]
    
    # update the absolute min and max according to the min-max of the snapshot 
    for i in range(2):
        if axis_min_max_snapshot[i][0] < axis_min_max[i][0]:
            axis_min_max[i][0] = axis_min_max_snapshot[i][0]
            
        if axis_min_max_snapshot[i][1] > axis_min_max[i][1]:
            axis_min_max[i][1] = axis_min_max_snapshot[i][1]
'''
            
            

def plot_snapshot(fig, n_file, snapshot_label):
    n_snapshot = str(n_file)

    # load data
    # data_el_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_el_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_msh_line_vertices = pd.read_csv(os.path.join(snapshot_path, 'line_mesh_n_' + n_snapshot + '.csv'), usecols=columns_line_vertices)
    data_v = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'def_v_fl_n_' + n_snapshot + '.csv'), usecols=columns_v)
    data_u_msh = pd.read_csv(os.path.join(snapshot_nodal_values_path, 'u_n_' + n_snapshot + '.csv'), usecols=columns_v)


    ax = fig.add_subplot(1, 1, 1)

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # plot snapshot label
    fig.text(0.55, 0.85, snapshot_label, fontsize=parameters['plot_label_font_size'], ha='center', va='center')

        
    # here X_ref, Y_ref are the coordinates of the points in the reference configuration of the mesh
    X_ref, Y_ref, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, norm_v = vec.interpolate_2d_vector_field(data_v,
                                                                                                    [0, 0],
                                                                                                    [parameters['L'], parameters['h']],
                                                                                                    parameters['n_bins_v'],
                                                                                                    clab.label_x_column,
                                                                                                    clab.label_y_column,
                                                                                                    clab.label_v_column)
    

    
    X_ref, Y_ref, u_n_X, u_n_Y, grid_norm_u_n, norm_u_n_min, norm_u_n_max, norm_u_n = vec.interpolate_2d_vector_field(data_u_msh,
                                                                                                                    [0, 0],
                                                                                                                    [parameters['L'], parameters['h']],
                                                                                                                    parameters['n_bins_v'],
                                                                                                                    clab.label_x_column,
                                                                                                                    clab.label_y_column,
                                                                                                                    clab.label_v_column)

    #X, Y are the positions of the mesh nodes in the current configuration    
    X = np.array(lis.add_lists_of_lists(X_ref, u_n_X))
    Y = np.array(lis.add_lists_of_lists(Y_ref, u_n_Y))

    # fork: 1) to plot the figure
    # 
    #obtain the min and max of the X and Y values of the mesh in the current configuration, in order to get the correct boundaries of the plot 
    axis_min_max = [lis.min_max(X),lis.min_max(Y)]
    # 

    # plot mesh under the membrane
    gr.plot_2d_mesh(ax, data_msh_line_vertices, parameters['plot_line_width'], 'black', parameters['alpha_mesh'])

    # plot velocity of fluid
    vec.plot_2d_vector_field(ax, [X, Y], [V_x, V_y], parameters['arrow_length'], 0.3, 30, 0.5, 1, 'color_from_map', 0)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min, norm_v_max, \
                        1, parameters['color_bar_position'], parameters['color_bar_size'], \
                        90, [-3.0, 0.5], r'$v \, [\met/\sec]$', parameters['colorbar_font_size'], tick_label_angle=parameters['colorbar_tick_label_angle'])
                        
    

    gr.plot_2d_axes(ax, [0, 0], [parameters['L'], parameters['h']], \
                          tick_length=parameters['tick_length'], line_width=parameters['axis_line_width'], \
                          axis_label=parameters['axis_label'], axis_label_angle=parameters['axis_label_angle'], \
                          axis_label_offset=parameters['axis_label_offset'], tick_label_offset=parameters['tick_label_offset'], tick_label_format=['f', 'f'], \
                          font_size=parameters['axis_font_size'], plot_label_font_size=parameters['plot_label_font_size'], 
                          plot_label_offset=parameters['plot_label_offset'], axis_origin=parameters['axis_origin'], 
                          axis_bounds=axis_min_max, margin=parameters['axis_margin'])



# plot_snapshot(fig, snapshot_max, rf'$t = \,$' + io.time_to_string(snapshot_max * T / number_of_frames, 's', 0))
plot_snapshot(fig, snapshot_max, rf'$n = \,$' + str(snapshot_max))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
