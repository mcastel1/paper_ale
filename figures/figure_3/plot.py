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


parameters = io.read_parameters_from_csv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))   



# add the path where to find the shared modules
module_path = parameters['root_path'] + "/figures/modules/"
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

# define the folder where to read the data
solution_path = os.path.join(parameters['root_path'], 'figures/figure_3/solution/')
snapshot_path = os.path.join(solution_path, 'snapshots/csv/')
figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])

# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_msh_n_', snapshot_path)

number_of_frames = snapshot_max

# labels of columns to read
columns_line_vertices = [clab.label_start_x_column, clab.label_start_y_column, clab.label_start_z_column,
                         clab.label_end_x_column,
                         clab.label_end_y_column, clab.label_end_z_column]
columns_v = [clab.label_x_column, clab.label_y_column, clab.label_v_column + clab.label_x_column,
             clab.label_v_column + clab.label_y_column]

fig = pplt.figure(figsize=(5, 1.5), left=8, bottom=0, right=2, top=-1, wspace=0, hspace=0)


def plot_snapshot(fig, n_file, snapshot_label):
    n_snapshot = str(n_file)

    # load data
    data_el_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_el_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_msh_line_vertices = pd.read_csv(solution_path + 'snapshots/csv/line_mesh_msh_n_' + n_snapshot + '.csv', usecols=columns_line_vertices)
    data_v = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/def_v_n_' + n_snapshot + '.csv', usecols=columns_v)
    data_u_msh = pd.read_csv(solution_path + 'snapshots/csv/nodal_values/u_msh_n_' + n_snapshot + '.csv', usecols=columns_v)

    ax = fig.add_subplot(1, 1, 1)

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # plot snapshot label
    fig.text(0.55, 0.85, snapshot_label, fontsize=8, ha='center', va='center')

    gr.set_2d_axes_limits(ax, [0, 0], [parameters['L'], parameters['h']], [0, 0])

    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_el_line_vertices, 0.2, 'red', parameters['alpha_mesh'])
    gr.plot_2d_mesh(ax, data_msh_line_vertices, 0.05, 'black', parameters['alpha_mesh'])


    X, Y, V_x, V_y, grid_norm_v, norm_v_min, norm_v_max, norm_v = vec.interpolate_2d_vector_field(data_v,
                                                                                                    [0, 0],
                                                                                                    [parameters['L'], parameters['h']],
                                                                                                    parameters['n_bins_v'],
                                                                                                    clab.label_x_column,
                                                                                                    clab.label_y_column,
                                                                                                    clab.label_v_column)
    


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
    vec.plot_2d_vector_field(ax, [X, Y], [V_x, V_y], parameters['shaft_length'], 0.3, 30, 0.5, 1, 'color_from_map', 0)

    gr.cb.make_colorbar(fig, grid_norm_v, norm_v_min, norm_v_max, \
                        parameters['v_colorbar_position'], parameters['v_colorbar_size'], 
                        label_pad=parameters['v_colorbar_label_pad'], 
                        label=r'$v \, [\met/\sec]$', 
                        font_size=parameters['color_map_font_size'])

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



plot_snapshot(fig, parameters['n_late_snapshot'], rf'$t = \,$' + io.time_to_string(parameters['n_late_snapshot'] * parameters['T'] / number_of_frames, 's', 0))

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()
