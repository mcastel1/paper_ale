'''
this animation plots 
    - in a first part, the mesh, by adding subsequently mesh edges over time
    - in a second part, a function `f` on the mesh surface, by coloring subsequent edges of the mesh in terms of a color code corresponding to `f`
'''


import matplotlib
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import proplot as pplt
import shutil
import warnings

import list.column_labels as clab
import input_output.utils as io
import graphics.utils as gr
import system.paths as paths


# clean up the cache directory and create a new one
cache_dir = os.path.expanduser("~/.matplotlib/tex.cache")
shutil.rmtree(cache_dir, ignore_errors=True)
os.makedirs(cache_dir, exist_ok=True)

matplotlib.use(
    "Agg"
)  # use a non-interactive backend so that figures do not pop up during run in debug mode

# Suppress the specific warning
warnings.filterwarnings(
    "ignore", message=".*Z contains NaN values.*", category=UserWarning)


plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{newpxtext,newpxmath} "
        r"\usepackage{xcolor} "
        r"\usepackage{glossaries} "
        rf"\input{{{paths.definitions_path}}}"
    )
})




parameters = io.read_parameters_from_csv_file(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.csv")
)
mesh_parameters = io.read_parameters_from_csv_file(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_parameters.csv")
)



# define the folder where to read the data
print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))

mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")
# mesh_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/generate_mesh/1d/line', "solution")

# solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
solution_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/poisson_equation/solve_u', "solution")

figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])

# labels of columns to read
data_vertices = pd.read_csv(os.path.join(mesh_path, "vertices.csv")).set_index('tag')

# load the data on the edges of the mesh: each edge is a list of 2 vertex labels, labelled as in `data_vertices`: these vertices delimit the edge
data_edges = pd.read_csv(os.path.join(mesh_path, "edges.csv"))

# load data for the field f
data_u = pd.read_csv(os.path.join(solution_path, "u.csv"))

h = np.max(data_u['f'])

print(f'u min and max = {np.min(data_u["f"])}, {np.max(data_u["f"])}')


min_max = [[min(data_vertices[":0"]),max(data_vertices[":0"])],[min(data_vertices[":1"]),max(data_vertices[":1"])],[min(data_vertices[":2"]),max(data_vertices[":2"])]]

# # function to be plotted on the 3d surface with color code
# def f(x, y, z):
#     return 1 + np.cos(2 * np.pi * x) / (1 + x ** 2)

'''
edge_coordinates = [
                        [[p_1_edge_0_x, p_1_edge_0_y, p_1_edge_0_z], [p_2_edge_0_x, p_2_edge_0_y, p_2_edge_0_z]],
                        [[p_1_edge_1_x, p_1_edge_1_y, p_1_edge_1_z], [p_2_edge_1_x, p_2_edge_1_y, p_2_edge_1_z]],
                        ...
                    ]

'''

edge_coordinates = []
for i in range(len(data_edges)):
    
    p_1_tag = data_edges['p_1'][i]
    p_2_tag = data_edges['p_2'][i]

    edge_coordinates.append([
        [data_vertices[':0'][p_1_tag], data_vertices[':1'][p_1_tag], data_vertices[':2'][p_1_tag]], 
        [data_vertices[':0'][p_2_tag], data_vertices[':1'][p_2_tag], data_vertices[':2'][p_2_tag]],
        ])

edge_coordinates = np.array(edge_coordinates)


'''
start = [
    [p_edge_0_start_x, p_edge_0_start_y, p_edge_0_start_z],
    [p_edge_1_start_x, p_edge_1_start_y, p_edge_1_start_z],
    ....
]
end = [
    [p_edge_0_end_x, p_edge_0_end_y, p_edge_0_end_z],
    [p_edge_1_end_x, p_edge_1_end_y, p_edge_1_end_z],
    ....
]
'''

start = []
end = []
for tri in edge_coordinates:        # tet is (4,3)
    start.append(tri[0])
    end.append(tri[1])

start = np.array(start)                  # (6 N, 3)
end   = np.array(end)

'''
edge_data_frame contains all the edges obtained from `edge_coordinates` and it has the structure

    p_start_edge_0_x,p_start_edge_0_y,p_start_edge_0_z,p_end_edge_0_x,p_end_edge_0_y,p_end_edge_0_z,
    p_start_edge_1_x,p_start_edge_1_y,p_start_edge_1_z,p_end_edge_1_x,p_end_edge_1_y,p_end_edge_1_z,
    ...
'''
edge_data_frame = pd.DataFrame({
    clab.label_start_x_column: start[:, 0],
    clab.label_start_y_column: start[:, 1],
    clab.label_end_x_column:   end[:, 0],
    clab.label_end_y_column:   end[:, 1]
})



'''
collect the x and y coordinates of the start and end mesh nodes of the edges in `edge_rows` and store them in `mesh_nodes_x_coord`, `mesh_nodes_y_coord`

mesh_nodes_x_coord = [
    x_coord_start_vertex_of_plotted_edge_0, 
    x_coord_start_vertex_of_plotted_edge_1,
    ..., 
    x_coord_end_vertex_of_plotted_edge_0, 
    x_coord_end_vertex_of_plotted_edge_1,
    ..., 
]

mesh_nodes_y_coord = [
    y_coord_start_vertex_of_plotted_edge_0, 
    y_coord_start_vertex_of_plotted_edge_1,
    ..., 
    y_coord_end_vertex_of_plotted_edge_0, 
    y_coord_end_vertex_of_plotted_edge_1,
    ..., 
]
'''
mesh_nodes_x_coord = pd.concat([edge_data_frame["start:0"], edge_data_frame["end:0"]])
mesh_nodes_y_coord = pd.concat([edge_data_frame["start:1"], edge_data_frame["end:1"]])

# list of centroids of the edges
centroids = edge_coordinates.mean(axis=1)          # (N, 3)
# live of values of `f` computed on each centroid
grid_f_values = data_u['f']
# norm used for the colorbar, which sets the color to be assigned to each face
norm_f_values = plt.Normalize(vmin=grid_f_values.min(), vmax=grid_f_values.max())

fig = pplt.figure(figsize=np.array(parameters['figure_size']),
                  left=parameters['figure_margin'][0][0],
                  right=parameters['figure_margin'][0][1],
                  bottom=parameters['figure_margin'][1][1],
                  top=parameters['figure_margin'][1][0],
                  wspace=parameters['wspace'],
                  hspace=parameters['hspace'])



# create axes
fig.add_subplot(1, 1, 1)

colorbar_axis = fig.add_axes([parameters['colorbar_position'][0],
                                    parameters['colorbar_position'][1],
                                    parameters['colorbar_size'][0],
                                    parameters['colorbar_size'][1]])

'''
plot the a list of DOFs
'''
def plot_dof(list, ax):

    dof_x_coord = data_u[':0'][list]
    dof_y_coord = data_u[':1'][list]

    ax.scatter(dof_x_coord, dof_y_coord,
                color=parameters['dof_color'], 
                s=parameters['dof_size'],
                clip_on=False,
                zorder=1)

all_dofs = [i for i in range(len(data_u))]
u_dofs_to_plot = []
colorbar = None



def plot_snapshot(n, fig):

    global u_dofs_to_plot, colorbar

    # =============
    # mesh plot
    # =============

    ax = fig.axes[0]  # Use the existing axis
    ax.set_box_aspect(1)
    ax.set_axis_off()

    if colorbar is None: 

        colorbar, _ = gr.cb.make_colorbar(fig, grid_f_values, np.min(grid_f_values), np.max(grid_f_values),
                    position=parameters['colorbar_position'], 
                    size=parameters['colorbar_size'],
                    label_pad=parameters['colorbar_axis_label_offset'],
                    label=parameters['colorbar_axis_label'],
                    font_size=parameters['colorbar_font_size'],
                    tick_label_offset=parameters['colorbar_tick_label_offset'],
                    tick_length=parameters['colorbar_tick_length'],
                    line_width=parameters['colorbar_line_width'],
                    axis=colorbar_axis
        )
            
    # plot the mesh 
    gr.plot_2d_mesh(ax, edge_data_frame, parameters['mesh_line_width'], 'black', parameters['alpha_mesh'], 
                zorder=0)

    '''
    plot the mesh nodes
    '''
    ax.scatter(mesh_nodes_x_coord, mesh_nodes_y_coord,
                color=parameters['mesh_node_color'], 
                s=parameters['mesh_node_size'],
                clip_on=False,
                zorder=0)

    '''
    plot DOFs
    ''' 
    plot_dof(all_dofs, ax)


    '''
    plot u DOFs
    '''

    dof_u_x_coord = data_u[':0'][u_dofs_to_plot]
    dof_u_y_coord = data_u['f'][u_dofs_to_plot]

    # build colors in order to color the points cooredponding to u DOF
    colors = data_u['f'][u_dofs_to_plot]

    cbar_vmin, cbar_vmax = colorbar.mappable.get_clim()
    # plot the points corresponding to u DOF
    ax.scatter(dof_u_x_coord, dof_u_y_coord,
                c=colors, 
                cmap=gr.cb.color_map_type,
                vmin=cbar_vmin,
                vmax=cbar_vmax,                
                s=parameters['f_dof_size'],
                clip_on=False,
                zorder=2)


    '''
    plot the lines between the x axis and u DOF
    '''

    start_p = list(zip(data_u[':0'][u_dofs_to_plot], data_u['f'][u_dofs_to_plot]))
    end_p = list(zip(data_u[':0'][u_dofs_to_plot], data_u[':1'][u_dofs_to_plot]))

    start_end_segments = np.stack([start_p, end_p], axis=1) 

    line_collection = LineCollection(start_end_segments,
                        linewidths=parameters['f_dof_line_width'],
                        color=parameters['f_dof_color'],
                        clip_on=False,
                        zorder=1
            )
    ax.add_collection(line_collection)

    if(n < len(data_u)):
        u_dofs_to_plot.append(n)

    
    gr.plot_2d_axes(ax, [0, 0], [mesh_parameters['L'], h],
                    tick_length=parameters['axis_tick_length'],
                    line_width=parameters['axis_line_width'],
                    axis_label=parameters['axis_label'],
                    # tick_label_format=['f', 'f'],
                    font_size=parameters['axis_font_size'],
                    tick_label_offset=parameters['axis_tick_label_offset'],
                    axis_label_offset=parameters['axis_label_offset'],
                    axis_origin=parameters['axis_origin'],
                    # plot_label=parameters["v_plot_panel_label"],
                    # plot_label_offset=parameters['panel_label_position'],
                    # plot_label_font_size=parameters['panel_label_font_size'],
                    n_minor_ticks=parameters['axis_n_minor_ticks'],
                    minor_tick_length=parameters['axis_minor_tick_length'],
                    tick_label_angle=parameters['axis_tick_label_angle'],
                    axis_label_angle=parameters['axis_label_angle'],
                    colorbar_axis=colorbar_axis,
                    colorbar_axis_offset=parameters['colorbar_position']
                    )
    
   
   



    
plot_snapshot(parameters['number_of_frames'], fig)
plt.savefig(figure_path + "_large.pdf")
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf'
)
