'''
this animation plots 
    - in a first part, the mesh, by adding subsequently mesh edges over time
    - in a second part, a function `f` on the mesh surface, by coloring subsequent edges of the mesh in terms of a color code corresponding to `f`
'''


import matplotlib
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


min_max = [[min(data_vertices[":0"]),max(data_vertices[":0"])],[min(data_vertices[":1"]),max(data_vertices[":1"])],[min(data_vertices[":2"]),max(data_vertices[":2"])]]

print(f'L = {[min_max[0][1] - min_max[0][0], min_max[1][1]-min_max[1][0], min_max[2][1]-min_max[2][0]]}')

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


fig = pplt.figure(figsize=np.array(parameters['figure_size']),
                  left=parameters['figure_margin'][0][0],
                  right=parameters['figure_margin'][0][1],
                  bottom=parameters['figure_margin'][1][1],
                  top=parameters['figure_margin'][1][0],
                  wspace=parameters['wspace'],
                  hspace=parameters['hspace'])



# create axes
fig.add_subplot(1, 1, 1)

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

dofs_to_plot = []

def plot_snapshot(n, fig):


    # =============
    # mesh plot
    # =============

    ax = fig.axes[0]  # Use the existing axis
    ax.set_box_aspect(1)
    ax.set_axis_off()
            
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
    plot_dof(dofs_to_plot, ax)

    if(n < len(data_u)):
        dofs_to_plot.append(n)

    gr.plot_2d_axes(ax, [0, 0], [mesh_parameters['x_r']-mesh_parameters['x_l'], h],
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
                    axis_label_angle=parameters['axis_label_angle']
                    )
    
   
   



    
plot_snapshot(parameters['number_of_frames'], fig)
plt.savefig(figure_path + "_large.pdf")
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf'
)
