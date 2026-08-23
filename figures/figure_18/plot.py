'''
this animation plots 
    - in a first part, the mesh, by adding subsequently mesh edges over time
    - in a second part, a function `f` on the mesh surface, by coloring subsequent edges of the mesh in terms of a color code corresponding to `f`
'''


import matplotlib
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import more_itertools 
import numpy as np
import os
import pandas as pd
import proplot as pplt
from scipy.interpolate import lagrange as lag
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

solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
# solution_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/poisson_equation/solve_u', "solution")

figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parameters['figure_name'])

# labels of columns to read
data_vertices = pd.read_csv(os.path.join(mesh_path, "vertices.csv")).set_index('tag')

# load the data on the edges of the mesh: each edge is a list of 2 vertex labels, labelled as in `data_vertices`: these vertices delimit the edge
data_edges = pd.read_csv(os.path.join(mesh_path, "edges.csv"))

# load data for the field f
data_u = pd.read_csv(os.path.join(solution_path, "u.csv"))


min_max = [[min(data_vertices[":0"]),max(data_vertices[":0"])],[min(data_vertices[":1"]),max(data_vertices[":1"])],[min(data_vertices[":2"]),max(data_vertices[":2"])]]
h = np.max(data_u['f'])

print(f'L = {[min_max[0][1] - min_max[0][0], min_max[1][1]-min_max[1][0], min_max[2][1]-min_max[2][0]]}')

# function to be plotted on the 3d surface with color code
def f(x, y, z):
    return np.sqrt(x**2) * np.cos(2.0*np.pi*x/(min_max[0][1] - min_max[0][0])) 

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


fig = pplt.figure(figsize=np.array(parameters['figure_size']),
                  left=parameters['figure_margin'][0][0],
                  right=parameters['figure_margin'][0][1],
                  bottom=parameters['figure_margin'][1][1],
                  top=parameters['figure_margin'][1][0],
                  wspace=parameters['wspace'],
                  hspace=parameters['hspace'])



# create axes
fig.add_subplot(1, 1, 1)



all_edges = [i for i in range(len(data_edges))]
all_edge_rows = [t for t in range(len(data_edges))]

# edges_to_plot=[i for i in range(len(data_edges))]
edges_to_plot = []





def plot_snapshot(n, fig, azimuth_altitude):

    global edges_to_plot


    # =============
    # mesh plot
    # =============

    ax = fig.axes[0]  # Use the existing axis
    ax.set_box_aspect(1)
    ax.set_axis_off()



    if (n == 0) or (n == parameters['number_of_frames']):
        # `plot_snapshot` has been called with n = 0 (first call) or n = parameters['number_of_frames'] (beginnning of color draw) -> set `edges_to_plot` to an empty list
        edges_to_plot = []
        edge_rows = []


    if n < parameters['number_of_frames']:

        # construct the list of rows to pick into `edge_data_frame` by converting `edges_to_plot` into the format of `edge_data_frame` (fill in 6 consecutive entries in edge_data_frame and select blocks of 6 consecutive entries according to `edges_to_plot`)
        edge_rows = [t for t in edges_to_plot]

    else: 

        if n <  parameters['number_of_frames']:

            # n > parameters['number_of_frames']: I want to plot the full mesh -> same as the case above, with `edges_to_plot` replaced by `all_edges`
            edge_rows = all_edge_rows

            plot_dof(ax)
            


    # plot the mesh 
    gr.plot_2d_mesh(ax, edge_data_frame.iloc[edge_rows], 
                    line_width=parameters['mesh_line_width'], 
                    color='black', 
                    alpha=parameters['alpha_mesh'], 
                    zorder=0)


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
    mesh_nodes_x_coord = pd.concat([edge_data_frame.iloc[edge_rows]["start:0"], edge_data_frame.iloc[edge_rows]["end:0"]])
    mesh_nodes_y_coord = pd.concat([edge_data_frame.iloc[edge_rows]["start:1"], edge_data_frame.iloc[edge_rows]["end:1"]])

    '''
    plot the mesh nodes
    '''
    ax.scatter(mesh_nodes_x_coord, mesh_nodes_y_coord,
                color=parameters['mesh_node_color'], 
                s=parameters['mesh_node_size'],
                clip_on=False,
                zorder=0)




    
    gr.plot_2d_axes(ax, [0, 0], [(mesh_parameters['x_r']-mesh_parameters['x_l']), h],
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
                    )
    
   
    # update `edges_to_plot`
    if len(edges_to_plot) > 1:

        match = pd.Series(False, index=data_edges.index)

        for i in range(len(edges_to_plot)):

            plotted_edge = data_edges.iloc[edges_to_plot[i]]

            '''
            tags of the vertices in the last edge in `edges_to_plot`
            '''
            p_1_vertex = plotted_edge[['p_1']].values[0]
            p_2_vertex = plotted_edge[['p_2']].values[0]

            '''
            find other tetrahedra that have `p_1_vertex` or `p_2_vertex` equal to either `p_1` or `p_2`: match[i] = True if the i-th edge contains either of these, and False otherwise
            '''
            match = match | ( 

                data_edges['p_1'].eq(p_1_vertex) 
                | data_edges['p_1'].eq(p_2_vertex)

                | data_edges['p_2'].eq(p_1_vertex)
                | data_edges['p_2'].eq(p_2_vertex)

                )

        # don't reuse rows already in the path
        match.iloc[edges_to_plot] = False

    else: 

        match = np.bool_(False)


    if match.any():

        next_edge = []
        for i in range(len(match)):
            if match[i]:
                next_edge.append(i)

    else:
        # match contains no Trues -> the search algorithm is stuch -> look for a new "connected component" by picking a new edge not in `edges_to_plot`

        remaining_edges = [i for i in range(len(data_edges)) if i not in edges_to_plot]
        next_edge = remaining_edges[-1] if remaining_edges else None

    if next_edge != None:

        edges_to_plot.append(next_edge)
        # flatten `edges_to_plot`
        edges_to_plot = list(more_itertools.collapse(edges_to_plot))

        pass



    
plot_snapshot(parameters['number_of_frames'], fig, [120, 45])
plt.savefig(figure_path + "_large.pdf")
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf'
)
