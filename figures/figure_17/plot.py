import matplotlib
import matplotlib.pyplot as plt
import more_itertools 
import numpy as np
import os
import pandas as pd
import proplot as pplt
import warnings

import list.column_labels as clab
import input_output.utils as io
import graphics.utils as gr
import system.paths as paths


matplotlib.use(
    "Agg"
)  # use a non-interactive backend so that figures do not pop up during run in debug mode

# Suppress the specific warning
warnings.filterwarnings(
    "ignore", message=".*Z contains NaN values.*", category=UserWarning)
# clean the matplotlib cache to load the correct version of definitions.tex
os.system(" rm -rf ~/.matplotlib/tex.cache")

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

# mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/solution/")
mesh_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/generate_mesh/3d/shapes/tomcat', "solution/")
# mesh_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/generate_mesh/3d/ball', "solution/")

figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), parameters['figure_name'])


# labels of columns to read
data_vertices = pd.read_csv(os.path.join(mesh_path, "vertices.csv")).set_index('tag')
data_triangles = pd.read_csv(os.path.join(mesh_path, "triangles.csv"))

min_max = [[min(data_vertices[":0"]),max(data_vertices[":0"])],[min(data_vertices[":1"]),max(data_vertices[":1"])],[min(data_vertices[":2"]),max(data_vertices[":2"])]]

'''
triangle_coordinates = [
                        [[p_1_triangle_0_x, p_1_triangle_0_y, p_1_triangle_0_z], ..., [p_3_triangle_0_x, p_3_triangle_0_y, p_3_triangle_0_z]],
                        [[p_1_triangle_1_x, p_1_triangle_1_y, p_1_triangle_1_z], ..., [p_3_triangle_1_x, p_3_triangle_1_y, p_3_triangle_1_z]],
                        ...
                        ]


'''
triangle_coordinates = []
for i in range(len(data_triangles)):
    
    p_1_tag = data_triangles['p_1'][i]
    p_2_tag = data_triangles['p_2'][i]
    p_3_tag = data_triangles['p_3'][i]

    triangle_coordinates.append([
        [data_vertices[':0'][p_1_tag], data_vertices[':1'][p_1_tag], data_vertices[':2'][p_1_tag]], 
        [data_vertices[':0'][p_2_tag], data_vertices[':1'][p_2_tag], data_vertices[':2'][p_2_tag]],
        [data_vertices[':0'][p_3_tag], data_vertices[':1'][p_3_tag], data_vertices[':2'][p_3_tag]],
        ])

triangle_coordinates = np.array(triangle_coordinates)


# the 6 edges of a tet, as index pairs into the 4 vertices
edge_pairs = [(0,1),(0,2),(1,2)]

start = []
end = []
for tri in triangle_coordinates:        # tet is (4,3)
    for a, b in edge_pairs:
        start.append(tri[a])
        end.append(tri[b])

start = np.array(start)                  # (6N, 3)
end   = np.array(end)

'''
edge_data_frame contains all the edges obtained from `triangle_coordinates` and it has the structure

    p_start_edge_0_x,p_start_edge_0_y,p_start_edge_0_z,p_end_edge_0_x,p_end_edge_0_y,p_end_edge_0_z,
    p_start_edge_1_x,p_start_edge_1_y,p_start_edge_1_z,p_end_edge_1_x,p_end_edge_1_y,p_end_edge_1_z,
    ...
'''
edge_data_frame = pd.DataFrame({
    clab.label_start_x_column: start[:, 0],
    clab.label_start_y_column: start[:, 1],
    clab.label_start_z_column: start[:, 2],
    clab.label_end_x_column:   end[:, 0],
    clab.label_end_y_column:   end[:, 1],
    clab.label_end_z_column:   end[:, 2],
})


fig = pplt.figure(figsize=np.array(parameters['figure_size']),
                  left=parameters['figure_margin'][0][0],
                  right=parameters['figure_margin'][0][1],
                  bottom=parameters['figure_margin'][1][1],
                  top=parameters['figure_margin'][1][0],
                  wspace=parameters['wspace'],
                  hspace=parameters['hspace'])

# create axes
# 3d axes
fig.add_subplot(1, 1, 1, projection="3d", auto_add_to_figure=False)

# triangles_to_plot=[i for i in range(len(data_triangles))]
triangles_to_plot = []

def plot_snapshot(fig, azimuth_altitude):

    global triangles_to_plot


    # =============
    # mesh plot
    # =============

    ax = fig.axes[0]  # Use the existing axis

    ax.set_box_aspect([min_max[0][1] - min_max[0][0], min_max[1][1]-min_max[1][0], min_max[2][1]-min_max[2][0]])
    gr.empty_panes(ax)
    ax.set_axis_off()
    ax.view_init(elev=azimuth_altitude[1], azim=azimuth_altitude[0])

    # construct the list of rows to pick into `edge_data_frame` by converting `triangles_to_plot` into the format of `edge_data_frame` (fill in 6 consecutive entries in edge_data_frame and select blocks of 6 consecutive entries according to `triangles_to_plot`)
    edge_rows = [3 * t + k for t in triangles_to_plot for k in range(3)]

    gr.plot_mesh(ax, edge_data_frame.iloc[edge_rows],
                parameters['mesh_line_width'], 'black', parameters['alpha_mesh'])


    gr.plot_3d_axes(ax, 
                    [min_max[0][0], min_max[1][0], min_max[2][0]], 
                    [min_max[0][1] - min_max[0][0], min_max[1][1]-min_max[1][0], min_max[2][1]-min_max[2][0]],
                    axis_origin=parameters['axis_origin_3d'],
                    axis_label=parameters['axis_label_3d'],
                    axis_label_offset=parameters['axis_label_offset_3d'],
                    tick_label_offset=parameters['tick_label_offset_3d'],
                    tick_label_format=parameters['tick_label_format_3d'],
                    tick_length=parameters['tick_length_3d'],
                    minor_tick_length=parameters['minor_tick_length_3d'],
                    n_minor_ticks=parameters['n_minor_ticks_3d'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width_3d'],
                    plot_label_position=parameters['plot_label_offset_3d'],
                    plot_label_font_size=parameters['plot_label_font_size'])
    
    # gr.set_axes_limits(ax, 
    #                 [min_max[0][0], min_max[1][0], min_max[2][0]], 
    #                 [min_max[0][1], min_max[1][1], min_max[2][1]])


    if len(triangles_to_plot) > 1:

        match = pd.Series(False, index=data_triangles.index)

        for i in range(len(triangles_to_plot)):

            plotted_triangle = data_triangles.iloc[triangles_to_plot[i]]

            '''
            tags of the vertices in the last triangle in `triangles_to_plot`
            '''
            p_1_vertex = plotted_triangle[['p_1']].values[0]
            p_2_vertex = plotted_triangle[['p_2']].values[0]
            p_3_vertex = plotted_triangle[['p_3']].values[0]

            '''
            find other tetrahedra that have `p_1_vertex` or ... `p_4_vertex` equal to either `p_1` or `p_2` or `p_3` or `p_4`: match[i] = True if the i-th tetrahedron contains either of these, and False otherwise
            '''
            match = match | ( 

                data_triangles['p_1'].eq(p_1_vertex) 
                | data_triangles['p_1'].eq(p_2_vertex)
                | data_triangles['p_1'].eq(p_3_vertex)

                | data_triangles['p_2'].eq(p_1_vertex)
                | data_triangles['p_2'].eq(p_2_vertex)
                | data_triangles['p_2'].eq(p_3_vertex)

                | data_triangles['p_3'].eq(p_1_vertex)
                | data_triangles['p_3'].eq(p_2_vertex)
                | data_triangles['p_3'].eq(p_3_vertex)

                )

        # don't reuse rows already in the path
        match.iloc[triangles_to_plot] = False

    else: 

        match = np.bool_(False)


    if match.any():

        next_triangle = []
        for i in range(len(match)):
            if match[i]:
                next_triangle.append(i)

    else:
        # match contains no Trues -> the search algorithm is stuch -> look for a new "connected component" by picking a new tetrahedron not in `tetrahedra_to_plot`
        remaining_triangles = [i for i in range(len(data_triangles)) if i not in triangles_to_plot]
        next_triangle = remaining_triangles[-1] if remaining_triangles else None

    if next_triangle != None:
        triangles_to_plot.append(next_triangle)
        # flatten `triangles_to_plot`
        triangles_to_plot = list(more_itertools.collapse(triangles_to_plot))




    # print(f'vertices_to_plot = {edges_to_plot}')

        



plot_snapshot(fig, [120, 45])
plt.savefig(figure_path + "_large.pdf")
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf'
)
