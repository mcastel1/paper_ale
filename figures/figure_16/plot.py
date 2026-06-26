import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import proplot as pplt
import warnings

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
mesh_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/generate_mesh/3d/ball/', "solution/")

figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), parameters['figure_name'])


# labels of columns to read
data_vertices = pd.read_csv(os.path.join(mesh_path, "vertices.csv")).set_index('tag')
data_edges = pd.read_csv(os.path.join(mesh_path, "edges.csv"))

edge_coordinates = []
for i in range(len(data_edges)):
    
    start_tag = data_edges['p_1'][i]
    end_tag = data_edges['p_2'][i]

    edge_coordinates.append([[data_vertices[':0'][start_tag], data_vertices[':1'][start_tag], data_vertices[':2'][start_tag]], 
                             [data_vertices[':0'][end_tag], data_vertices[':1'][end_tag], data_vertices[':2'][end_tag]]])

edge_coordinates = np.array(edge_coordinates)


edge_data_frame = pd.DataFrame({
    'p_1:0': edge_coordinates[:, 0, 0],
    'p_1:1': edge_coordinates[:, 0, 1],
    'p_1:2': edge_coordinates[:, 0, 2],
    'p_2:0':   edge_coordinates[:, 1, 0],
    'p_2:1':   edge_coordinates[:, 1, 1],
    'p_2:2':   edge_coordinates[:, 1, 2],
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

edges_to_plot=[]


def plot_snapshot(fig, azimuth_altitude):


    # =============
    # mesh plot
    # =============

    ax = fig.axes[0]  # Use the existing axis

    ax.set_box_aspect([1] * 3)
    gr.empty_panes(ax)
    ax.set_axis_off()
    ax.view_init(elev=azimuth_altitude[1], azim=azimuth_altitude[0])


    gr.plot_mesh(ax, edge_data_frame.iloc[edges_to_plot],
                parameters['mesh_line_width'], 'black', parameters['alpha_mesh'])


    gr.plot_3d_axes(ax, [-mesh_parameters['r'], -mesh_parameters['r'], -mesh_parameters['r']], [2*mesh_parameters['r'], 2*mesh_parameters['r'], 2*mesh_parameters['r']],
                    scale_factor=[1, 1, parameters['scale_factor_z']],
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
    
    gr.set_axes_limits(ax, [-mesh_parameters['r'], -mesh_parameters['r'], -mesh_parameters['r']], [mesh_parameters['r'], mesh_parameters['r'], mesh_parameters['r']])

    if len(edges_to_plot) > 1:

        last_plotted_edge = data_edges.iloc[edges_to_plot[-1]]


        '''
        tags of the vertices in the last edge in `edges_to_plot`
        '''
        p_1_vertex = last_plotted_edge[['p_1']].values[0]
        p_2_vertex = last_plotted_edge[['p_2']].values[0]

        '''
        find other edges that have `p_1_vertex` of `p_2_vertex` as either start or end point: match[i] = True if the i-th edge contains either of these, and False otherwise
        '''
        match = ( 
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
        # match containts at least one True -> find the first True entry in match -> this will be the next edge to add 
        next_edge = match.idxmax()
    else:
        # match contains no Trues -> the search algorithm is stuch -> look for a new "connected component" by picking a new edge not in `edges_to_plot`
        remaining_edges = [i for i in range(len(data_edges)) if i not in edges_to_plot]
        next_edge = remaining_edges[0] if remaining_edges else None

    if next_edge != None:
        edges_to_plot.append(next_edge)


        



# plot_snapshot(fig, [0, 45])

plt.savefig(figure_path + "_large.pdf")
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf'
)
