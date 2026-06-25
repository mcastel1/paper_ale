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
data_line_vertices = pd.read_csv(os.path.join(mesh_path, "line_vertices.csv"))


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

edges_to_plot=[0]





def plot_snapshot(fig, azimuth_altitude):

    data_line_vertices_to_plot = data_line_vertices.iloc[edges_to_plot[-1]]
    data_line_vertices_start = data_line_vertices[["start:0", "start:1", "start:2"]]
    data_line_vertices_end = data_line_vertices[["end:0", "end:1", "end:2"]]

    '''
    start and end vertex of the last edge in `edges_to_plot`
    
    '''
    start_vertex = data_line_vertices_to_plot[["start:0", "start:1", "start:2"]].values
    end_vertex = data_line_vertices_to_plot[["end:0", "end:1", "end:2"]].values

    '''
    find other edges that have `start_vertex` of `end_vertex` as either start or end point: match[i] = True if the i-th edge contains either of these, and False otherwise
    '''
    match = ( 
        data_line_vertices_start.eq(start_vertex).all(axis=1) 
        | data_line_vertices_start.eq(end_vertex).all(axis=1)
        | data_line_vertices_end.eq(start_vertex).all(axis=1)
        | data_line_vertices_end.eq(end_vertex).all(axis=1)
        )

    # don't reuse rows already in the path
    match.iloc[edges_to_plot] = False

    if match.any():
        # match containts at least one True -> find the first True entry in match -> this will be the next edge to add 
        next_edge = match.idxmax()
    else:
        # match contains no Trues -> the search algorithm is stuch -> look for a new "connected component" by picking a new edge not in `edges_to_plot`
        remaining_edges = [i for i in range(len(data_line_vertices)) if i not in edges_to_plot]
        next_edge = remaining_edges[0] if remaining_edges else None

    if next_edge != None:
        edges_to_plot.append(next_edge)

    # print(f'vertices_to_plot = {edges_to_plot}')

    # =============
    # mesh plot
    # =============

    ax = fig.axes[0]  # Use the existing axis

    ax.set_box_aspect([1] * 3)
    gr.empty_panes(ax)
    ax.set_axis_off()
    ax.view_init(elev=azimuth_altitude[1], azim=azimuth_altitude[0])

    gr.plot_mesh(ax, data_line_vertices.iloc[edges_to_plot],
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
        



# plot_snapshot(fig, parameters['azimuth'], parameters['altitude'])

plt.savefig(figure_path + "_large.pdf")
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf'
)
