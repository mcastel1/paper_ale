'''
this animation plots 
    - in a first part, the mesh, by adding subsequently mesh triangles over time
    - in a second part, a function `f` on the mesh surface, by coloring subsequent triangles of the mesh in terms of a color code corresponding to `f`
'''


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import os
import pandas as pd
import proplot as pplt
import shutil
import warnings


import list.column_labels as clab
import input_output.utils as io
import graphics.utils as gr
import graphics.animation as ani
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
# mesh_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/generate_mesh/2d/square', "solution")/

solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution/")
# solution_path = os.path.join('/Users/michelecastellana/Documents/finite_elements/poisson_equation/solve_u', "solution")

figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), parameters['figure_name'])


# labels of columns to read
data_vertices = pd.read_csv(os.path.join(mesh_path, "vertices.csv")).set_index('tag')


# load the data on the triangles of the mesh: each triangle is a list of three vertex labels, labelled as in `data_vertices`: these vertices delimit the triangle
data_triangles = pd.read_csv(os.path.join(mesh_path, "triangles.csv"))

# load data for the field f
data_u = pd.read_csv(os.path.join(solution_path, "u.csv"))

'''
chose `number_of_frames_mesh` equal to a custom value (every time more than one triangle is added to the animation, so `number_of_frames_mesh` needs to be chosen manually rather than set equal to len(data_triangles))
chose `number_of_frames_vertices` equal to the number of vertices because at every step one vertex is added at a time, and similarly for `number_of_frames_dofs` and `number_of_frames_colored_triangles`

'''
number_of_frames_mesh = parameters['number_of_frames_mesh']
number_of_frames_vertices = len(data_vertices)
number_of_frames_dofs = len(data_u)
number_of_frames_colored_triangles = len(data_triangles)

number_of_frames = number_of_frames_mesh + number_of_frames_vertices + number_of_frames_dofs + number_of_frames_colored_triangles


min_max = [[min(data_vertices[":0"]),max(data_vertices[":0"])],[min(data_vertices[":1"]),max(data_vertices[":1"])],[min(data_vertices[":2"]),max(data_vertices[":2"])]]

print(f'L = {[min_max[0][1] - min_max[0][0], min_max[1][1]-min_max[1][0], min_max[2][1]-min_max[2][0]]}')

# function to be plotted on the 3d surface with color code
def f(x, y, z):
    return 1 + x**2 + 2*y**2

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


# list of centroids of the triangles
centroids = triangle_coordinates.mean(axis=1)          # (N, 3)
# live of values of `f` computed on each centroid
grid_f_values = f(centroids[:, 0], centroids[:, 1], centroids[:, 2])  # (N,)
# norm used for the colorbar, which sets the color to be assigned to each face
norm_f_values = plt.Normalize(vmin=grid_f_values.min(), vmax=grid_f_values.max())
# the color map
# cmap = plt.get_cmap('viridis')

# 
vertex_coordinates_x = data_vertices[':0'].to_numpy()
vertex_coordinates_y = data_vertices[':1'].to_numpy()
vertex_f_values = f(
    data_vertices[':0'].to_numpy(),
    data_vertices[':1'].to_numpy(),
    data_vertices[':2'].to_numpy()
)


triangles = np.array([
    [p1-1, p2-1, p3-1]
    for p1, p2, p3 in data_triangles[['p_1', 'p_2', 'p_3']].itertuples(index=False)
])
# 

# the 3 edges of a triangle, as index pairs into the 3 vertices
edge_pairs = [(0,1),(0,2),(1,2)]

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
for tri in triangle_coordinates:        # tet is (4,3)
    for a, b in edge_pairs:
        start.append(tri[a])
        end.append(tri[b])

start = np.array(start)                  # (6 N, 3)
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
fig.add_subplot(1, 1, 1)

'''
plot the a list of vertices
'''
def plot_vertices(list, ax):

    vertex_x_coordinates = data_vertices[':0'][list]
    vertex_y_coordinates = data_vertices[':1'][list]

    ax.scatter(vertex_x_coordinates, vertex_y_coordinates,
                color=parameters['vertex_color'], 
                s=parameters['vertex_size'],
                clip_on=False,
                zorder=0)

'''
plot the a list of DOFs
'''
def plot_dof(list, ax):

    dof_x_coordinates = data_u[':0'][list]
    dof_y_coordinates = data_u[':1'][list]

    color = gr.cb.color_map_type(norm_f_values(f(dof_x_coordinates, dof_y_coordinates, np.zeros(len(dof_x_coordinates)))))
    color = np.asarray(color).reshape(-1, 4)

    if len(color) == 1:
        color = color[0]

    ax.scatter(dof_x_coordinates, dof_y_coordinates,
                color=color, 
                s=parameters['dof_size'],
                edgecolors='black',
                clip_on=False,
                zorder=1)

colorbar_axis = fig.add_axes([parameters['colorbar_position'][0],
                                    parameters['colorbar_position'][1],
                                    parameters['colorbar_size'][0],
                                    parameters['colorbar_size'][1]])

all_triangles = [i for i in range(len(data_triangles))]

# triangles_to_plot=[i for i in range(len(data_triangles))]
mesh_triangles_to_plot = []
colored_triangles_to_plot = []

colorbar = None
dofs_to_plot = []
vertices_to_plot = []



def plot_snapshot(n, fig):

    global mesh_triangles_to_plot, colored_triangles_to_plot, colorbar


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



    edge_rows = [3 * t + k for t in mesh_triangles_to_plot for k in range(3)]

    #1. plot the mesh 
    gr.plot_2d_mesh(ax, edge_data_frame.iloc[edge_rows], parameters['mesh_line_width'], 'black', parameters['alpha_mesh'], 
                 zorder=1)

    
    #2. plot the contour plot of `u` in `colored_triangles_to_plot`
    if len(colored_triangles_to_plot) > 0:

        # build a list of the triangles in which the contour plot of `f` will be made
        triangles_to_color = triangles[colored_triangles_to_plot]

        # build a triangulation based on `triangles_to_color`
        triangulation_colored = mtri.Triangulation(
            vertex_coordinates_x,
            vertex_coordinates_y,
            triangles_to_color
        )


        # make a contour plot of `f` within the region delimited by triangulation_colored
        matplotlib.axes.Axes.tripcolor(
            ax,
            triangulation_colored,
            vertex_f_values,
            cmap=gr.cb.color_map_type,
            vmin=grid_f_values.min(),
            vmax=grid_f_values.max(),
            shading='gouraud',
            alpha=parameters['alpha_u'],
            zorder=0
        )



    #3. plot DOFs
    plot_vertices(vertices_to_plot, ax)

    #4. plot DOFs
    plot_dof(dofs_to_plot, ax)


    gr.plot_2d_axes(ax, [0, 0], [mesh_parameters['L'], mesh_parameters['h']],
                    tick_length=parameters['axis_tick_length'],
                    line_width=parameters['axis_line_width'],
                    axis_label=parameters['axis_label'],
                    # tick_label_format=['f', 'f'],
                    font_size=parameters['axis_font_size'],
                    tick_label_offset=parameters['axis_tick_label_offset'],
                    axis_label_offset=parameters['axis_label_offset'],
                    axis_origin=parameters['axis_origin'],
                    n_minor_ticks=parameters['axis_n_minor_ticks'],
                    minor_tick_length=parameters['axis_minor_tick_length'],
                    tick_label_angle=parameters['axis_tick_label_angle'],
                    axis_label_angle=parameters['axis_label_angle'],
                    colorbar_axis=colorbar_axis,
                    colorbar_axis_offset=parameters['colorbar_position']
                    )

    # update `mesh_triangles to plot`
    if n < number_of_frames_mesh: 

        ani.add_element(mesh_triangles_to_plot, data_triangles)

    # update `colored_triangles_to_plot`
    if (n >= number_of_frames_mesh) and (n < number_of_frames_mesh + number_of_frames_vertices):

        m = n - number_of_frames_mesh + 1
        if (m not in vertices_to_plot) and (m < len(data_vertices)):
                vertices_to_plot.append(m)


    elif (n >= number_of_frames_mesh + number_of_frames_vertices) and (n < number_of_frames_mesh + number_of_frames_vertices + number_of_frames_dofs):

        m = n - (number_of_frames_mesh + number_of_frames_vertices) + 1
        
        if (m not in dofs_to_plot) and (m < len(data_u)):
            dofs_to_plot.append(m)

    elif (n >= number_of_frames_mesh + number_of_frames_vertices + number_of_frames_dofs):

        ani.add_element(colored_triangles_to_plot, data_triangles)











        

plot_snapshot(number_of_frames_vertices, fig)
plt.savefig(figure_path + "_large.pdf")
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf'
)
