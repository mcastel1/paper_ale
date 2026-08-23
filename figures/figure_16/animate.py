import matplotlib.animation as ani
import os
import time

import graphics.utils as gr
import text.utils as text
import plot


animation_duration_in_sec = (
    len(plot.data_triangles) / plot.parameters['frame_stride']) / plot.parameters['frames_per_second']
animation_path = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'animation_' + plot.parameters['figure_name'] + '.mp4')


print(
    f"number of frames: {len(plot.data_triangles) - len(plot.mesh_triangles_to_plot)} \n frames per second: {plot.parameters['frames_per_second']} \n animation duration : {animation_duration_in_sec} [s]\n frame stride = {plot.parameters['frame_stride']}\n number of frames to draw ~ {int(plot.number_of_frames/plot.parameters['frame_stride'])}",
    flush=True)


Writer = ani.writers['ffmpeg']
writer = Writer(fps=plot.parameters['frames_per_second'], metadata=dict(
    artist='Michele'), bitrate=(int)(plot.parameters['bit_rate']))

def init_animation():
    plot.dofs_to_plot = []
    plot.colorbar = None
    plot.mesh_triangles_to_plot = []
    plot.colored_triangles_to_plot = []

def update_animation(n):
    print("Calling update_animation with n = ", n, " ... ", flush=True)
    start_time = time.time()

    # clear only the major axes of the plot. The colorbar axes need not be cleaned because make_colorbar already clears them
    for ax in plot.fig.axes[:1]:
        ax.clear()

    # Clear text objects (the snapshot label accumulates)
    for txt in plot.fig.texts[:]:
        txt.remove()
    # plot.gr.delete_all_axes(plot.fig)

    text.clear_labels_with_patterns(
        plot.fig, ["\second", "\msecond", "\minute", "\hour", "\pas"])

    plot.plot_snapshot(n, plot.fig, gr.azimuth_altitude(n, plot.number_of_frames, 
                                                     [plot.parameters['azimuth_min'], plot.parameters['azimuth_max']],
                                                     [plot.parameters['altitude_min'], plot.parameters['altitude_max']]))

    # plot.plot_snapshot(plot.fig,    [plot.parameters['azimuth_min'], plot.parameters['altitude_min']])

    # Stop timer
    end_time = time.time()
    print(f"... done in {end_time - start_time:.2f} s", flush=True)




plot.mesh_triangles_to_plot = []     

animation = ani.FuncAnimation(
    fig=plot.fig,
    init_func=init_animation,
    func=update_animation,
    frames=range(0, plot.number_of_frames), 
    interval=30
)

animation.save(animation_path, dpi=plot.parameters['dpi'], writer=writer)
