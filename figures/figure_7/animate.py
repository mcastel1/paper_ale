import matplotlib.animation as ani
import os
import system.utils as sys_utils
import time

import input_output.utils as io
import text.utils as text
import plot


animation_duration_in_sec = (
    plot.number_of_frames / plot.parameters['frame_stride']) / plot.parameters['frames_per_second']
animation_path = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'animation_' + plot.parameters['figure_name'] + '.mp4')

print(
    f"number of frames: {plot.number_of_frames} \n frames per second: {plot.parameters['frames_per_second']} \n animation duration : {animation_duration_in_sec} [s]\n frame stride = {plot.parameters['frame_stride']}\n number of frames to draw ~ {int(plot.number_of_frames/plot.parameters['frame_stride'])}",
    flush=True)


Writer = ani.writers['ffmpeg']
writer = Writer(fps=plot.parameters['frames_per_second'], metadata=dict(
    artist='Michele'), bitrate=(int)(plot.parameters['bit_rate']))

text.empty_texts(plot.fig)


def update_animation(n):
    print("Calling update_animation with n = ", n, " ... ", flush=True)
    start_time = time.time()

    # clear only the major axes of the plot. The colorbar axes need not be cleaned because make_colorbar already clears them
    for ax in plot.fig.axes[:4]:
        ax.clear()

    # Clear text objects (the snapshot label accumulates)
    for txt in plot.fig.texts[:]:
        txt.remove()
    # plot.gr.delete_all_axes(plot.fig)

    text.clear_labels_with_patterns(
        plot.fig, ["\second", "\msecond", "\minute", "\hour", "\pas"])

    plot.plot_snapshot(plot.fig, n, rf'$t = \,$' + io.time_to_string(n *
                       plot.parameters['T'] / plot.number_of_frames, 's', 2))

    # Stop timer
    end_time = time.time()
    print(f"... done in {end_time - start_time:.2f} s", flush=True)


animation = ani.FuncAnimation(
    fig=plot.fig,
    func=update_animation,
    frames=range(plot.snapshot_min, plot.snapshot_max,
                 plot.parameters['frame_stride']),
    interval=30
)


animation.save(animation_path, dpi=plot.parameters['dpi'], writer=writer)
