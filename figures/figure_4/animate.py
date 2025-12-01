import matplotlib.animation as ani
import os
import time

import calculus.utils as cal
import input_output.utils as io
import plot
import text.utils as text

animation_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'animation_figure_4.mp4')

'''
this is used to compute the absolute mins and maxs not on the entire range [snapshot_min, snapshot_max] but on part of it, because on the first few frames some fields may be large and thus give unreasonable absolute mins and max
'''
snapshot_min_with_margin = plot.snapshot_min + \
    plot.parameters['snapshot_min_margin']

# compute min and max of fields across all snapshots
X_min_max_abs = [
    cal.min_max_files('X_n_12_', plot.snapshot_path, snapshot_min_with_margin,
                      plot.snapshot_max, plot.parameters['frame_stride'], field_column_name='f:0'),
    cal.min_max_files('X_n_12_', plot.snapshot_path, snapshot_min_with_margin,
                      plot.snapshot_max, plot.parameters['frame_stride'], field_column_name='f:1')
]
norm_v_min_max_abs = cal.norm_min_max_files(
    'v_n_', plot.snapshot_path, snapshot_min_with_margin, plot.snapshot_max, plot.parameters['frame_stride'])
w_min_max_abs = cal.min_max_files(
    'w_n_', plot.snapshot_path, snapshot_min_with_margin, plot.snapshot_max, plot.parameters['frame_stride'])
# sigma_min_max_abs = cal.min_max_files('sigma_n_12_', plot.snapshot_path, snapshot_min_with_margin, plot.snapshot_max, plot.parameters['frame_stride'])


# the first frame may have z == 0 for all bins, which creates problems when plotted (division by zero), thus you may want to start with a frame > 1

animation_duration_in_sec = (
    plot.number_of_frames / plot.parameters['frame_stride']) / plot.parameters['frames_per_second']

print(
    f"number of frames: {plot.number_of_frames} \n frames per second: {plot.parameters['frames_per_second']} \n animation duration : {animation_duration_in_sec} [s]\n frame stride = {plot.parameters['frame_stride']}\n number of frames to draw ~ {int(plot.number_of_frames/plot.parameters['frame_stride'])} \n snapshot_min/max: {[plot.snapshot_min, plot.snapshot_max]}",
    flush=True)

Writer = ani.writers['ffmpeg']
writer = Writer(fps=plot.parameters['frames_per_second'], metadata=dict(
    artist='Michele'), bitrate=(int)(plot.parameters['bit_rate']))

text.empty_texts(plot.fig)


def update_animation(n):
    print("Calling update_animation with n = ", n, " ... ", flush=True)
    start_time = time.time()

    # clear only the major axes of the plot. The colorbar axes need not be cleaned because make_colorbar already clears them
    for ax in plot.fig.axes[:7]:
        ax.clear()

    # Clear text objects (the snapshot label accumulates)
    for txt in plot.fig.texts[:]:
        txt.remove()

    text.clear_labels_with_patterns(
        plot.fig, ["\met", "\msecond", "\minute", "\hour", "a. u."])

    plot.plot_snapshot(plot.fig, n,
                       X_min_max=X_min_max_abs,
                       norm_v_min_max=norm_v_min_max_abs,
                       w_min_max=w_min_max_abs,
                       # sigma_min_max=sigma_min_max_abs,
                       snapshot_label=rf'$t = \,$' +
                       io.time_to_string(
                           n * plot.parameters['T'] / plot.number_of_frames, 'min_s', 1)
                       )

    # Stop timer
    end_time = time.time()
    print(f"... done in {end_time - start_time:.2f} s", flush=True)


animation = ani.FuncAnimation(
    fig=plot.fig,
    func=update_animation,
    frames=range(snapshot_min_with_margin, plot.snapshot_max,
                 plot.parameters['frame_stride']),
    interval=30
)

animation.save(animation_path, dpi=plot.parameters['dpi'], writer=writer)
