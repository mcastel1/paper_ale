import matplotlib.animation as ani
import os
import system.utils as sys_utils
import time

import input_output as io
import text
import plot as pfig


print(f'snapshot_path = {pfig.snapshot_path}')
number_of_frames = sys_utils.count_v_files('line_mesh_n_', pfig.snapshot_path)

# the first frame may have z == 0 for all bins, which creates problems when plotted (division by zero), thus you may want to start with a frame > 1
# n_first_frame = 1
frame_stride = 20000
frame_stride = 20

# frames_per_second = 30
frames_per_second = 1
animation_duration_in_sec = (number_of_frames / frame_stride) / frames_per_second

print(
    f"number of frames: {number_of_frames} \n frames per second: {frames_per_second} \n animation duration : {animation_duration_in_sec} [s]\n frame stride = {frame_stride}",
    flush=True)

bit_rate = 300000
dpi = 300

Writer = ani.writers['ffmpeg']
writer = Writer(fps=frames_per_second, metadata=dict(artist='Michele'), bitrate=(int)(bit_rate))

text.empty_texts(pfig.fig)


def update_animation(n):
    print("Calling update_animation with n = ", n, " ... ", flush=True)
    start_time = time.time()

    for ax in pfig.fig.axes:
        ax.clear()
    pfig.gr.delete_all_axes(pfig.fig)

    text.clear_labels_with_patterns(pfig.fig, ["\second", "\msecond", "\minute", "\hour"])

    pfig.plot_snapshot(pfig.fig, n, rf'$t = \,$' + io.time_to_string(n * pfig.T / number_of_frames, 's', 2))

    # Stop timer
    end_time = time.time()
    print(f"... done in {end_time - start_time:.2f} s", flush=True)


animation = ani.FuncAnimation(
    fig=pfig.fig,
    func=update_animation,
    frames=range(n_first_frame, number_of_frames, frame_stride),
    interval=30
)

animation.save('animation_large_' + pfig.figure_name + '.mp4', dpi=dpi, writer=writer)
