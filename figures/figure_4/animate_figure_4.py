import matplotlib.animation as ani
import os 
import time

import plot_figure_4 as pfig
import system.system_io as sysio
import text.text as text

number_of_frames = sysio.count_v_files('X_n_12_', pfig.snapshot_path)
animation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animation_figure_4.mp4')



# the first frame may have z == 0 for all bins, which creates problems when plotted (division by zero), thus you may want to start with a frame > 1

animation_duration_in_sec = (number_of_frames / pfig.parameters['frame_stride']) / pfig.parameters['frames_per_second']

print(
    f"number of frames: {number_of_frames} \n frames per second: {pfig.parameters['frames_per_second']} \n animation duration : {animation_duration_in_sec} [s]\n frame stride = {pfig.parameters['frame_stride']}",
    flush=True)

Writer = ani.writers['ffmpeg']
writer = Writer(fps=pfig.parameters['frames_per_second'], metadata=dict(artist='Michele'), bitrate=(int)(pfig.parameters['bit_rate']))

text.empty_texts(pfig.fig)


def update_animation(n):
    print("Calling update_animation with n = ", n, " ... ", flush=True)
    start_time = time.time()

    for ax in pfig.fig.axes:
        ax.clear()
    # graph.remove_all_axes(pfig.fig)
    for ax in pfig.fig.axes[:]:
        if ax.get_label() == "colorbar":
            pfig.fig.delaxes(ax)

    text.clear_labels_with_patterns(pfig.fig, ["\met", "\msecond", "\minute", "\hour", "a. u."])
    pfig.plot_column(pfig.fig, n)

    # Stop timer
    end_time = time.time()
    print(f"... done in {end_time - start_time:.2f} s", flush=True)


animation = ani.FuncAnimation(
    fig=pfig.fig,
    func=update_animation,
    frames=range(pfig.parameters['n_first_frame'], number_of_frames, pfig.parameters['frame_stride']),
    interval=30
)

animation.save(animation_path, dpi=pfig.parameters['dpi'], writer=writer)
