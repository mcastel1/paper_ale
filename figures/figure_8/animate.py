import matplotlib.animation as ani
import numpy as np
import os
import time

import plot
import text.utils as text

animation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animation_' + plot.parameters['figure_name'] + '.mp4')


# altitude of the plot as a function of the step 'n' of the animation
def snapshot_altitude(n):
    # return plot.parameters['altitude_min'] + (plot.parameters['altitude_max'] - plot.parameters['altitude_min']) * np.sin(2* np.pi * n / (plot.parameters['number_of_frames'] - 1))
    return plot.parameters['altitude_min'] + (plot.parameters['altitude_max'] - plot.parameters['altitude_min']) * (1 - np.cos(2 * np.pi * n / plot.parameters['number_of_frames'])) / 2


# azimuth of the plot as a functiomn of the step 'n' of the animaiton
def snapshot_azimuth(n):
    return plot.parameters['azimuth_min'] + (plot.parameters['azimuth_max'] - plot.parameters['azimuth_min']) * n / plot.parameters['number_of_frames']



print(f'number of frames: {plot.parameters["number_of_frames"]}, '
      f'frames per second: {plot.parameters["frames_per_second"]}, '
      f'animation duration = : {plot.parameters["number_of_frames"] / plot.parameters["frames_per_second"]}', flush=True)


Writer = ani.writers['ffmpeg']
writer = Writer( fps=plot.parameters['frames_per_second'], metadata=dict( artist='Michele' ), bitrate=(int)( plot.parameters['bit_rate'] ) )


def update_animation(n):

    print("Calling update_animation with n = ", n, " ... ", flush=True)
    start_time = time.time()


    # clear only the major axes of the plot. The colorbar axes need not be cleaned because make_colorbar already clears them
    for ax in plot.fig.axes[:5]:
        ax.clear()
        
    # Clear text objects (the snapshot label accumulates)
    for txt in plot.fig.texts[:]:
        txt.remove()
    # plot.gr.delete_all_axes(plot.fig)
    
    text.clear_labels_with_patterns(plot.fig, ["\met", "\msecond", "\minute", "\hour", "a. u."])
    
    plot.plot_snapshot(plot.fig, altitude=snapshot_altitude(n), azimuth=snapshot_azimuth(n))

    # Stop timer
    end_time = time.time()
    print(f"... done in {end_time - start_time:.2f} s", flush=True)

 
print( "... done" )

animation = ani.FuncAnimation(
    fig=plot.fig, \
    func=update_animation, \
    frames=range(0, plot.parameters['number_of_frames'], plot.parameters['frame_stride']),    # interval between subsequent frames in milliseconds\
    interval=30 \
    )

animation.save(animation_path, dpi=plot.parameters['dpi'], writer=writer)
