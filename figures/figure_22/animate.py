import gc
import matplotlib.animation as ani
import os
import time

import calculus.utils as cal
import input_output.utils as io
import plot
import text.utils as text


animation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animation_' +  plot.parameters['figure_name'] + '.mp4')
print(f'snapshot_path = {plot.snapshot_path}')


# compute absolute minima and maxima across snapshots



axis_min_max_abs = cal.X_curr_min_max_abs(
                                            plot.snapshot_min, 
                                            plot.snapshot_max, 
                                            plot.parameters['frame_stride'],
                                            plot.snapshot_nodal_values_path,
                                            [[0, plot.parameters['L']], [0, plot.parameters['h']]],
                                            plot.parameters['n_bins_v_fl']
                                        )




animation_duration_in_sec = (plot.number_of_frames / plot.parameters['frame_stride']) / plot.parameters['frames_per_second']

print(
    f"number of frames: {plot.number_of_frames} \n frames per second: {plot.parameters['frames_per_second']} \n animation duration : {animation_duration_in_sec} [s]\n frame stride = {plot.parameters['frame_stride']}\n number of frames to draw ~ {int(plot.number_of_frames/plot.parameters['frame_stride'])} \n snapshot_min/max: {[plot.snapshot_min, plot.snapshot_max]}",
    flush=True)

Writer = ani.writers['ffmpeg']
writer = Writer(fps=plot.parameters['frames_per_second'], metadata=dict(artist='Michele'), bitrate=(int)(plot.parameters['bit_rate']))

text.empty_texts(plot.fig)


def update_animation(n):
    print("Calling update_animation with n = ", n, " ... ", flush=True)
    start_time = time.time()

    # clear only the major axes of the plot. The colorbar axes need not be cleaned because make_colorbar already clears them
    # Clear ProPlot's internal legend registry first
    for ax in plot.fig.axes:
        if hasattr(ax, '_legend_dict'):
            ax._legend_dict.clear()
        if hasattr(ax, '_colorbar_dict'):
            ax._colorbar_dict.clear()
        ax.clear()
        
    # Clear text objects (the snapshot label accumulates)
    for txt in plot.fig.texts[:]:
        txt.remove()
    # plot.gr.delete_all_axes(plot.fig)

    text.clear_labels_with_patterns(plot.fig, ["\second", "\msecond", "\minute", "\hour", "\met"])

    plot.plot_snapshot(plot.fig, n, 
                    snapshot_label=rf'$t = \,$' + io.time_to_string(n * plot.solution_parameters['T'] / plot.solution_parameters['N'], 'min_s', plot.parameters['n_decimals_snapshot_label']),
                    axis_min_max=axis_min_max_abs
                    )

    # garbace collection to avoid memory leaks
    gc.collect()


    # Stop timer
    end_time = time.time()
    print(f"... done in {end_time - start_time:.2f} s", flush=True)


animation = ani.FuncAnimation(
    fig=plot.fig,
    func=update_animation,
    frames=range(plot.snapshot_min, plot.snapshot_max, plot.parameters['frame_stride']),
    interval=30
)

animation.save(animation_path, dpi=plot.parameters['dpi'], writer=writer)