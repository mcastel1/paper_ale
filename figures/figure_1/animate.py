import matplotlib.animation as ani
import os 
import time

import calculus.utils as cal
import text.utils as text
import plot
import input_output.utils as io

parameters = io.read_parameters_from_csv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))

animation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animation_figure_1.mp4')

'''
compute absolute mins and maxs across multiple files
'''
norm_v_min_max_abs = cal.min_max_vector_field(
                                            plot.snapshot_min, plot.snapshot_max, parameters['frame_stride'], 
                                            os.path.join(plot.solution_path + 'snapshots/csv/nodal_values'), 
                                            'def_v_n_', 
                                            parameters['n_bins_v'],
                                            [[0, 0],[parameters['L'], parameters['h']]]
                                        )   

sigma_min_max_abs = cal.min_max_files(
                'def_sigma_n_12_', 
                os.path.join(plot.solution_path + 'snapshots/csv/nodal_values'),
                plot.snapshot_min + parameters['colorbar_sigma_snapshot_min_offset'], 
                plot.snapshot_max, 
                parameters['frame_stride']
                 )


# the first frame may have z == 0 for all bins, which creates problems when plotted (division by zero), thus you may want to start with a frame > 1

animation_duration_in_sec = (plot.number_of_frames / parameters['frame_stride']) / parameters['frames_per_second']

print(
    f"number of frames: {plot.number_of_frames} \n frames per second: {plot.parameters['frames_per_second']} \n animation duration : {animation_duration_in_sec} [s]\n frame stride = {plot.parameters['frame_stride']}\n number of frames to draw ~ {int(plot.number_of_frames/plot.parameters['frame_stride'])}", 
    flush=True)

Writer = ani.writers['ffmpeg']
writer = Writer(fps=parameters['frames_per_second'], metadata=dict(artist='Michele'), bitrate=(int)(parameters['bit_rate']))

text.empty_texts(plot.fig)


def update_animation(n):
    print("Calling update_animation with n = ", n, " ... ", flush=True)
    start_time = time.time()

    # clear only the major axes of the plot. The colorbar axes need not be cleaned because make_colorbar already clears them
    for ax in plot.fig.axes[:2]:
        ax.clear()
        
    # Clear text objects (the snapshot label accumulates)
    for txt in plot.fig.texts[:]:
        txt.remove()
    # plot.gr.delete_all_axes(plot.fig)
    
    text.clear_labels_with_patterns(plot.fig, ["\second", "\msecond", "\minute", "\hour", "\pas"])

    plot.plot_snapshot(
                        plot.fig, 
                        n, 
                        snapshot_label=rf'$t = \,$' + io.time_to_string(n * parameters['T'] / plot.number_of_frames, 's', parameters['n_decimals_snapshot_label']),
                        norm_v_min_max=norm_v_min_max_abs
                        # sigma_min_max=sigma_min_max_abs
                     )

    # Stop timer
    end_time = time.time()
    print(f"... done in {end_time - start_time:.2f} s", flush=True)


animation = ani.FuncAnimation(
    fig=plot.fig,
    func=update_animation,
    frames=range(parameters['n_first_frame'], plot.number_of_frames, parameters['frame_stride']),
    interval=30
)

animation.save(animation_path, dpi=parameters['dpi'], writer=writer)
