import gc
import matplotlib.animation as ani
import os
import numpy as np
import pandas as pd
import system.utils as sys_utils
import time

import calculus.utils as cal
import graphics.vector_plot as vp
import input_output.utils as io
import list.utils as lis
import plot
import text.utils as text


animation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animation_' +  plot.parameters['figure_name'] + '.mp4')
print(f'snapshot_path = {plot.snapshot_path}')
number_of_frames = sys_utils.count_v_files('line_mesh_n_', plot.snapshot_path)



axis_min_max_abs = [[np.inf,-np.inf],[np.inf,-np.inf]]

# run through all snapshots
for n_snapshot in range(plot.snapshot_min, plot.snapshot_max, plot.parameters['frame_stride']):

    data_u_msh = pd.read_csv(os.path.join(plot.snapshot_nodal_values_path, 'u_n_' + str(n_snapshot) + '.csv'), usecols=plot.columns_v)

    X_ref, Y_ref, u_n_X, u_n_Y, _, _, _, _ = vp.interpolate_2d_vector_field(data_u_msh,
                                                                            [0, 0],
                                                                            [plot.parameters['L'], plot.parameters['h']],
                                                                            plot.parameters['n_bins_v'])
    
    #X, Y are the positions of the mesh nodes in the current configuration    
    X = np.array(lis.add_lists_of_lists(X_ref, u_n_X))
    Y = np.array(lis.add_lists_of_lists(Y_ref, u_n_Y))

    # compute the min-max of the snapshot
    X_min_max = [lis.min_max(X),lis.min_max(Y)]
    
    # update the absolute min and max according to the min-max of the snapshot 
    for i in range(2):
        if X_min_max[i][0] < axis_min_max_abs[i][0]:
            axis_min_max_abs[i][0] = X_min_max[i][0]
            
        if X_min_max[i][1] > axis_min_max_abs[i][1]:
            axis_min_max_abs[i][1] = X_min_max[i][1]
# 
norm_v_min_max_abs = cal.norm_min_max_files('def_v_fl_n_', plot.snapshot_path, plot.snapshot_min, plot.snapshot_max, plot.parameters['frame_stride'])

 



animation_duration_in_sec = (number_of_frames / plot.parameters['frame_stride']) / plot.parameters['frames_per_second']

print(
    f"number of frames: {number_of_frames} \n frames per second: {plot.parameters['frames_per_second']} \n animation duration : {animation_duration_in_sec} [s]\n frame stride = {plot.parameters['frame_stride']}\n number of frames to draw ~ {int(plot.number_of_frames/plot.parameters['frame_stride'])} \n snapshot_min/max: {[plot.snapshot_min, plot.snapshot_max]}",
    flush=True)

Writer = ani.writers['ffmpeg']
writer = Writer(fps=plot.parameters['frames_per_second'], metadata=dict(artist='Michele'), bitrate=(int)(plot.parameters['bit_rate']))

text.empty_texts(plot.fig)


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

    text.clear_labels_with_patterns(plot.fig, ["\second", "\msecond", "\minute", "\hour", "\met"])

    plot.plot_snapshot(plot.fig, n, 
                    snapshot_label=rf'$t = \,$' + io.time_to_string(n * plot.parameters['T'] / plot.number_of_frames, 's', plot.parameters['n_decimals_snapshot_label']),
                    axis_min_max=axis_min_max_abs,
                    norm_v_min_max=norm_v_min_max_abs     
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