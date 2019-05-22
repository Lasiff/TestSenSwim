from tagged_video import *
from functions import *
import sys, os

parameters_file = 'params'
# following block is using to set up path for python interpreter to the folder that contains this script
# It also reads all the parameters from parameters file and writes them to the dictionary called params
path = sys.argv[0][:-11]
try:
    os.chdir(path)
    params = ReadingParameters(parameters_file).params
except FileNotFoundError:
    os.chdir(repr(path))
    params = ReadingParameters(parameters_file).params

results_file = []
# read parameters from parameters dict
vid_dir = join(params['input_files_dir'])
video_name = params['vid_name_0']
vid_path = join(vid_dir, video_name)
lines = ['Left', 'Center', 'Right']

ml_output_file = join(vid_dir, '{}_probabilities.csv'.format(video_name[:-4]))

# checks type of the system is being used right now and reads file with tagged events or ml events
if params['results_from_events']:
    events_path = join(vid_dir, video_name[:-4] + '_events.csv')
    tags = MlEventsFile(events_path)
    out_dir = params['event_result_files_folder_name']
else:
    tag_path = join(vid_dir, video_name + '.json')
    tags = TaggedEvents(tag_path, vid_path)
    amount_of_frames = total_vid_frames(vid_path)
    out_dir = params['prob_result_files_folder_name']

# takes every possible set of parameters and estimates csv file for each of them
# it sums up every particular set to the list to the results_file variable
for th in params['th']:
    th = round(th, 2)
    for window_size in params['window_size']:
        for holes in params['holes']:
            for moving_average_size in params['moving_average_size']:
                for min_window_size in params['min_window_size']:
                    ml_file = MlEvents(ml_output_file, th, holes, window_size, min_window_size,
                                       moving_average_size, lines, params['mean_type'])
                    results = CompareEvents(tags.events, ml_file.events)
                    df = to_data_frame(results.composition_of_events)
                    if params['results_from_events']:
                        results_file.append([th, window_size, min_window_size, holes, moving_average_size,
                                             *est_params(df)])
                    else:
                        results_file.append([th, window_size, min_window_size, holes, moving_average_size,
                                             *est_params(df), amount_of_frames])
                    save_dir = join(params['results_folder_name'], out_dir)
                    file_name = 'th={}window_size={}_holes={}_moving_average_size={}_min_window_size{}.csv'.format(th,
                                                          window_size, holes, moving_average_size, min_window_size)
                    name_of_file = join(save_dir, file_name)
                    df.to_csv(name_of_file, index=False)

# it makes final csv file composed of every particular set of parameters
name_res_file = video_name + '_results.csv'
dir_res_file = params['result_csv_folder_name']
res_dir_name = join(params['results_folder_name'], dir_res_file, name_res_file)
if params['results_from_events']:
    df_res = results_to_df_events(results_file)
else:
    # if system uses a video it produces frames for FP and FN events
    df_res = results_to_df(results_file)
    fn_frames, fn_frame_dict = list_of_frames(df_res, params['frames_stride'], params['frames_to_show'], 'fn_frames')
    fp_frames, fp_frame_dict = list_of_frames(df_res, params['frames_stride'], params['frames_to_show'], 'fp_frames')
    extract_frames(vid_path, join(params['results_folder_name'], params['result_images_folder_name']), fp_frames,
                   fn_frames, fn_frame_dict, fp_frame_dict)
df_res.to_csv(res_dir_name)



