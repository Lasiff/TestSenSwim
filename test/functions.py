from collections import OrderedDict
from math import sqrt
import cv2
import os
import pandas as pd
from os.path import join


def list_of_frames(df, frames_stride, frames_to_show, column):
    """It produces frames numbers from the summary file"""
    frames = []
    lanes = []
    for index, row in df.iterrows():
        if row[column]:
            frames += [int(x[:-1]) for x in row[column].split('|')]
            lanes += [str(x[-1]) for x in row[column].split('|')]
    lane_dict = dict(zip(frames, lanes))
    frames = set(frames)
    list_frames = []
    dict_lanes = dict()
    for elem in frames:
        list_frames += [elem + frames_stride * x for x in range(frames_to_show)]
        dict_lanes.update({elem + frames_stride * x: lane_dict[elem] for x in range(frames_to_show)})
    return sorted(list_frames), dict_lanes


def extract_frames(path_in, path_out, fp_frames, fn_frames, fn_frame_dict=None, fp_frame_dict=None):
    """The function takes video as input and produces images of all FP and FN events"""

    cap = cv2.VideoCapture(path_in)
    count = 0
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            if count in fp_frames:
                cv2.imwrite(os.path.join(path_out, "frame{:d}_{}_FP.jpg".format(count, fp_frame_dict[count])), frame)
            if count in fn_frames:
                cv2.imwrite(os.path.join(path_out, "frame{:d}_{}_FN.jpg".format(count, fn_frame_dict[count])), frame)
            count += 1
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def to_data_frame(items):
    labels = ['tag', 'start', 'end', 'high_prob_frame', 'high_prob', 'ml_start', 'ml_end', 'ml_tag',
              'event_start_frame', 'window_size', 'distance_between_events_frames', 'event']
    df = pd.DataFrame.from_records(items, columns=labels)
    df = df.sort_values('event_start_frame')
    df.drop('event_start_frame', axis=1, inplace=True)
    return df


def est_params(df):
    """
    This function takes all the results from comparing tagged events (or events from ml system) and estimates all
    the metrics of the system accuracy
    To produce bias STD it firstly calculates difference between tagged event start.
    Secondly, it adds this result to high probability frame number and evaluates STD from it.
    """
    gap = 0
    avg_gap = 0
    gap_new = 0
    counter = 0
    fn_events = []
    fp_events = []
    for index, row in df.iterrows():
        if row['event'] == 'TP':
            gap += (int(row['high_prob_frame']) - int(row['start'])) ** 2
            avg_gap += int(row['high_prob_frame']) - int(row['start'])
            counter += 1
        elif row['event'] == 'FP':
            fp_events.append(str(row['high_prob_frame']) + str(row['ml_tag']))
        elif row['event'] == 'FN':
            fn_events.append(str(row['start']) + str(row['tag'][0]))
    for index, row in df.iterrows():
        if row['event'] == 'TP':
            gap_new += (int(row['high_prob_frame'] - avg_gap / counter) - int(row['start'])) ** 2
    std = sqrt(gap / counter)
    bias_std = sqrt(gap_new / counter)
    fn = len(fn_events)
    fp = len(fp_events)
    fn_events = '|'.join(fn_events)
    fp_events = '|'.join(fp_events)
    return df['event'].size, counter, fn, fp, fn_events, fp_events, avg_gap / counter, std, bias_std


def results_to_df(results):
    """Output of this function is data frame. Input is the results from comparing tagged events and ml probabilities"""
    labels = ['th', 'window_size', 'min_window_size', 'holes', 'moving_average_size', 'number_of_events', 'TP', 'FN',
              'FP', 'fn_frames', 'fp_frames', 'avg_gap', 'STD', 'bias_STD', 'total_frames']
    df_res = pd.DataFrame.from_records(results, columns=labels)
    return df_res


def results_to_df_events(results):
    """Output of this function is data frame. Input is the results from comparing ml events and ml probabilities"""
    labels = ['th', 'window_size', 'min_window_size', 'holes', 'moving_average_size',
              'number_of_events', 'TP', 'FN', 'FP', 'fn_frames', 'fp_frames', 'avg_gap', 'STD', 'bias_STD']
    df_res = pd.DataFrame.from_records(results, columns=labels)
    return df_res


def total_vid_frames(vid_path):
    """The function produces total amount of frames it the video"""
    cap = cv2.VideoCapture(vid_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def summary(df):
    """It makes the summary file from all sets of the parameters it one file"""
    counter = 0
    avg_gap = 0
    gap_mean = 0
    bias_gap_mean = 0
    for index, row in df.iterrows():
        try:
            gap = int(row['gap'])
        except ValueError:
            continue
        gap_mean += gap ** 2
        avg_gap += gap
        counter += 1
    avg_gap //= counter
    for index, row in df.iterrows():
        try:
            gap_bias = int(row['gap']) - avg_gap
        except ValueError:
            continue
        bias_gap_mean = gap_bias ** 2
    std = sqrt(gap_mean / counter)
    bias_std = sqrt(bias_gap_mean / counter)
    names = ['avg_gap', 'std', 'bias_std']
    items = [avg_gap, std, bias_std]
    out_df = pd.DataFrame(items, names)
    return out_df
