import json
from imutils.video import count_frames
from collections import defaultdict
import pandas as pd
import numpy as np
from functions import *
from distutils.util import strtobool
import os
from os.path import join


class TaggedEvents(object):
    """
    input of the class is a video and a json file.
    Json file consist of label, start frame and end frame for each event.
    The class parsing a json file and makes a default dictionary of it.
    Each key is a lane label, value is a list of every event ot this lane.
    Every event is a list with 3 elements [max_prob_frame, start_frame, end_frame]

    """
    def __init__(self, tag_path, vid_path):
        self.tag_path = tag_path
        self.vid_path = vid_path
        with open(self.tag_path) as f:
            vid_tags = json.load(f)
        self._load_tags(vid_tags)
        self._group_events()

    def _group_events(self, ):
        """
        Takes a list of sets for every frame of the video.

        :return: Dictionary with keys as lane labels and values as [max_prob_frame, start_frame, end_frame]
        """
        events = defaultdict(list)
        for frame, frame_tag in enumerate(self.vid_tags):
            for tag in frame_tag:
                if len(events[tag]) > 0 and events[tag][-1][-1] + 1 == frame:
                    events[tag][-1][-1] += 1
                else:
                    events[tag].append([frame, frame, frame])
        self.events = events

    def _load_tags(self, vid_tags):
        """
        At first it reads a video file and calculates amount of frames in it.
        Secondly, it iterates through every frame number and creates empty set for each one.
        If it founds event at this frame it fills up set with tag label and frame number
        :param vid_tags: json file with tags
        :return: List of sets for each frame of the video, empty or with event information.
        """
        try:
            self.frame_count = count_frames(self.vid_path)
        except FileNotFoundError:
            print('Wrong path to the video')
        vid_frame_tags = []

        for frame in range(self.frame_count):
            vid_frame_tags.append(set())
            for region in vid_tags['frames'].get(str(frame), [{'tags': []}]):
                region_tags = set(region['tags'])
                if len(region_tags) == 1:
                    if len(region_tags & vid_frame_tags[-1]) > 0:
                        raise ValueError('Found duplicate tag in frame %s' % frame)
                    else:
                        for k in region_tags:
                            vid_frame_tags[-1].add(k)
                elif len(region_tags) > 1:
                    raise ValueError('Expected exactly 1 tag per region. Found %s in frame %s' %
                                     (len(region['tags']), frame))
        self.vid_tags = vid_frame_tags
        self.indices_with_tags = [i for i, t in enumerate(self.vid_tags) if len(t) > 0]


class MlEvents(object):
    """
    This class takes a particular set of parameters, file with ml probabilities and makes events from it.
    logic:
        Initializing all class parameters, reading file with raw probabilities csv file as dataframe
        Smoothing raw probabilities with one of the functions,



    """
    def __init__(self, ml_output_file, th, holes, window_size, min_window_size, moving_average_size, lines, mean_type):
        self.th = th
        self.holes = holes
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.moving_average_size = int(moving_average_size)
        self.mean_type = mean_type
        self.ml_output_file = ml_output_file
        self.lines = lines
        self.df = pd.read_csv(self.ml_output_file)
        self._smooth_probabilities()
        self._events()

    def _mean_filter_same(self, series, line):
        """
        :param series: array of probabilities for a single line
        :param line: name of the lane
        Takes self.moving_average_size as amount of frames that would be smoothed
        for every frame returns half of frames probabilities before particular frame, half after,
        the frame probability itself and divides it by self.moving_average_size
        :return: series with probabilities that have been smoothed
        """
        self.df[line] = np.convolve(series, np.ones((self.moving_average_size,)) / self.moving_average_size,
                                    mode='same')

    def _mean_filter_full(self, series, line):
        """
        :param series: array of probabilities for a single line
        :param line: name of the lane
        Takes self.moving_average_size as amount of frames that would be smoothed
        :return: series with probabilities that have been smoothed
        it sums up (self.moving_average_size - 1) elements after the particular frame and frame itself and divides it by
        self.moving_average_size for every frame
        at the start of the series it divides (self.moving_average_size - 1)
        frames to make series the same length with the input frame
        """
        self.df[line] = np.convolve(series, np.ones((self.moving_average_size,
                                                     )) / self.moving_average_size)[self.moving_average_size - 1:]

    def _mean_filter_valid(self, series, line):
        """
        :param series: array of probabilities for a single line
        :param line: name of the lane
        Takes self.moving_average_size as amount of frames that would be smoothed
        for every frame returns summed up probabilities of
        :return: series with probabilities that have been smoothed
        it sums up (self.moving_average_size - 1) elements before the particular frame and frame itself and divides it by
        self.moving_average_size for every frame
        at the start of the series it adds few zeros to make series the same length with the input frame
        """
        additional_elements = [0 for x in range(self.moving_average_size - 1)]
        self.df[line] = np.append(additional_elements,
                                  np.convolve(series, np.ones((self.moving_average_size,
                                                               )) / self.moving_average_size, mode='valid'))

    def _smooth_probabilities(self, ):
        """
        It uses one of three smoothing functions to smooth probabilities
        :return: df that have been smoothed
        """
        for line in self.lines:
            series = self.df[line]
            if self.mean_type == 'same':
                self._mean_filter_same(series, line)
            elif self.mean_type == 'full':
                self._mean_filter_full(series, line)
            elif self.mean_type == 'valid':
                self._mean_filter_valid(series, line)
            else:
                raise TypeError('Wrong type of the mean')

    def _events(self):
        """
        creates events from probabilities
        :return: default dictionary consist of events
        """
        self.events = defaultdict(list)
        for line in self.lines:
            self._make_events(line)

    def _make_events(self, line):
        """
        :param line: takes name of the lane

        continues_frames counts maximun amount of frames above the th in a row and if it would be at least equal to min_window_size
        than event would be counted
        logic:
        Function iterates through every row of the df, obtains probabilities of particular lane and frame numbers.
        If probability of current frame rise above the th it would raise event_flag.
        If the flag is raised it would add to the event following frames until amount of holes wouldn't be equal to
        holes parameter or counter_frames wouldn't be equal to window_size parameter.
        If counter_frames parameter would be equal to window_size it would skip every frame above the th after it
        and finish the event on the first frame under the th.
        :return:
        """
        continues_frames = 0
        event_length_flag = False
        holes_counter = 0
        event_flag = False
        slicing_counter = 0
        for frame, row in self.df.iterrows():
            if slicing_counter == self.window_size:
                if row[line] > self.th:
                    continue
                else:
                    slicing_counter = 0
                    holes_counter = 0
                    event_flag = False
                    continues_frames = 0
                    if not event_length_flag:
                        del self.events[line[0].lower()][-1]
                    event_length_flag = False
            elif row[line] > self.th and event_flag:
                if self.events[line[0].lower()][-1][-1] + 1 == frame:
                    self.events[line[0].lower()][-1][-1] = frame
                    slicing_counter += 1
                    continues_frames += 1
                    if continues_frames == self.min_window_size:
                        event_length_flag = True
                    if row[line] > self.events[line[0].lower()][-1][1]:
                        self.events[line[0].lower()][-1][0] = frame
                        self.events[line[0].lower()][-1][1] = row[line]
            elif row[line] > self.th and not event_flag:
                self.events[line[0].lower()].append([frame, row[line], frame, frame])
                event_flag = True
                slicing_counter += 1
                continues_frames += 1
            elif row[line] < self.th and event_flag:
                continues_frames = 0
                if holes_counter < self.holes:
                    holes_counter += 1
                    self.events[line[0].lower()][-1][-1] = frame
                else:
                    # self.events[line[0].lower()][-1][-1] -= int(self.holes)
                    holes_counter = 0
                    event_flag = False
                    slicing_counter = 0
                    if not event_length_flag:
                        del self.events[line[0].lower()][-1]
                    event_length_flag = False
                    continues_frames = 0


class MlEventsFile(object):
    """
    Reads a file consist of events produced by ml system and creates a dictionary similar to dictionary from tagged file
    """
    def __init__(self, events_path):
        self.df = pd.read_csv(events_path)
        self._make_events()

    def _make_events(self, ):
        self.events = defaultdict(list)
        for index, row in self.df.iterrows():
            self.events[row['lane'][0]].append([row['max_prob_frame_num'],
                                                row['start_frame'],
                                                row['end_frame']])


class CompareEvents(object):
    """
    input: dictionaries with events from tagged file and events from ml probabilities
    output: list with events
    Function iterates though dictionary with tagged events (or events from ml system) and compares to a dictionary with
    events calculated from ml probabilities(ml events). If it finds intersection between them it adds an event to output list.
    It's adding fist intersection with ml event. After that, it iterates through ml events and compares them
    to tag events. All events labeled as TP, FP, FN based on the results of the function.
    """
    def __init__(self, tag_ev, ml_ev):
        self.list_of_counted_ml_events = []
        self.list_of_counted_tag_events = []
        self.tag = tag_ev
        self.ml = ml_ev
        self.final_list = []
        self.composition_of_events = []
        self._comparing()

    def _comparing(self):
        """
        Function is searching through tagged events and comparing them with ml events for TP and FN events.
        Secondly, it searches through remaining ml events and making FP events from them.
        """
        for line, values_tag in self.tag.items():
            self._true_and_false_events(line, values_tag)
        for line, values_ml in self.ml.items():
            self._false_positive_events(line, values_ml)

    def _false_positive_events(self, line, values_ml):
        """ Producing FP events """
        for ml_event in values_ml:
            if ml_event in self.list_of_counted_ml_events:
                continue
            flag = True
            for key in self.tag:
                if key.startswith(line) and flag:
                    for tag_event in self.tag[key]:
                        if self.intersection(tag_event[1:], ml_event[-2:]) and (tag_event[1:]
                                                                            not in self.list_of_counted_tag_events):
                            flag = False
                            break
                elif not flag:
                    break
            else:
                if flag:
                    self.composition_of_events.append((*list('---'), *ml_event, line, ml_event[0],
                                                       ml_event[-1] - ml_event[-2] + 1, '-', 'FP'))

    @staticmethod
    def intersection(tag_list, ml_list):
        """It takes start and end frames of tagged and ml events and produces two lists from them.
        After that it compares them and returns True or False based on the intersection of them"""
        if list(set(range(tag_list[0], tag_list[1] + 1)) & set(range(ml_list[0], ml_list[1] + 1))):
            return True
        else:
            return False

    def _true_and_false_events(self, line, values_tag):
        """
        Making TP and FN events
        """
        for tag_event in values_tag:
            for ml_event in self.ml[line[0]]:
                if self.intersection(tag_event[1:], ml_event[-2:]) & (ml_event not in self.list_of_counted_ml_events):
                    self.composition_of_events.append(
                        (line, *tag_event[1:], *ml_event, line[0],
                         min(tag_event[1], ml_event[0]), ml_event[-1] - ml_event[-2] + 1, ml_event[0] - tag_event[0], 'TP'))
                    self.list_of_counted_ml_events.append(ml_event)
                    self.list_of_counted_tag_events.append(tag_event)
                    break
                elif ml_event == self.ml[line[0]][-1]:
                    self.composition_of_events.append((line, *tag_event[1:], *list('-----'),
                                                       tag_event[1], *list('--'), 'FN'))


class ReadingParameters(object):
    """
    This class takes the file named "params" as input and parse it to make a dictionary with all the variables in it
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.params = OrderedDict(parameters_file=self.file_name)
        self._get_params_dict()
        self._parse_parameters()
        self._make_dirs()

    def _get_params_dict(self):
        try:
            with open(join('parameters', self.file_name + '.txt')) as f:
                for line in f:
                    key, value = line.rstrip().split(" = ")
                    if 'th' in key:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                        self.params[key] = value
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                        self.params[key] = value
        except FileNotFoundError:
            with open(join('..', 'parameters', self.file_name + '.txt')) as f:
                for line in f:
                    key, value = line.rstrip().split(" = ")
                    if 'th' in key:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                        self.params[key] = value
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                        self.params[key] = value

    def _parse_parameters(self):
        self.params['th'] = self.read_th(self.params['th'], self.params['total_amount_of_ths'])
        self.params['window_size'] = self.params_list(self.params['window_size'])
        self.params['holes'] = self.params_list(self.params['holes'])
        self.params['moving_average_size'] = self.params_list(self.params['moving_average_size'])
        self.params['results_from_events'] = strtobool(self.params['results_from_events'])
        self.params['results_folder_name'] = self.make_folder_name(self.params['results_folder_name'])
        self.params['min_window_size'] = self.params_list(self.params['min_window_size'])

    @staticmethod
    def params_list(param):
        try:
            return list(map(int, param.replace(',', '').split()))
        except AttributeError:
            return [int(param)]

    @staticmethod
    def read_th(params, total_amount_of_ths=2):
        try:
            start, end = params.rstrip().split(" - ")
            start, end = float(start), float(end)
            return np.linspace(start, end, total_amount_of_ths)
        except AttributeError:
            return [params]

    @staticmethod
    def make_folder_name(folder_name):
        folder_name_counter = 0
        while True:
            if not os.path.exists(folder_name + '_{}'.format(folder_name_counter)):
                break
            folder_name_counter += 1
        return folder_name + '_{}'.format(folder_name_counter)

    def _make_dirs(self):
        os.makedirs(join(self.params['results_folder_name'],
                         self.params['result_csv_folder_name']))
        os.makedirs(join(self.params['results_folder_name'],
                         self.params['result_images_folder_name']))
        if self.params['results_from_events']:
            os.makedirs(join(self.params['results_folder_name'],
                             self.params['event_result_files_folder_name']))
        else:
            os.makedirs(join(self.params['results_folder_name'],
                             self.params['prob_result_files_folder_name']))
