from os import listdir, path
from src.io import *
import pandas as pd
from bs4 import BeautifulSoup as bs
import math
import toml

songs = pd.read_csv(ONEHOT_SONG_ID_FP)


def get_filtered_clusters():
    context_names = listdir(CONTEXT_DATA_DIR / "2020-10-01")
    context_names = [c.split('.')[0] for c in context_names if '.txt' in c]
    context_names_filtered = []
    for c in context_names:
        c_file = c + '.xml'
        if path.isfile(CONTEXT_DATA_DIR / 'xml' / c_file):
            with open(CONTEXT_DATA_DIR / 'xml' / c_file, "r") as file:
                content = file.readlines()
                content = "".join(content)
                bs_content = bs(content, "html.parser")
            if bs_content.find("selection-rule", {"field": 'Dancing Style'}) is None:
                if bs_content.find("selection-rule", {"field": 'Instrument'}) is None:
                    if bs_content.find("selection-rule", {"field": 'Origin'}) is None:
                        if bs_content.find("selection-rule", {"field": 'Instrumental Accompaniment'}) is None:
                            if bs_content.find("selection-rule", {"field": 'Compilation'}) is None:
                                if bs_content.find("selection-rule", {"field": 'Music Trend'}) is None:
                                    if bs_content.find("selection-rule", {"field": 'Continent'}) is None:
                                        if bs_content.find("selection-rule", {"field": 'Performance Rights'}) is None:
                                            if bs_content.find("selection-rule", {"field": 'Voice'}) is None:
                                                context_names_filtered.append(c)

    all_contexts = {}
    all_valid_songs = set(songs['song_id'].tolist())
    for c in context_names_filtered:
        c_file = c + '.xml.txt'
        s = open(CONTEXT_DATA_DIR / "/2020-10-01" / c_file, "r").read().splitlines()
        all_contexts[c.split('.')[0]] = set([int(i) for i in s]).intersection(all_valid_songs)
    all_contexts.pop('0000008979', None)  # removing test context made by koen

    return context_names_filtered, all_contexts


def get_subcontext_count(context_name):
    c_file = context_name + '.xml'
    if path.isfile(CONTEXT_DATA_DIR / 'xml' / c_file):
        with open(CONTEXT_DATA_DIR / 'xml' / c_file, "r") as file:
            content = file.readlines()
            content = "".join(content)
            bs_content = bs(content, "html.parser")
            return len(bs_content.find_all('subcontext'))


def evaluator(predicted=None, target=None):
    if len(predicted) > 0:
        recall = len(predicted.intersection(target)) / len(target)
        precision = len(predicted.intersection(target)) / len(predicted)
    else:
        recall = math.nan
        precision = math.nan
    
    if recall == 0 and precision == 0:
        f1 = math.nan
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    
    return {'# songs target': len(target), '# songs predicted': len(predicted),
            'recall': recall,
            'precision': precision,
            'f1-score': f1}


def load_encoding_from_disk():
    with open(ENCODING_FP) as f:
        e = toml.load(f)

    for k, v in e.items():
        e[k] = {int(a): b for a, b in v.items()}

    return e


def get_songs_from_query_song_query(query):
    if len(query) > 0:
        encoding = load_encoding_from_disk()
        songs_filtered = songs.copy()

        all_filters = []
        for q in query:
            all_keys = list(query[q].keys())
            all_filters_q = []
            for k in all_keys:
                filter_with = ['''(songs_filtered["{}_{}"] == 1)'''.format(k, [d for d in encoding[k] if
                                                                               encoding[k][d] == v][0]) for v in
                               query[q][k]]

                filter_with = ' | '.join(filter_with)
                all_filters_q.append('({})'.format(filter_with))

            all_filters_q = ' & '.join(all_filters_q)
            all_filters.append('({})'.format(all_filters_q))
        all_filters = ' | '.join(all_filters)
        filter_with = 'songs_filtered[{}]'.format(all_filters)
        songs_filtered = eval(filter_with)
        songs_filtered.reset_index(inplace=True, drop=True)
        return set(songs_filtered['song_id'].tolist())
    else:
        return set()


def get_songs_from_query_dt_rules(query):
    if len(query) > 0:
        encoding = load_encoding_from_disk()
        all_filters = []
        for q in query:
            all_keys = list(query[q].keys())
            all_keys.remove('count')
            all_filters_q = []
            for k in all_keys:
                filter_with = ['''(songs_filtered["{}_{}"] == 1)'''.format(k, [d for d in encoding[k] if
                                                                               str(encoding[k][d]) == v][0])
                               if v.split(' ')[0] != 'Not' else
                               '''(songs_filtered["{}_{}"] == 0)'''.format(k, [d for d in encoding[k] if
                                                                               str(encoding[k][d]) == ' '.join(
                                                                                   v.split(' ')[1:])][0]) for v in
                               query[q][k]]
                all_filters_q.extend(filter_with)

            all_filters_q = ' & '.join(all_filters_q)
            all_filters.append('({})'.format(all_filters_q))
        all_filters = ' | '.join(all_filters)
        filter_with = 'songs_filtered[{}]'.format(all_filters)
        songs_filtered = eval(filter_with)
        songs_filtered.reset_index(inplace=True, drop=True)
        return set(songs_filtered['song_id'].tolist())
    else:
        return set()
