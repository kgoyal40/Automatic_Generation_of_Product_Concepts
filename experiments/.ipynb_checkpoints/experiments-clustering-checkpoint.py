import sys

# sys.path.append('/Users/kshitijgoyal/Desktop/musical-context-discovery/src')
sys.path.append('/home/kshitij/musical-context-discovery/src')

from scoop import futures
from lmc.core import *
from lmc.exps import *
from lmc.io import *
import random, time

all_contexts_names, all_contexts = get_filtered_clusters()
context_overview = pd.read_csv(CONTEXT_NAMES_FP)
context_overview = context_overview[context_overview['mc_id'].isin([int(n) for n in all_contexts_names])]
context_overview['# songs'] = [len(all_contexts['000000{}'.format(c)]) for c in context_overview['mc_id']]
context_overview = context_overview[context_overview['# songs'] >= 100]
context_overview.reset_index(inplace=True, drop=True)

CLUSTERING_TARGET_DIR = Path(".").absolute() / "clustering"
NUM_EXPERIMENTS = 10
MIN_PLAYLIST_PER_CONTEXT = 5
MAX_PLAYLIST_PER_CONTEXT = 15
NUM_SOURCE_CONTEXTS = 40
MIN_PLAYLIST_SIZE = 15
MAX_PLAYLIST_SIZE = 40
PARALLEL = True


def compare_source_with_learned_contexts(source_context=None, learned_context=None):
    all_context_songs = all_contexts['000000{}'.format(source_context)]
    return learned_context.score(all_context_songs)


def make_clusters(source_contexts=None, playlists=None, discard_threshold=0,
                  reliable_negative_method=None, rocchio_threshold=0, exp_num=None):
    assert (type(source_contexts) == list)
    assert (type(playlists) == list)
    assert (type(playlists[0]) == Playlist)
    assert (exp_num is not None)

    learner = ContextLearner()
    learner.fit(playlists, n_clusters=NUM_SOURCE_CONTEXTS, reliable_negative_method=reliable_negative_method,
                discard_threshold=discard_threshold, rocchio_threshold=rocchio_threshold)

    for j in range(len(learner.learned_contexts)):
        context = learner.learned_contexts[j]
        for s in source_contexts:
            score = compare_source_with_learned_contexts(source_context=s, learned_context=context)
            print(score)
            clustering_results_file = open(CLUSTERING_TARGET_DIR / "results_clustering_0906.txt", "a+")
            clustering_results_file.write(
                '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(exp_num, j + 1, s,
                                                                          reliable_negative_method,
                                                                          discard_threshold, rocchio_threshold,
                                                                          score['dt-query']['# songs target'],
                                                                          'dt-query',
                                                                          score['dt-query'][
                                                                              '# songs predicted'],
                                                                          score['dt-query']['recall'],
                                                                          score['dt-query']['precision'],
                                                                          score['dt-query']['f1-score']))

            clustering_results_file.write(
                '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(exp_num, j + 1, s, reliable_negative_method,
                                                                          discard_threshold, rocchio_threshold,
                                                                          score['songs-query']['# songs target'],
                                                                          'songs-query',
                                                                          score['songs-query']['# songs predicted'],
                                                                          score['songs-query']['recall'],
                                                                          score['songs-query']['precision'],
                                                                          score['songs-query']['f1-score']))
            clustering_results_file.close()
        del context
    return None


def pipeline(exp_num=None, source_contexts=None):
    assert (exp_num is not None)
    assert (type(source_contexts) == list)
    print('selected source contexts:', source_contexts)
    all_playlists = []

    for s in source_contexts:
        num_playlist_source_context = random.randint(MIN_PLAYLIST_PER_CONTEXT, MAX_PLAYLIST_PER_CONTEXT)
        clustering_overview_file = open(CLUSTERING_TARGET_DIR / "overview_clustering_0906.txt", "a+")
        clustering_overview_file.write(
            '{}, {}, {}\n'.format(exp_num, s, num_playlist_source_context))
        clustering_overview_file.close()

        for j in range(num_playlist_source_context):
            playlist_size = random.randint(MIN_PLAYLIST_SIZE, MAX_PLAYLIST_SIZE)
            playlist_songs = set(random.sample(all_contexts['000000{}'.format(s)], playlist_size))
            onehot = songs.merge(pd.DataFrame(list(playlist_songs), columns=['song_id']), on='song_id',
                                 how='right')
            all_playlists.append(Playlist(songdata=onehot, source_contexts=s))

    inside_input_0 = []
    inside_input_1 = []
    inside_input_2 = []
    inside_input_3 = []
    inside_input_4 = []
    inside_input_5 = []
    for t in [0, 0.1, 0.2]:
        for r in ['l', 'r']:
            for rt in [0]:
                if not (r == 'l' and rt in [0.1, 0.2]):
                    inside_input_0.append(source_contexts)
                    inside_input_1.append(all_playlists)
                    inside_input_2.append(t)
                    inside_input_3.append(r)
                    inside_input_4.append(rt)
                    inside_input_5.append(exp_num)

    print(len(inside_input_0))
    futures.map(make_clusters, inside_input_0, inside_input_1,
                inside_input_2, inside_input_3, inside_input_4, inside_input_5)

    return None


if __name__ == "__main__":
    input_1 = []
    input_2 = []
    overview = pd.DataFrame()
    all_comparisons = pd.DataFrame()

    for i in range(NUM_EXPERIMENTS):
        source_contexts = random.sample(context_overview.mc_id.tolist(), NUM_SOURCE_CONTEXTS)
        input_1.append(i + 1)
        input_2.append(source_contexts)

    if PARALLEL:
        start = time.time()
        list(futures.map(pipeline, input_1, input_2))
        print('total runtime:', time.time() - start)
    else:
        for i in range(len(input_1)):
            pipeline(input_1[i], input_2[i])
