from scoop import futures
from src.core import *
from src.exps import *
from src.io import *
import random
import pickle

all_contexts_names, all_contexts = get_filtered_clusters()
context_overview = pd.read_csv(CONTEXT_NAMES_FP)
context_overview = context_overview[context_overview['mc_id'].isin([int(n) for n in all_contexts_names])]
context_overview['# songs'] = [len(all_contexts['000000{}'.format(c)]) for c in context_overview['mc_id']]
context_overview = context_overview[context_overview['# songs'] > 100]
context_overview.reset_index(inplace=True, drop=True)

CONTEXT_TARGET_DIR = Path(".").absolute() / "learned_contexts"


def get_query(context_id=None, positive_sample_size=20, discard_threshold=0, noise=0,
              reliable_negative_method=None, rocchio_threshold=0, exp_num=1):
    assert (reliable_negative_method is not None)
    print('context id: {}, sample size: {}, discard threshold: {}, noise ratio: {}, '
          'reliable_negative_method: {}, rocchio threshold {}, experiment num: {}'.format(context_id,
                                                                                          positive_sample_size,
                                                                                          discard_threshold, noise,
                                                                                          reliable_negative_method,
                                                                                          rocchio_threshold, exp_num))

    all_songs_context = all_contexts[context_id]
    positive_samples = random.sample(all_contexts[context_id], min(positive_sample_size, len(all_contexts[context_id])))

    if noise > 0:
        noisy_songs = random.choices(list(set(songs['song_id'].tolist()) - all_songs_context),
                                     k=math.ceil(noise * len(positive_samples)))
        positive_samples = list(set(positive_samples).union(set(noisy_songs)))

    learner = QueryLearner()
    learner.fit(positive_samples, onehot=songs, discard_threshold=discard_threshold,
                reliable_negative_method=reliable_negative_method, rocchio_threshold=rocchio_threshold)
    score = learner.score(all_songs_context)
    pickle.dump(learner.learned_queries,
                open(CONTEXT_TARGET_DIR / "queries_{}_{}_{}_{}_{}_{}_{}.p".format(context_id, positive_sample_size,
                                                                                  discard_threshold,
                                                                                  noise, rocchio_threshold,
                                                                                  reliable_negative_method,
                                                                                  exp_num), "wb"))

    print([context_id, positive_sample_size, discard_threshold, noise, reliable_negative_method,
           len(learner.learned_queries['dt']), score])

    output_file = open("results_query_learning_1006.txt", "a+")
    output_file.write(
        '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(context_id, positive_sample_size,
                                                                              discard_threshold,
                                                                              noise,
                                                                              reliable_negative_method,
                                                                              rocchio_threshold, exp_num,
                                                                              len(learner.learned_queries['dt']),
                                                                              learner.metadata[
                                                                                  'tree'].tree.tree_.max_depth,
                                                                              score['dt-query']['# songs target'],
                                                                              'dt-query',
                                                                              score['dt-query']['# songs predicted'],
                                                                              score['dt-query']['recall'],
                                                                              score['dt-query']['precision'],
                                                                              score['dt-query']['f1-score']))
    output_file.write(
        '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(context_id, positive_sample_size,
                                                                              discard_threshold,
                                                                              noise,
                                                                              reliable_negative_method,
                                                                              rocchio_threshold, exp_num,
                                                                              len(learner.learned_queries['songs']),
                                                                              learner.metadata[
                                                                                  'tree'].tree.tree_.max_depth,
                                                                              score['dt-query']['# songs target'],
                                                                              'songs-query',
                                                                              score['songs-query']['# songs predicted'],
                                                                              score['songs-query']['recall'],
                                                                              score['songs-query']['precision'],
                                                                              score['songs-query']['f1-score']))
    output_file.close()

    return None


if __name__ == "__main__":

    parallel = True
    num_experiments_per_config = 5
    input_1 = []
    input_2 = []
    input_3 = []
    input_4 = []
    input_5 = []
    input_6 = []
    input_7 = []
    for c in list(context_overview.mc_id):
        for sample_size in [20, 30, 50, 100, 200, 300, 400, 500, 700, 1000]:
            for threshold in [0, 0.1, 0.2]:
                for noise_ratio in [0, 0.1, 0.2]:
                    for reliable_negative_method in ['l', 'r']:
                        for n in range(num_experiments_per_config):
                            for rt in [0]:
                                if not (reliable_negative_method == 'l' and rt in [0.1, 0.2]):
                                    input_1.append('000000{}'.format(c))
                                    input_2.append(sample_size)
                                    input_3.append(threshold)
                                    input_4.append(noise_ratio)
                                    input_5.append(reliable_negative_method)
                                    input_6.append(rt)
                                    input_7.append(n)

    if parallel:
        list(futures.map(get_query, input_1, input_2, input_3, input_4, input_5, input_6, input_7, ))
        print('parallel process finished!')
    else:
        for i in range(len(input_1)):
            get_query(input_1[i], input_2[i], input_3[i], input_4[i], input_5[i], input_6[i], input_7[i])
