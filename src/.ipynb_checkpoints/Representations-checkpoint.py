import pandas as pd
import sklearn.tree as tree
import numpy as np


class DTRepr:
    def __init__(self, positive_samples=None, negative_samples=None, encoding=None, discard_threshold=0.1, onehot=None):
        if positive_samples is None or negative_samples is None or encoding is None or onehot is None:
            raise ValueError('please provide valid inputs')

        self.query_s = None
        self.query_r = None
        self.tree = None
        self.leaf_metadata = {}
        self.discard_threshold = discard_threshold
        self.fit(positive_samples, negative_samples, encoding, onehot)
        return

    @staticmethod
    def make_training_data(positive_samples, negative_samples, onehot):
        y = [1] * len(positive_samples)
        y.extend([0] * len(negative_samples))
        all_samples = positive_samples.copy()
        all_samples.extend(negative_samples)
        all_samples = pd.DataFrame(all_samples, columns=['song_id'])
        X = onehot.merge(all_samples, on='song_id', how='right')
        X.loc[:, 'target'] = y
        X = X.sample(frac=1)
        X.reset_index(inplace=True, drop=True)
        y = X['target'].tolist()
        all_samples = X['song_id'].tolist()
        X = X.drop(columns=['song_id', 'target', 'title', 'group'])
        return X, y, all_samples

    def fit(self, positive_samples, negative_samples, encoding, onehot):
        X, y, all_samples = DTRepr.make_training_data(positive_samples, negative_samples, onehot)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y)

        values = clf.tree_.value
        leave_id = clf.apply(X)
        self.tree = clf

        query_s = {}
        songs_in_leaves = {}
        for leaf in np.unique(leave_id):
            if values[leaf][0][0] == 0:
                data_indices_in_leaf = np.where(leave_id == leaf)[0]
                data_leaf = X.iloc[data_indices_in_leaf, :]
                songs_in_leaves[leaf] = set([all_samples[s] for s in data_indices_in_leaf])
                if values[leaf][0][1] >= self.discard_threshold * len(positive_samples):
                    representation_leaf = list(data_leaf.sum())
                    representation_leaf = [v / data_leaf.shape[0] for v in representation_leaf]
                    query_s[leaf] = DTRepr.get_query_songs(representation_leaf, X.columns, encoding)

        print(sum(y), sum([len(songs_in_leaves[k]) for k in query_s]))
        self.query_s = query_s
        self.leaf_metadata = songs_in_leaves
        self.query_r = self._get_query_dt_rules(onehot.columns[3:], encoding)

    @staticmethod
    def get_query_songs(representation, columns, encoding):
        q = {}
        for i in range(len(representation)):
            if representation[i] > 0:
                feature = ['_'.join(columns[i].split('_')[:-1]) if len(columns[i].split('_')[:-1]) > 1 else
                           columns[i].split('_')[:1][0]][0]
                value = encoding[feature][int(columns[i].split('_')[-1])]
                if feature in list(q.keys()):
                    q[feature].append(value)
                else:
                    q[feature] = [value]
        for i in encoding.keys():
            if i not in list(q.keys()):
                q[i] = '-'
        return q

    def _get_query_dt_rules(self, columns, encoding):
        values = self.tree.tree_.value
        query_r = {}
        for l in self.query_s.keys():
            leave_id = l
            paths = {}
            for leaf in np.unique(leave_id):
                path_leaf = []
                self.find_path(0, path_leaf, leaf)
                paths[leaf] = np.unique(np.sort(path_leaf))
            for key in paths:
                if values[key][0][0] == 0:
                    query_r[key] = self.get_rule(paths[key], columns, key, encoding)
        return query_r

    def get_rule(self, path, column_names, leaf_id, encoding):
        children_left = self.tree.tree_.children_left
        values = self.tree.tree_.value
        feature = self.tree.tree_.feature
        mask = {'count': values[leaf_id][0][1]}
        for index, node in enumerate(path):
            if index != len(path) - 1:
                f = ['_'.join(column_names[feature[node]].split('_')[:-1])
                     if len(column_names[feature[node]].split('_')[:-1]) > 1
                     else column_names[feature[node]].split('_')[:1][0]][0]
                v = encoding[f][int(column_names[feature[node]].split('_')[-1])]
                if children_left[node] == path[index + 1]:
                    if f in mask.keys():
                        mask[f].add('Not {}'.format(v))
                    else:
                        mask[f] = {'Not {}'.format(v)}
                else:
                    mask[f] = {str(v)}
        return mask

    def find_path(self, node_numb, path, x):
        children_left = self.tree.tree_.children_left
        children_right = self.tree.tree_.children_right
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False
        if children_left[node_numb] != -1:
            left = self.find_path(children_left[node_numb], path, x)
        if children_right[node_numb] != -1:
            right = self.find_path(children_right[node_numb], path, x)
        if left or right:
            return True
        path.remove(node_numb)
        return False


class Playlist:
    def __init__(
        self,
        songdata=None,
        source_contexts=None
    ):
        self.onehot = songdata
        self.source_contexts = source_contexts
        self.prototype = self.prototype_repr()
        
    def prototype_repr(self):
        onehot_processed = self.onehot.drop(columns=['song_id', 'title', 'group'])
        counts = np.array(onehot_processed.sum())
        return counts / onehot_processed.shape[0]
