import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from data import data_util


class ImdbClassfier:
    def __init__(self, data_file, target_file, test_file):
        self.data_array = np.array
        self.test_array = np.array
        self.data_set = data_util.load_data(data_file, True)
        self.test_set = data_util.load_data(test_file, True)
        self.target_array = data_util.load_data(target_file, False)
        self.comment_clf = ExtraTreesClassifier

    def __train_clf(self, data, target, trees, depth):
        self.comment_clf = ExtraTreesClassifier(
            n_estimators=trees, max_depth=depth, max_features="log2", n_jobs=-1,
            min_samples_split=50, min_samples_leaf=20, max_leaf_nodes=140,
            min_impurity_decrease=8e-7)
        self.comment_clf.fit(data, target)

    def __data_group(self, vali_group, division):
        train_data = np.vsplit(self.data_array, division)
        vali_data = train_data[vali_group]
        del train_data[vali_group]
        return vali_data, np.vstack(train_data)

    def __target_group(self, vali_group, division):
        train_target = np.hsplit(self.target_array, division)
        vali_target = train_target[vali_group]
        del train_target[vali_group]
        return vali_target, np.hstack(train_target)

    def predict(self, trees, depth):
        self.__train_clf(self.data_array, self.target_array,
                         trees=trees, depth=depth)
        result_array = self.comment_clf.predict(self.test_array)
        np.savetxt('result.txt', result_array.reshape(-1, 1), fmt='%d')


    def data_process(self, isdoc, vector_size, windows, epochs):
        print('Doc model train:')
        if isdoc:
            data_util.doc2vec(
                self.data_set + self.test_set, vector_size, windows, epochs)
        else:
            data_util.word2vec(
                self.data_set, vector_size, windows, epochs)

    def doc_vectorized(self, isdoc, vector_size):
        if isdoc:
            self.data_array, self.test_array = data_util.doc_data_convert(
                self.data_set, self.test_set, vector_size)
        else:
            self.data_array, self.test_array = data_util.word_data_convert(
                self.data_set, self.test_set, vector_size)

    def cross_validation(self, trees, depth, division=5):
        if len(self.data_set) % division == 0:
            vali_prec, train_prec = 0, 0
            for i in range(0, division):
                vali_data, train_data = self.__data_group(i, division)
                vali_target, train_target = self.__target_group(i, division)
                self.__train_clf(train_data, train_target,
                                 trees=trees, depth=depth)
                vali_prec = vali_prec + \
                    self.comment_clf.score(vali_data, vali_target)
                train_prec = train_prec + \
                    self.comment_clf.score(train_data, train_target)
            return vali_prec / division, train_prec / division
        else:
            return 0.
