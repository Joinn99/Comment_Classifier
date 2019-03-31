from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models import KeyedVectors
import numpy as np


def load_data(data_file, is_data=True):
    with open(data_file, 'r', encoding='utf-8') as txtfile:
        lines = txtfile.readlines()
        data_set = list(lines)
        if is_data:
            for i, line in enumerate(lines):
                data_set[i] = line.strip().split(' ')[1:]
            return data_set
        else:
            for i, line in enumerate(lines):
                data_set[i] = int(line.strip())
            return np.array(data_set)


def word2vec(data_set, vec_size, windows, epoch):
    model = Word2Vec(size=vec_size, window=windows, min_count=2, workers=4)
    model.build_vocab(data_set)
    model.train(sentences=data_set, total_examples=len(data_set), total_words=10000,
                epochs=epoch)
    model.wv.save('model/wordvec.kv')
    del model


def doc2vec(data_set, vec_size, windows, epoch, use_model):
    doc = [TaggedDocument(doc, [i]) for i, doc in enumerate(data_set)]
    if use_model:
        model_dbow = Doc2Vec.load('model/dbow.model')
        model_dm = Doc2Vec.load('model/dm.model')
    else:
        model_dm = Doc2Vec(min_count=2, window=windows, vector_size=vec_size,
                           sample=1e-3, negative=5, workers=4, epochs=epoch)
        model_dbow = Doc2Vec(min_count=2, window=windows, vector_size=vec_size,
                             sample=1e-3, negative=5, workers=4, dm=0, epochs=epoch)
        model_dm.build_vocab(doc)
        model_dbow.build_vocab(doc)

    model_dm.train(
        documents=doc, total_examples=model_dm.corpus_count, epochs=model_dm.epochs)
    model_dbow.train(
        documents=doc, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
    model_dbow.save('model/dbow.model')
    model_dm.save('model/dm.model')
    del model_dm, model_dbow


def doc_data_convert(data_set, test_set, vec_size):
    vec_dbow = Doc2Vec.load('model/dbow.model')
    vec_dm = Doc2Vec.load('model/dm.model')
    vec_size = vec_dm.vector_size
    data_array = np.zeros([len(data_set), 2 * vec_size])
    test_array = np.zeros([len(test_set), 2 * vec_size])
    offset = len(data_set)
    for i, _ in enumerate(data_set):
        data_array[i] = np.hstack((vec_dbow[i], vec_dm[i]))
    for j, _ in enumerate(test_set):
        test_array[j] = np.hstack((vec_dbow[j + offset], vec_dm[j + offset]))
    return data_array, test_array


def word_data_convert(data_set, test_set, vec_size):
    vec = KeyedVectors.load('model/wordvec.kv')
    data_array = np.zeros([len(data_set), vec_size])
    test_array = np.zeros([len(test_set), vec_size])
    for i, _ in enumerate(data_set):
        for _, word in enumerate(data_set[i]):
            data_array[i] = data_array[i] + vec[word]
        data_array[i] = data_array[i] / len(data_set[i])
    for j, _ in enumerate(test_set):
        for _, word in enumerate(test_set[j]):
            test_array[j] = test_array[j] + vec[word]
        test_array[j] = test_array[j] / len(test_set[j])
    return data_array, test_array
