#!/usr/bin/python
# encoding: utf-8

import argparse
import imdb


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", help="The mode: traindoc/buildtree/predict",
        choices=['traindoc', 'buildclf', 'predict'], default='buildclf')
    parser.add_argument(
        "-p", "--type", help="The classifier type: extratree/randomforest/gaussianbayes",
        choices=['extratree', 'randomforest', 'gaussianbayes'], default='gaussianbayes')
    parser.add_argument(
        "-a", "--auto", help="Auto train", action="store_true")
    parser.add_argument(
        "-v", "--doc2vec", help="Use Doc2Vec (Default is Word2vec)", action="store_true")
    parser.add_argument(
        "-u", "--usemodel", help="Use trained model file", action="store_true")
    parser.add_argument(
        "-e", "--epoch", help="Training epoches", type=int, default=1)
    parser.add_argument(
        "-t", "--trees", help="The number of trees in random forest", type=int, default=100)
    parser.add_argument(
        "-d", "--depth", help="The max depth of decision tree", type=int, default=10)
    parser.add_argument(
        "-w", "--window", help="The window of Word2Vec/Doc2Vec", type=int, default=24)
    parser.add_argument(
        "-s", "--size", help="The word/doc vector size", type=int, default=64)
    return parser


def train(args):
    imdbclf = imdb.ImdbClassfier(
        'data/train_data.txt', 'data/train_labels.txt', 'data/test_data.txt')
    if args.mode == 'traindoc':
        imdbclf.data_process(args=args)
    elif args.mode == 'buildclf':
        imdbclf.doc_vectorized(args=args)
        vali_precision, train_precision = imdbclf.cross_validation(args)
        info = str('\tValidation Precision: ' + str(vali_precision * 100)[:5] + '%' +
                   '\tTrain Precision: ' + str(train_precision * 100)[:5] + '%')
        print(info)
    else:
        imdbclf.doc_vectorized(args=args)
        imdbclf.predict(args=args)


def autotrain(args):
    imdbclf = imdb.ImdbClassfier(
        'data/train_data.txt', 'data/train_labels.txt', 'data/test_data.txt')
    if args.usemodel:
        vec_sizes = [-1]
    else:
        vec_sizes = [80, 84, 88, 92, 96, 100, 104]
    epoches = args.epoch
    args.epoch = 1
    for size in vec_sizes:
        print('Vector size: ' + str(size))
        args.size = size
        for epoch in range(0, epoches):
            imdbclf.data_process(args=args)
            imdbclf.doc_vectorized(args=args)
            vali_precision, train_precision = imdbclf.cross_validation(args)
            info = str('Epoch: ' + str(epoch + 1) + '\tVali Prec: ' + str(vali_precision * 100)[:5]
                       + '%' + '\tTrain Prec: ' + str(train_precision * 100)[:5] + '%')
            print(info)


if __name__ == "__main__":
    PARSER = parse_arguments()
    ARGS = PARSER.parse_args()
    if ARGS.auto:
        autotrain(ARGS)
    else:
        train(ARGS)
