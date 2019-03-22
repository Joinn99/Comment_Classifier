#!/usr/bin/python
# encoding: utf-8

import argparse
import imdb


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", help="The mode: traindoc/buildtree/predict",
        choices=['traindoc', 'buildtree', 'predict'], default='buildtree')
    parser.add_argument(
        "-v", "--doc2vec", help="Use Doc2Vec (Default is Word2vec)", action="store_true")
    parser.add_argument(
        "-e", "--epoch", help="Training epoches", type=int, default=10)
    parser.add_argument(
        "-t", "--trees", help="The number of trees in random forest", type=int, default=20)
    parser.add_argument(
        "-d", "--depth", help="The max depth of decision tree", type=int, default=6)
    parser.add_argument(
        "-w", "--window", help="The window of Word2Vec/Doc2Vec", type=int, default=16)
    parser.add_argument(
        "-s", "--size", help="The word/doc vector size", type=int, default=64)
    return parser


def train(args):
    imdbclf = imdb.ImdbClassfier(
        'data/train_data.txt', 'data/train_labels.txt', 'data/test_data.txt')
    if args.mode == 'traindoc':
        imdbclf.data_process(isdoc=args.doc2vec, epochs=args.epoch,
                             windows=args.window, vector_size=args.size)
    elif args.mode == 'buildtree':
        imdbclf.doc_vectorized(isdoc=args.doc2vec, vector_size=args.size)
        vali_precision, train_precision = imdbclf.cross_validation(
            trees=args.trees, depth=args.depth)
        info = str('\tValidation Precision: ' + str(vali_precision * 100) + '%' +
                   '\t\tTrain Precision: ' + str(train_precision * 100) + '%')
        print(info)
    else:
        imdbclf.doc_vectorized(isdoc=args.doc2vec, vector_size=args.size)
        imdbclf.predict(trees=args.trees, depth=args.depth)


if __name__ == "__main__":
    PARSER = parse_arguments()
    ARGS = PARSER.parse_args()
    train(ARGS)
