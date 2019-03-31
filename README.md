# IMDb Comment Classifier

### Dependencies
+ numpy
+ scikit-learn
+ gensim

### Test
The classifier uses Word2Vec/Doc2Vec to process comment text.

To train Word2Vec/Doc2Vec model ,use:
```
python3 train.py -m traindoc [other options]
```

To classify the comments using decision tree or bayes classifier, use:
```
python3 train.py -m buildclf [other options]
```
The accuracy result will print on screen.

To generate prediction file, use:
```
python3 train.py -m predict
```

To see help of setting parameters, use:
```
python3 train.py -h
```
