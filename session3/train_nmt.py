#!/usr/bin/env python
from nmt import train, test
#-------------------#
# Dataset Creation. #
#-------------------#
# train = TextIterator(datasets[0], datasets[1],
# data_iterator.TextIterator
# def next()
#   read from source file and map to word index
#   source.append(ss)
#   target.append(tt)
#   return source, target
#   ss = self.source.readline()
#   If there is an empty line then set end_of_data = True
#     if ss == "": raise IOError except IOError: self.end_of_data = True
#   Now Strip and split ss
#   Now map
#   ss = [(self.source_dict[w] if w in self.source_dict else 1)
#          for w in ss]
# Basically 1 is the index of OOV token.
DATASETS = ['lower_string.train.tok',
            'upper_string.train.tok']
VALID_DATASETS = ['lower_string.dev.tok',
                  'upper_string.dev.tok']
#----------------------#
# Dictionary Creation. #
#----------------------#
DICTIONARIES = ['dict.pkl', 'dict.pkl']

def main(do_test):
    n_words = 28
    f = test if do_test else train
    validerr = f(saveto='model_session3.npz',
                 reload_=(do_test),
                 dim_word=150,
                 dim=124,
                 n_words=n_words,
                 n_words_src=n_words,
                 decay_c=0.,
                 clip_c=1.,
                 lrate=1e-4,
                 optimizer='adadelta',
                 maxlen=25,
                 batch_size=32,
                 valid_batch_size=32,
                 datasets=DATASETS,
                 valid_datasets=VALID_DATASETS,
                 dictionaries=DICTIONARIES,
                 validFreq=500000,
                 dispFreq=10,
                 saveFreq=100,
                 sampleFreq=500,
                 use_dropout=True or (not do_test))
    return validerr

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        assert sys.argv[1] == '--test'
        main(True)
    else:
        main(False)
        main(True)
