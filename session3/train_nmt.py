#!/usr/bin/env python
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
import rasengan

DATASETS = ['lower_string.train.tok',
            'upper_string.train.tok']
VALID_DATASETS = ['lower_string.dev.tok',
                  'upper_string.dev.tok']
#----------------------#
# Dictionary Creation. #
#----------------------#
DICTIONARIES = ['dict.pkl', 'dict.pkl']

def main(do_test=0, use_dropout=0, reload_=0):
    saveto = 'model_session3_use_dropout=%s.npz'%str(use_dropout)
    print 'saveto', saveto, 'reload_', reload_
    if do_test:
        from nmt import test
        f = test
    else:
        from nmt import train
        f = train
    validerr = f(saveto=saveto,
                 reload_=reload_, # Remove saveto to disable reloading
                 use_dropout=use_dropout,
                 dim_word=150,
                 dim=124,
                 n_words=28, n_words_src=28,
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
                 validFreq=500,
                 dispFreq=50,
                 saveFreq=500,
                 sampleFreq=500,
                 )
    return validerr

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--test', default=0, type=int, help='Default={0}')
    arg_parser.add_argument('--dropout', default=0, type=int, help='Default={0}')
    arg_parser.add_argument('--reload_', default=0, type=int, help='Default={0}')
    args=arg_parser.parse_args()
    with rasengan.debug_support():
        main(args.test, args.dropout)
        if not args.test:
            main(True, args.dropout)
