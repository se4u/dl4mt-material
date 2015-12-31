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


#----------------------#
# Dictionary Creation. #
#----------------------#

def main(args):
    DATASETS = ['lower_string.train.tok',
                'upper_string.train.tok']
    VALID_DATASETS = ['lower_string.dev.tok',
                      'upper_string.dev.tok']
    DICTIONARIES = ['dict.pkl', 'dict.pkl']
    saveto = 'model_session3_use_dropout=%s.npz'%str(args.use_dropout)
    if args.prefix is not None:
        prefix_adder = lambda arr: [args.prefix + '.' + e for e in arr]
        DATASETS = prefix_adder(DATASETS)
        VALID_DATASETS = prefix_adder(VALID_DATASETS)
        DICTIONARIES = prefix_adder(DICTIONARIES)
        saveto = args.prefix + '.' + saveto
    print 'saveto', saveto
    if args.do_test:
        from nmt import test
        f = test
    else:
        from nmt import train
        f = train
    validerr = f(saveto=saveto,
                 reload_=args.reload_, # Remove saveto to disable reloading
                 use_dropout=args.use_dropout,
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
    arg_parser.add_argument('--use_dropout', default=0, type=int, help='Default={0}')
    arg_parser.add_argument('--reload_', default=0, type=int, help='Default={0}')
    arg_parser.add_argument('--prefix', default=None, type=str, help='Default={None}')
    _args=arg_parser.parse_args()
    with rasengan.debug_support():
        main(_args)
        if not args.test:
            args.test = True
            main(args)
