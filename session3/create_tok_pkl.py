#!/usr/bin/env python
'''
| Filename    : create_tok_pkl.py
| Description : Create token files and pkl dictionaries for the morphology data.
| Author      : Pushpendre Rastogi
| Created     : Sun Dec 27 17:03:13 2015 (-0500)
| Last-Updated: Wed Dec 30 21:46:28 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 19
'''
import sys
import os
def main(script_arg):
    sys.path.append('/home/prastog3/projects/neural-context/src/python/')
    import transducer_data
    import rasengan
    prefix = script_arg.prefix
    d = (r'/home/prastog3/projects/neural-context/src/python/transducer/data'
         if prefix is None
         else script_arg.dir)
    train_fn = os.path.join(d, 'train')
    dev_fn = os.path.join(d, 'dev')
    test_fn = os.path.join(d, 'test')
    args = rasengan.Namespace()
    args.limit_corpus = 0
    args.mix_validation_into_training = 0
    args.replace_validation_by_training = 0
    args.jump_to_validation = 0
    args.win = 1
    args.train_fn = train_fn
    args.dev_fn = dev_fn
    args.test_fn = test_fn
    args.partition_dev_into_test = 0
    args.partition_dev_into_train = 0
    data = transducer_data.main(args)
    train_set_lower = [list(e) + ['$'] for (e, _) in data.train_data]
    train_set_upper = [['^'] + list(e) + ['$'] for (_, e) in data.train_data]
    dev_set_lower = [list(e) + ['$'] for (e, _) in data.val_data]
    dev_set_upper = [['^'] + list(e) + ['$'] for (_, e) in data.val_data]
    def write_to_file(fn, lst):
        with open(fn, 'wb') as f:
            for elem in lst:
                f.write(' '.join(elem))
                f.write('\n')
    write_to_file(prefix + '.lower_string.train.tok', train_set_lower)
    write_to_file(prefix + '.upper_string.train.tok', train_set_upper)
    write_to_file(prefix + '.lower_string.dev.tok', dev_set_lower)
    write_to_file(prefix + '.upper_string.dev.tok', dev_set_upper)
    data.label2idx['$'] = 0
    # {'$': 0, '^': 1
    # I.e. BOS = 1, EOS = 0. This is the correct way to create the pickle.
    import cPickle as pkl
    pkl.dump(data.label2idx, open(prefix + '.dict.pkl', 'wb'))
    print data.label2idx

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description='Create Token Pickle.')
    arg_parser.add_argument('--prefix', default=None, type=str, help='Default={None}')
    arg_parser.add_argument('--dir', default=None, type=str, help='Default={None}')
    main(arg_parser.parse_args())
