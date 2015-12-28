#!/usr/bin/env python
'''
| Filename    : create_tok_pkl.py
| Description : Create token files and pkl dictionaries for the morphology data.
| Author      : Pushpendre Rastogi
| Created     : Sun Dec 27 17:03:13 2015 (-0500)
| Last-Updated: Sun Dec 27 18:03:09 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 16
'''
import sys
sys.path.append('/home/prastog3/projects/neural-context/src/python/')
import transducer_data
import rasengan
args = rasengan.Namespace()
args.limit_corpus = 0
args.mix_validation_into_training = 0
args.replace_validation_by_training = 0
args.jump_to_validation = 0
args.win = 1
d = r'/home/prastog3/projects/neural-context/src/python/'
args.train_fn = d + 'transducer/data/train'
args.dev_fn = d + 'transducer/data/dev'
args.test_fn = d + 'transducer/data/test'
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
write_to_file('lower_string.train.tok', train_set_lower)
write_to_file('upper_string.train.tok', train_set_upper)
write_to_file('lower_string.dev.tok', dev_set_lower)
write_to_file('upper_string.dev.tok', dev_set_upper)
data.label2idx['$'] = 0
# {'$': 0, '^': 1
# I.e. BOS = 1, EOS = 0. This is the correct way to create the pickle.
import cPickle as pkl
pkl.dump(data.label2idx, open('dict.pkl', 'wb'))
print data.label2idx
