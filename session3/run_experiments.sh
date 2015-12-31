#!/usr/bin/env bash

debug=echo
ensure_files () {
    prefix=$1
    task=$2
    fold=$3
    if [[ ! -e $prefix.lower_string.train.tok ]];
    then
        ./create_tok_pkl.py \
            --prefix $prefix \
            --dir $HOME/projects/neural-context/res/celex/$task/0500/$fold ;
    fi
    if [[ ! -e $prefix.lower_string.train.tok ]];
    then
        exit 1;
    fi
}
#-----------------------------------------------------------------------------------#
# The goal of this script is to run experiments on folds of the transduction tasks. #
#-----------------------------------------------------------------------------------#
# The basic idea is that there are folds that contain loops.
for task in 13SIA-13SKE 2PIE-13PKE 2PKE-z rP-pA
do
    for fold in {0..4}
    do
        for trial in 1
        do
            prefix="task=$task.fold=$fold.trial=$trial"
            # Corresponding to each prefix you have to ensure.
            # $prefix.lower_string.train.tok
            # $preifx.upper_string.train.tok
            # $prefix.lower_strin.dev.tok
            # $prefix.upper_string.dev.tok
            # $prefix.dict.pkl
            ensure_files $prefix $task $fold || exit $? ;
            $debug "./train_nmt.py --prefix $prefix | tee $prefix.log" ;
        done
    done
done

#----------------------------------------------------------------------------------#
# The goal of this part is to run experiments on folds of the lemmatization tasks. #
#----------------------------------------------------------------------------------#
