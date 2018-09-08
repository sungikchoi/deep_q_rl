#!/bin/bash

targetrom='breakout'
THEANO_FLAGS='device=gpu0, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train1.log &
sleep 10

THEANO_FLAGS='device=gpu0, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train2.log &
sleep 10

THEANO_FLAGS='device=gpu1, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train3.log &
sleep 10

THEANO_FLAGS='device=gpu1, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train4.log &
sleep 10


THEANO_FLAGS='device=gpu2, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train5.log &
sleep 10

THEANO_FLAGS='device=gpu2, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train6.log &
sleep 10

THEANO_FLAGS='device=gpu3, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train7.log &
sleep 10


THEANO_FLAGS='device=gpu3, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train8.log &
sleep 10

THEANO_FLAGS='device=gpu4, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train9.log &
sleep 10


THEANO_FLAGS='device=gpu4, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train10.log &
sleep 10

THEANO_FLAGS='device=gpu5, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train11.log &
sleep 10

THEANO_FLAGS='device=gpu5, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train12.log &
sleep 10

THEANO_FLAGS='device=gpu6, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train13.log &
sleep 10

THEANO_FLAGS='device=gpu6, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train14.log &
sleep 10

THEANO_FLAGS='device=gpu7, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train15.log &
sleep 10

THEANO_FLAGS='device=gpu7, allow_gc=False' python run_OT.py -r $targetrom --close2 | tee train16.log &
sleep 10
