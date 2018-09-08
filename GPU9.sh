#!/bin/bash

targetrom='breakout'

THEANO_FLAGS='device=gpu0, allow_gc=False' python run_OT.py -r $targetrom --Seed 12| tee LogFiles/breakout/breakout_backward_12.log &
sleep 10

THEANO_FLAGS='device=gpu0, allow_gc=False' python run_OT.py -r $targetrom --Seed 123 | tee LogFiles/breakout/breakout_backward_123.log &
sleep 10

THEANO_FLAGS='device=gpu1, allow_gc=False' python run_OT.py -r $targetrom --Seed 1234 | tee LogFiles/breakout/breakout_backward_1234.log &
sleep 10

THEANO_FLAGS='device=gpu1, allow_gc=False' python run_OT.py -r $targetrom --Seed 12345 | tee LogFiles/breakout/breakout_backward_12345.log &
sleep 10

THEANO_FLAGS='device=gpu2, allow_gc=False' python run_OT.py -r $targetrom --Seed 123456 | tee LogFiles/breakout/breakout_backward_123456.log &
sleep 10