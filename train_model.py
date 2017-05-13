#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#adapted from https://github.com/Sentdex/pygta5
"""
Created on Thu Apr 20 18:21:57 2017

@author: Mariyan
"""

import numpy as np
from alexnet import alexnet
from random import shuffle

#setting up variables
WIDTH=80
HEIGHT=60
LR=1e-3
EPOCHS=8
MODEL_NAME='odo-{}-{}-{}-epochs-model'.format(LR,'alexnet',EPOCHS)

model=alexnet(WIDTH,HEIGHT,LR)

train_data=np.load('train_data.npy')
shuffle(train_data)
train=train_data[:-500]
test=train_data[-500:]

#setting up the labels and the feature sets
X=np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y=([i[1] for i in train])

test_x=np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y=([i[1] for i in test])

model.fit({'input':X},{'targets':Y},n_epoch=EPOCHS,
          validation_set=({'input':test_x},{'targets':test_y}),
                          snapshot_step=500,show_metric=True,
                          run_id=MODEL_NAME)
# tensorboard --logdir=foo:C:/Users/krist/Desktop/odo/Visual-Odometry/Visual-Odometry/log

model.save(MODEL_NAME)