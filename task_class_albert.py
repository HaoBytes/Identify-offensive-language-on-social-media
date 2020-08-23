#!/usr/bin/env python3
#coding:utf-8



# text classify model

import time
import argparse
import logging
import os
import sys
import re
import pandas as pd
import numpy as np
import warnings
import math
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.tokenizers import SpTokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
import tensorflow.keras
from keras.layers import Lambda, Dense

#file
logging.basicConfig(level = logging.DEBUG,
            format='[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S',
            filename= os.path.join('./', 'server.log'),
            filemode='a')

#################################################################################################
#Define a StreamHandler, print log information of INFO level or higher to standard error, and add it to the current log processing object#
console = logging.StreamHandler()
console.setLevel(logging.INFO)
#formatter = logging.Formatter('[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
formatter = logging.Formatter('[%(asctime)s]%(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
#################################################################################################

# command parameters
parser = argparse.ArgumentParser(description='classifier albert')
parser.add_argument('--model', type=str, default='large',  help='pre train model: large or xxlarge')
parser.add_argument('--do_train', type=int, default=0,  help='do train')
parser.add_argument('--do_predict', type=int, default=0, help='do predict')
parser.add_argument('--bert_path', type=str, default='../ALbert/albert_xxlarge/', help='bert_path')
parser.add_argument('--file_pre', type=str, default='a', help='data file name')
parser.add_argument('--maxlen', type=int, default=128, help='maxlen')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
args = parser.parse_args()

set_gelu('tanh')  # gelu

do_train = args.do_train
do_predict = args.do_predict
logging.info('do_train:%d, do_predict:%d' % (do_train, do_predict))

file_pre = args.file_pre
# the number of class
dic_nums = {'a':2, 'b':2, 'c':3}
num_classes = dic_nums[file_pre]
maxlen = args.maxlen
batch_size = args.batch_size
logging.info('Running Parm: File: Training_%s, num_classes:%d, maxlen: %d, batch_size: %d' % 
            (file_pre, num_classes, maxlen, batch_size))

# pre-train model
model = args.model
print(model)
if model == 'xxlarge':
    bert_path = r'../ALbert/albert_xxlarge/'
#elif model == 'large':
#    bert_path = r'../ALbert/albert_large/'
else:
    print('')
    sys.exit()
print("123")
config_path = os.path.join(bert_path, 'albert_config.json')
checkpoint_path = os.path.join(bert_path, 'model.ckpt-best')
dict_path = os.path.join(bert_path, '30k-clean.vocab')
spm_path = os.path.join(bert_path, '30k-clean.model')

# load data
def load_data(filename):
    D = []
    with open(filename, encoding='gb2312') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# Create a tokenizer
tokenizer = SpTokenizer(spm_path)
#tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """data generator
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen) 
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# load pre-train model
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
#model.summary()

# an optimizer with piecewise linear learning rate。
# The name parameter is optional, but it is best to fill in to distinguish different derived optimizers。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy',
    #optimizer=Adam(1e-5),  # small lr
    optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
        1000: 1,
        2000: 0.1,
    }),
    metrics=['accuracy'],
)

# evaluate
def evaluate(data, save=0):
    total, right = 0., 0.
    pred = list()
    label = list()
    if num_classes == 2:
        TP = 0.1
        FP = 0.1
        FN = 0.1
        TN = 0.1
        TPR = list()
        FPR = list()
        confusion = [[0,0],[0,0]]
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            pred.append(y_pred)
            label.append(y_true)
            total += len(y_true)
            if y_true == 1 and y_pred == 1:
                TP += 1
                confusion[1][1] += 1
            if y_true == 1 and y_pred == 0:
                FN += 1
                confusion[1][0] += 1
            if y_true == 0 and y_true == 1:
                FP += 1
                confusion[0][1] += 1
            if y_true == 0 and y_true == 0:
                TN += 1
                confusion[0][0] += 1
            print(total)
            right += (y_true == y_pred).sum()
            TPR.append(TP/(TP+FN))
            FPR.append(FP/(FP+TN))

        P = TP/(TP+FP)
        R = TP/(TP+FN)
        P2 = TN/(TN+FN)
        R2 = TN/(FP+TN)
        
        guess = ['NOT','OFF']
        fact = ['NOT','OFF']
        classes = list(set(fact))
        classes.sort()
        plt.figure()
        plt.imshow(confusion, cmap=plt.cm.Blues)
        indices = range(len(confusion))
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        plt.colorbar()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                plt.text(first_index, second_index, confusion[first_index][second_index])

        print("F1-macro: {}".format((2*P*R)/(P+R)))
        print("MCC: {}".format((TP * TN - FP * FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))))
        print("Precision of class 1: {}".format(P))
        print("Precision of class 2: {}".format(P2))
        print("Recall of class 1: {}".format(R))
        print("Recall of class 2: {}".format(R2))
    else:
        TP1 = 0.1
        FN1 = 0.1
        FP1 = 0.1
        TN1 = 0.1
        TP2 = 0.1
        FN2 = 0.1
        FP2 = 0.1
        TN2 = 0.1
        TP3 = 0.1
        FN3 = 0.1
        FP3 = 0.1
        TN3 = 0.1        
        TPR = list()
        FPR = list()
        confusion = [[0,0,0],[0,0,0],[0,0,0]]
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            pred.append(y_pred)
            label.append(y_true)
            total += len(y_true)

            if y_true == 1 and y_pred == 1:
                TP2 += 1
                TN1 += 1
                TN3 += 1 
                confusion[1][1] += 1
            if y_true == 1 and y_pred == 0:
                FP1 += 1
                TN3 += 1
                FN2 += 1
                confusion[1][0] += 1
            if y_true == 0 and y_pred == 1:
                FN1 += 1
                FP2 += 1
                TN3 += 1
                confusion[0][1] += 1
            if y_true == 0 and y_pred == 0:
                TP1 += 1
                TN2 += 1
                TN3 += 1 
                confusion[0][0] += 1
            if y_true == 2 and y_pred == 0:
                FN3 += 1
                FP1 += 1
                TN2 += 1
                confusion[2][0] += 1
            if y_true == 2 and y_pred == 1:
                FN3 += 1
                FP2 += 1
                TN1 += 1
                confusion[2][1] += 1
            if y_true == 0 and y_pred == 2:
                FP3 += 1
                FN1 += 1
                TN2 += 1
                confusion[0][2] += 1
            if y_true == 1 and y_pred == 2:
                FP3 += 1
                FN2 += 1
                TN1 += 1
                confusion[1][2] += 1
            if y_true == 2 and y_pred == 2:
                TP3 += 1
                TN1 += 1
                TN2 += 1
                confusion[2][2] += 1

            print(total)
            right += (y_true == y_pred).sum()
            TPR.append(TP1/(TP1+FN1))
            FPR.append(FP1/(FP1+TN1))

        P1 = TP1/(TP1+FP1)
        R1 = TP1/(TP1+FN1)
        P2 = TP2/(TP2+FP2)
        R2 = TP2/(TP2+FN2)
        P3 = TP3/(TP3+FP3)
        R3 = TP3/(TP3+FN3)
        guess = ['IND', 'OTH', 'GRP']
        fact = ['IND', 'OTH', 'GRP']
        classes = list(set(fact))
        classes.sort()
        plt.figure()
        plt.imshow(confusion, cmap=plt.cm.Blues)
        indices = range(len(confusion))
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        plt.colorbar()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                plt.text(first_index, second_index, confusion[first_index][second_index])

        print("F1-macro: {}".format((2*P1*R1)/(P1+R1)))
        print("MCC: {}".format((TP1 * TN1 - FP1 * FN1)/math.sqrt((TP1+FP1)*(TP1+FN1)*(TN1+FP1)*(TN1+FN1))))
        print("Precision of class 1: {}".format(P1))
        print("Precision of class 2: {}".format(P2))
        print("Precision of class 3: {}".format(P3))
        print("Recall of class 1: {}".format(R1))
        print("Recall of class 2: {}".format(R2))
        print("Recall of class 3: {}".format(R3))

    
        guess = ['IND', 'OTH', 'GRP']
        fact = ['IND', 'OTH', 'GRP']
        classes = list(set(fact))
        classes.sort()
        plt.figure()
        plt.imshow(confusion, cmap=plt.cm.Blues)
        indices = range(len(confusion))
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        plt.colorbar()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                plt.text(first_index, second_index, confusion[first_index][second_index])
    #ROC
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(FPR, TPR, color='darkorange',
         lw=lw) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return (right / total)


#
class Evaluator(keras.callbacks.Callback):
    def __init__(self, f_key=''):
        self.best_val_acc = 0.
        self.f_key = f_key

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model_name = '../models/best_model_%s.weights' % self.f_key
            model.save_weights(model_name)
            logging.info('model save to:%s' % model_name)
        test_acc = evaluate(test_generator)
        logging.info(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

# Load Dataset
logging.info('loading dataset...')
train_data = load_data('../ALbert/data/training_%s_train.tsv' % file_pre)
valid_data = load_data('../ALbert/data/training_%s_val.tsv' % file_pre)
test_data = load_data('../ALbert/data/training_%s_test.tsv' % file_pre)

logging.info('train:valid:test =  %d:%d:%d' % (len(train_data),len(valid_data),len(test_data)) )

# transfet dataset
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

# Training
if do_train:
    logging.info('training model...')
    evaluator = Evaluator(f_key=file_pre)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=1,
        callbacks=[evaluator]
    )
    time_start = time.time()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

if do_predict:
    model_name = './models/best_model_%s.weights' % file_pre
    logging.info('Load model:%s' % model_name)
    model.load_weights(model_name)
    logging.info(u'final test acc: %05f\n' % (evaluate(test_generator)))

