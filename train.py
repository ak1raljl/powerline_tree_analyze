import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import numpy as np
import time





def parse_args():
    parser = argparse.ArgumentParser('Training Script')
    parser.add_argument('--model', type=str, default='seg', help='Model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory to save logs and models')
    return parser.parse_args()

def main(args):
    args = parse_args()
    



if __name__ == '__main__':
    args = parse_args()
    main(args)