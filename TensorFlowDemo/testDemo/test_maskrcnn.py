import keras as K
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('face_path')
parser.add_argument('-t', '--train', default="1")
args = parser.parse_args()

face_path = args.face_path


