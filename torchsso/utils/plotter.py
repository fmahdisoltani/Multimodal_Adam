import os
import sys

import numpy as np

import pandas as pd
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams["font.size"] = 21
from matplotlib.ticker import MultipleLocator

colors = ['gray', 'gray', 'black', 'gray']

xlabels = {'iteration': 'iteration',
           'epoch': 'epoch',
           'elapsed_time': 'wall time (min)'}

ylabels = {'accuracy': 'accuracy [%]',
           'loss': 'log likelihood'}

linewidths = [2, 4, 4, 4]

linestyles = ['--', '--', '-', '-']


def plot(idx, label, log_json, xtarget, ytarget, ax):
    with open(log_json, 'r') as f:
        json_text = f.read()
        data = json.loads(json_text)
    df = pd.DataFrame(data)
    df = df.drop_duplicates([xtarget], keep='last')

    if ytarget == 'accuracy':
        y_val = df['accuracy']
    else:
        y_val = df['loss']

    index = y_val.notnull()
    y_val = y_val[index]
    x = df[xtarget][index]

    if xtarget == 'elapsed_time':
        x /= 60  # sec to min

#    ax.plot(x, y_val, label=label, color=colors[idx],
#            linewidth=linewidths[idx], linestyle=linestyles[idx])
    ax.plot(x, y_val, label=label)
    return max(x)


def plot_all(ax, fig, labels, logfiles, xtarget, xlabel, ytarget, ylabel, right=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(linestyle=':')

    x_max = -1

    for idx, (label, log_file) in enumerate(zip(labels, logfiles)):
        _x_max = plot(idx, label, log_file, xtarget, ytarget, ax)
        x_max = max(x_max, _x_max)

def main(labels, logfiles, xtarget, xlabel, figpath):
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2)

    left = 0
    right = 50
    # Loss
    ax = fig.add_subplot(gs[0, 0])
    plot_all(ax, fig, labels, logfiles, xtarget, xlabel, 'loss', ylabels['loss'], right)
    ax.set_ylim(bottom=0, top=None)
    ax.set_xlim(left=left, right=None)
    # plt.yticks([-3.0,-2.5,-2.0,-1.5])
    # plt.yticks([-3.0,-2.5,-2.0,-1.5])

    # Accuracy
    ax = fig.add_subplot(gs[0, 1])
    plot_all(ax, fig, labels, logfiles, xtarget, xlabel, 'accuracy', ylabels['accuracy'], right)
    ax.set_ylim(bottom=0, top=None)
    ax.set_xlim(left=left, right=None)
    # plt.yticks([30,40,50,60,70])
    ax.legend(frameon=True, fontsize=20, loc='lower right')

    fig.tight_layout()
    fig.savefig(figpath)

if __name__ == '__main__':
    argc = len(sys.argv)
    # assert argc > 3, 'You need to specify logdir, xtarget, and figname.'
    # logdir = sys.argv[1]
    # xtarget = sys.argv[2]
    # figname = sys.argv[3]
    figname = "loss.jpg"
    logdir = "/Users/farzaneh/PycharmProjects/multimodal_madam/examples/classification/result_mnist_c2"
    xtarget = "iteration"
    ytarget = "loss"

    logfiles_path = os.path.join(logdir, 'logfiles.csv')
    with open(logfiles_path, 'r') as f:
        reader = csv.reader(f, dialect='excel')
        labels, logfiles = zip(*list(reader))

    figpath = os.path.join(logdir, figname)
    print(figpath)

main(labels, logfiles, xtarget, xlabels[xtarget], figpath)