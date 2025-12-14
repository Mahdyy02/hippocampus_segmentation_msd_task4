#!/usr/bin/env python3
"""
Plot training metrics (train vs validation) from history.json
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def to_array(list_of_lists):
    if len(list_of_lists) == 0:
        return np.zeros((0,))
    arr = np.array(list_of_lists)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def mean_axis1(arr):
    if arr.size == 0:
        return np.array([])
    return np.mean(arr, axis=1)

def save_and_show(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, default='./results_hippocampus/history.json')
    parser.add_argument('--outdir', type=str, default='./results_hippocampus/plots')
    args = parser.parse_args()

    if not os.path.exists(args.history):
        raise FileNotFoundError(f"history.json not found: {args.history}")

    with open(args.history, 'r') as f:
        history = json.load(f)

    epochs = np.arange(1, len(history['train_loss']) + 1)

    train_loss = np.array(history['train_loss'], dtype=float)
    val_loss   = np.array(history['val_loss'], dtype=float)

    train_dice = mean_axis1(to_array(history['train_dice']))
    val_dice   = mean_axis1(to_array(history['val_dice']))
    val_nsd    = mean_axis1(to_array(history['val_nsd']))

    os.makedirs(args.outdir, exist_ok=True)

    # ============================================================
    # 1) LOSS: Train vs Validation
    # ============================================================
    fig = plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.grid(True)
    plt.legend()
    save_and_show(fig, os.path.join(args.outdir, 'loss_train_vs_val.png'))

    # ============================================================
    # 2) DICE (MEAN): Train vs Validation
    # ============================================================
    fig = plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_dice, label='Train Dice (mean)')
    plt.plot(epochs, val_dice, label='Validation Dice (mean)')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Training vs Validation Dice (Mean)')
    plt.grid(True)
    plt.legend()
    save_and_show(fig, os.path.join(args.outdir, 'dice_train_vs_val.png'))

    # ============================================================
    # 3) NSD (Validation only – no train NSD logged)
    # ============================================================
    fig = plt.figure(figsize=(9, 5))
    plt.plot(epochs, val_nsd, label='Validation NSD (mean)')
    plt.xlabel('Epoch')
    plt.ylabel('NSD')
    plt.title('Validation NSD (Mean)')
    plt.grid(True)
    plt.legend()
    save_and_show(fig, os.path.join(args.outdir, 'val_nsd.png'))

    print("Plots saved to:", os.path.abspath(args.outdir))

if __name__ == '__main__':
    main()
