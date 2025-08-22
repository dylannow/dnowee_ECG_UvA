#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d data -o outputs -s scores.csv
#
# where 'data' is a folder containing files with the reference signals and labels for the data, 'outputs' is a folder containing
# files with the outputs from your models, and 'scores.csv' (optional) is a collection of scores for the model outputs.
#
# Each data or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs are also
# described on the Challenge webpage.

import argparse
import numpy as np
import os
import os.path
import sys

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Evaluate the Challenge model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-s', '--score_file', type=str, required=False)
    return parser

# Evaluate the models.
def evaluate_model(data_folder, output_folder):
    # Find the records.
    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No records found.')

    labels = np.zeros(num_records)
    binary_outputs = np.zeros(num_records)
    probability_outputs = np.zeros(num_records)

    # Load the labels and model outputs.
    for i, record in enumerate(records):
        label_filename = os.path.join(data_folder, record)
        print(f"Loading label from: {label_filename}")
        label = load_label(label_filename)

        output_filename = os.path.join(output_folder, record + '.txt')
        output = load_text(output_filename)
        binary_output = get_label(output, allow_missing=True)
        probability_output = get_probability(output, allow_missing=True)

        # Missing model outputs are interpreted as zero for the binary and probability outputs.
        labels[i] = label
        if not is_nan(binary_output):
            binary_outputs[i] = binary_output
        else:
            binary_outputs[i] = 0
        if not is_nan(probability_output):
            probability_outputs[i] = probability_output
        else:
            probability_outputs[i] = 0

    # Evaluate the model outputs.
    challenge_score = compute_challenge_score(labels, probability_outputs)
    auroc, auprc = compute_auc(labels, probability_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)

    confusion_matrix = compute_confusion_matrix(labels, binary_outputs)

    # Per-class metrics from the confusion matrix
    tp, fn = confusion_matrix[0, 0], confusion_matrix[0, 1]
    fp, tn = confusion_matrix[1, 0], confusion_matrix[1, 1]

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')  # Recall, TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')  # TNR
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')    # PPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')          # Negative predictive value

    return challenge_score, auroc, auprc, accuracy, f_measure, confusion_matrix, sensitivity, specificity, precision, npv

# Run the code.
def run(args):
    results = evaluate_model(args.data_folder, args.output_folder)
    (
        challenge_score, auroc, auprc, accuracy, f_measure,
        confusion_matrix, sensitivity, specificity, precision, npv
    ) = results

    output_string = (
        f'Challenge score: {challenge_score:.3f}\n'
        f'AUROC: {auroc:.3f}\n'
        f'AUPRC: {auprc:.3f}\n'
        f'Accuracy: {accuracy:.3f}\n'
        f'F-measure: {f_measure:.3f}\n'
        f'Confusion Matrix:\n{confusion_matrix.astype(int)}\n'
        f'Sensitivity (TPR): {sensitivity:.3f}\n'
        f'Specificity (TNR): {specificity:.3f}\n'
        f'Precision (PPV): {precision:.3f}\n'
        f'NPV: {npv:.3f}\n'
    )

    if args.score_file:
        save_text(args.score_file, output_string)
    else:
        print(output_string)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
