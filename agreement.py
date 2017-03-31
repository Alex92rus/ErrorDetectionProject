from collections import defaultdict
from datetime import datetime

from fleiss_kappa import fleiss_kappa

import csv
import convert_m2
import sys
import fce_api as fd
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import settings


def get_annotations(golden=defaultdict(list)):
    """
    Retrieves all annotations per sentence from AMT batch
    Args:
        golden: expert annotation per sentence: opt
    Returns:
        result_annotations: dict of the form sentence-> annotations
    """
    result_annotations = defaultdict(list)
    with open(settings.AMT_FILE) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            sentence = row['Input.sentence']
            answer = json.loads(row['Answer.ChosenWord'])
            for annotation in answer['selectedTokens']:
                result_annotations[sentence].append(annotation['start'])
        for key in result_annotations.keys():
            for error_index in golden[key]:
                result_annotations[key].append(error_index)
    return result_annotations


def create_confusion_matrix(data, predictions):
    """
    Produces a confusion matrix in a form of a dictionary from (gold_label,guess_label)` pairs to counts.
    Args:
        data: list containing the gold labels.
        predictions: list containing the prediction labels

    Returns:
        confusion matrix in form of dictionary with counts for (gold_label, guess_label)
    """
    confusion = defaultdict(int)
    for y_gold, y_guess in zip(data, predictions):
        confusion[(y_gold, y_guess)] += 1
    return confusion


def plot_confusion_matrix_dict(matrix_dict, classes=None, rotation=45, outside_label=''):
    """
    Plots the confusion matrix
    Args:
        matrix_dict: the dict of confusion matrix - output of create_confusion_matrix
        classes: list containing the classes for the category labels, if empty, whole numbering will be used for
        category names
        rotation: the degree orientation of the axis labels
        outside_label: the label to disregard -  excluded by default
    """
    labels = set([y for y, _ in matrix_dict.keys()] + [y for _, y in matrix_dict.keys()])
    sorted_labels = sorted(labels, key=lambda x: -x)
    matrix = np.zeros((len(sorted_labels), len(sorted_labels)))
    for i1, y1 in enumerate(sorted_labels):
        for i2, y2 in enumerate(sorted_labels):
            if y1 != outside_label or y2 != outside_label:
                matrix[i1, i2] = matrix_dict[y1, y2]

    threshold = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, int(matrix[i, j]),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > threshold else "black")

    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    if classes is None:
        classes = sorted_labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=rotation)
    plt.yticks(tick_marks, classes)
    plt.xlabel('turker labels')
    plt.ylabel('gold labels')
    plt.tight_layout()
    plt.show()


def full_evaluation_table(confusion_matrix, classes=list()):
    """
    Produces a pandas data-frame with Precision, F1 and Recall for all labels.
    Args:
        confusion_matrix: the confusion matrix to calculate metrics from.
        classes: the categories of the confusion matrix

    Returns:
        a pandas Dataframe with one row per gold label, and one more row for the aggregate of all labels.
    """
    labels = sorted(list({l for l, _ in confusion_matrix.keys()} | {l for _, l in confusion_matrix.keys()}))
    if len(labels) == len(classes):
        labels = classes
    gold_counts = defaultdict(int)
    guess_counts = defaultdict(int)
    for (gold_label, guess_label), count in confusion_matrix.items():
        if gold_label != "None":
            gold_counts[gold_label] += count
            gold_counts["[All]"] += count
        if guess_label != "None":
            guess_counts[guess_label] += count
            guess_counts["[All]"] += count

    result_table = []
    for label in labels:
        if label != "None":
            result_table.append((label, gold_counts[label], guess_counts[label], *evaluate(confusion_matrix, {label})))

    result_table.append(("[All]", gold_counts["[All]"], guess_counts["[All]"], *evaluate(confusion_matrix)))
    return pd.DataFrame(result_table, columns=('Label', 'Gold', 'Guess', 'Precision', 'Recall', 'F1'))


def evaluate(conf_matrix, label_filter=None):
    """
    Evaluate Precision, Recall and F1 based on a confusion matrix as produced by `create_confusion_matrix`.
    Args:
        conf_matrix: a confusion matrix in form of a dictionary from `(gold_label,guess_label)` pairs to counts.
        label_filter: a set of gold labels to consider. If set to `None` all labels are considered.

    Returns:
        Precision, Recall, F1 triple.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for (gold, guess), count in conf_matrix.items():
        if label_filter is None or gold in label_filter or guess in label_filter:
            if gold == 'None' and guess != gold:
                fp += count
            elif gold == 'None' and guess == gold:
                tn += count
            elif gold != 'None' and guess == gold:
                tp += count
            elif gold != 'None' and guess == 'None':
                fn += count
            else:  # both gold and guess are not-None, but different
                fp += count if label_filter is None or guess in label_filter else 0
                fn += count if label_filter is None or gold in label_filter else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec * recall > 0 else 0.0
    return prec, recall, f1


def evaluate_metrics(conf_matrix, label_filter=None):
    """
    Evaluate Precision, Recall and F1 based on a confusion matrix as produced by `create_confusion_matrix`.
    Args:
        conf_matrix: a confusion matrix in form of a dictionary from `(gold_label,guess_label)` pairs to counts.
        label_filter: a set of gold labels to consider. If set to `None` all labels are considered.

    Returns:
        Precision, Recall, F1, F0.5 triple.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for (gold, guess), count in conf_matrix.items():
        if label_filter is None or gold in label_filter or guess in label_filter:
            if gold == 'None' and guess != gold:
                fp += count
            elif gold == 'None' and guess == gold:
                tn += count
            elif gold != 'None' and guess == gold:
                tp += count
            elif gold != 'None' and guess == 'None':
                fn += count
            else:  # both gold and guess are not-None, but different
                fp += count if label_filter is None or guess in label_filter else 0
                fn += count if label_filter is None or gold in label_filter else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec * recall > 0 else 0.0
    f05 = ((1 + (0.5 ** 2)) * prec * recall) / (recall + (0.5 ** 2) * prec) if prec * recall > 0 else 0.0
    return prec, recall, f1, f05


def extract_sentences_with_errors():
    """
    Extracts AMT sentences along with their errors to a file.
    """
    with open(settings.TRAINING_DATA_FILE, 'r') as file:
        # read the lines
        readlines = file.readlines()
        with open(settings.AMT_SENTENCE_BATCH, 'r') as amt_batch:
            csv_reader = csv.DictReader(amt_batch)
            with open(settings.AMT_FCE_M2, 'w+') as destination:
                for row in csv_reader:
                    sentence = row['sentence']
                    i = 0
                    while i < len(readlines):
                        if sentence == readlines[i][2:-1]:
                            destination.write(readlines[i])
                            i += 1
                            while readlines[i][0] != 'S' and i < len(readlines):
                                destination.writelines(readlines[i])
                                i += 1
                        else:
                            i += 1


# compare the annotations with gold
def compare_annotations(gold_sentences, annotation_labels):
    """
    Produces gold and annotation error detection labels from given annotations and gold data
    Args:
        gold_sentences: a list  of tuples containing the sentences and the related gold error annotations.
        annotation_labels: labels from the annotation representing the start index of the error

    Returns:
        gold and predicted labels
    """
    gold = []
    predicted = []

    for sentence in gold_sentences:
        labels = annotation_labels[sentence[0][1:]]
        for label in labels:
            counted = 0
            error_spans = sentence[1]
            if label == - 2 and len(error_spans) == 0:
                gold.append(0)
                predicted.append(0)
                counted = 1
            if label == -2 and len(error_spans) > 0:
                gold.append(1)
                predicted.append(0)
                counted = 1
            for error_span in error_spans:
                if error_span[0] <= int(label) < error_span[1]:
                    gold.append(1)
                    predicted.append(1)
                    counted = 1
                if label == str(error_span[0]) and label == str(error_span[1]):
                    gold.append(1)
                    predicted.append(1)
                    counted = 1
            if counted == 0:
                gold.append(0)
                predicted.append(1)
    return gold, predicted


def create_agreement_dictionary(annotations, gold_labels, shadow=False):
    """
    Produces the agreement dictionary used for inter-rater agreement and accuracy scores
    sentence -> (worker_id, annotations)
    Args:
        annotations: the annotations for each sentence
        gold_labels: the golden annotations
        shadow: apply no labeled annotations
    Returns:
        agreement_dictionary: a dictionary for inter-rater agreement
    """
    agreement_dictionary = defaultdict(list)
    annotations_for_sentence = defaultdict(list)
    with open(settings.AMT_FILE) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            sentence = row['Input.sentence']
            worker_id = row['WorkerId']
            answer = json.loads(row['Answer.ChosenWord'])
            worker_annotations = [x['start'] for x in answer['selectedTokens']]
            if len(annotations_for_sentence[sentence]) < 1:
                annotations_for_sentence[sentence] = list(set(annotations[sentence]))
            tokens = re.split(r'(\s+)', sentence)
            shadow_annotations = []
            if shadow:
                shadow_annotations = [0] * (len(tokens) + 2 - len(annotations_for_sentence[sentence]))
            explicit_annotations = [1 if x in worker_annotations else 0 for x in annotations_for_sentence[sentence]]
            agreement_dictionary[sentence].append((worker_id, explicit_annotations + shadow_annotations))
    for sentence in agreement_dictionary.keys():
        if len(annotations_for_sentence[sentence]) < 1:
            annotations_for_sentence[sentence] = list(set(annotations[sentence]))
        tokens = re.split(r'(\s+)', sentence)
        shadow_annotations = []
        if shadow:
            shadow_annotations = [0] * (len(tokens) + 2 - len(annotations_for_sentence[sentence]))
        explicit_annotations = [1 if x in gold_labels[sentence] else 0 for x in annotations_for_sentence[sentence]]
        agreement_dictionary[sentence].append(('expert', explicit_annotations + shadow_annotations))
    return agreement_dictionary, annotations_for_sentence


def create_binary_agreement_dictionary(gold_labels):
    """
    Produces the binary agreement dictionary used for inter-rater agreement and accuracy scores
    Binary puts only 1 annotation per sentence - without an error or with error
    Args:
        gold_labels: the golden annotations
    Returns:
        agreement_dictionary: a dictionary for inter-rater agreement
    """
    agreement_dictionary = defaultdict(list)
    with open(settings.AMT_FILE) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            sentence = row['Input.sentence']
            worker_id = row['WorkerId']
            answer = json.loads(row['Answer.ChosenWord'])
            if answer['selectedTokens'][0]['start'] == -2:
                annotation = 0
            else:
                annotation = 1
            agreement_dictionary[sentence].append((worker_id, [annotation]))
    for key in agreement_dictionary.keys():
        if gold_labels[key][0] == -2:
            annotation = 0
        else:
            annotation = 1
        agreement_dictionary[key].append(('expert', [annotation]))
    return agreement_dictionary


def create_gold_dict(data):
    """
    Produces gold annotation labels in the same format as the selected tokens from the workers
    Args:
        data: list of the golden  data - fce_api standard (sentence, error_span) list

    Returns:
        selected tokens of the golden annotations
    """
    gold_dict = defaultdict(list)
    for sentence, error_spans in data:
        if len(error_spans) == 0:
            gold_dict[sentence[1:]].append(-2)
        else:
            for span in error_spans:
                gold_dict[sentence[1:]].append(span[0])
                for i in range(span[0] + 1, span[1]):
                    gold_dict[sentence[1:]].append(i)
    return gold_dict


def calculate_agreement(agreement_dictionary):
    """
    Calculates agreement averaging the accuracies from all possible combinations of turkers (Snow et al,. 2008)
    Args:
         agreement_dictionary: holding sentence annotation records - settings.RESPONSE_COUNT
         from non-experts and one expert
         sentence -> list of annotations (size settings.RESPONSE_COUNT + 1)

    Returns:
        The accuracies from combined agreement from one to nine non-experts with the expert
    """
    sequence = list(range(settings.RESPONSE_COUNT))
    combinations = []
    for i in range(settings.RESPONSE_COUNT + 1):
        combinations.append(list(itertools.combinations(sequence, i)))
    accuracies = [0]
    standard_deviations = [0]
    for i in range(1, settings.RESPONSE_COUNT + 1):
        current_combinations = combinations[i]
        combination_accuracies = []
        for combination in current_combinations:
            correct = 0
            incorrect = 0
            for sentence in agreement_dictionary.keys():
                expert_annotations = agreement_dictionary[sentence][-1][1]
                chosen_annotations = [agreement_dictionary[sentence][x][1] for x in combination]
                votes = np.sum(chosen_annotations, axis=0)
                result_votes = [0] * len(votes)
                for j in range(len(votes)):
                    if votes[j] < len(chosen_annotations) / 2:
                        result_votes[j] = 0
                    elif votes[j] > len(chosen_annotations) / 2:
                        result_votes[j] = 1
                    else:
                        result_votes[j] = 2
                for j in range(len(votes)):
                    if result_votes[j] == 2:
                        correct += 0.5
                        incorrect += 0.5
                    elif expert_annotations[j] == result_votes[j]:
                        correct += 1
                    else:
                        incorrect += 1
            combination_accuracy = correct / (correct + incorrect)
            combination_accuracies.append(combination_accuracy)
        standard_deviation = np.std(combination_accuracies)
        standard_deviations.append(standard_deviation)
        accuracy = sum(combination_accuracies) / len(combination_accuracies)
        accuracies.append(accuracy)
    return accuracies, standard_deviations


def calculate_agreement_random(agreement_dictionary):
    """
    Calculates agreement averaging the accuracies from all possible combinations of turkers
    random breaking ties
    Args:
         agreement_dictionary: holding sentence annotation records - settings.RESPONSE_COUNT
         from non-experts and one expert
         sentence -> list of annotations (size settings.RESPONSE_COUNT + 1)

    Returns:
        The accuracies from combined agreement from one to nine non-experts with the expert
    """
    sequence = list(range(settings.RESPONSE_COUNT))
    combinations = []
    for i in range(settings.RESPONSE_COUNT + 1):
        combinations.append(list(itertools.combinations(sequence, i)))
    accuracies = [0]
    standard_deviations = [0]
    for i in range(1, settings.RESPONSE_COUNT + 1):
        current_combinations = combinations[i]
        combination_accuracies = []
        for combination in current_combinations:
            correct = 0
            incorrect = 0
            for sentence in agreement_dictionary.keys():
                expert_annotations = agreement_dictionary[sentence][-1][1]
                chosen_annotations = [agreement_dictionary[sentence][x][1] for x in combination]
                votes = np.sum(chosen_annotations, axis=0)
                chair = 0
                if len(combination) % 2 == 0:
                    chair = random.choice([x for x in list(combination)])
                result_votes = [0] * len(votes)
                for j in range(len(votes)):
                    if votes[j] < len(chosen_annotations) / 2:
                        result_votes[j] = 0
                    elif votes[j] > len(chosen_annotations) / 2:
                        result_votes[j] = 1
                    else:
                        result_votes[j] = agreement_dictionary[sentence][chair][1][j]
                for j in range(len(votes)):
                    if expert_annotations[j] == result_votes[j]:
                        correct += 1
                    else:
                        incorrect += 1
            combination_accuracy = correct / (correct + incorrect)
            combination_accuracies.append(combination_accuracy)
        standard_deviation = np.std(combination_accuracies)
        standard_deviations.append(standard_deviation)
        accuracy = sum(combination_accuracies) / len(combination_accuracies)
        accuracies.append(accuracy)
    return accuracies, standard_deviations


def calculate_agreement_sum(agreement_dictionary):
    """
    Calculates agreement over the sum (logical or) of annotations e.g [1, 0, 1] x [1, 0, 0] = [1, 0, 1]
    Args:
         agreement_dictionary: holding sentence annotation records - settings.RESPONSE_COUNT
         from non-experts and one expert
         sentence -> list of annotations (size settings.RESPONSE_COUNT + 1)

    Returns:
        The accuracies from combined agreement from one to nine non-experts with the expert
    """
    sequence = list(range(settings.RESPONSE_COUNT))
    combinations = []
    for i in range(settings.RESPONSE_COUNT + 1):
        combinations.append(list(itertools.combinations(sequence, i)))
    accuracies = [0]
    standard_deviations = [0]
    for i in range(1, settings.RESPONSE_COUNT + 1):
        current_combinations = combinations[i]
        combination_accuracies = []
        for combination in current_combinations:
            correct = 0
            incorrect = 0
            for sentence in agreement_dictionary.keys():
                expert_annotations = agreement_dictionary[sentence][-1][1]
                chosen_annotations = [agreement_dictionary[sentence][x][1] for x in combination]
                votes = np.sum(chosen_annotations, axis=0)
                result_votes = [0] * len(votes)
                for j in range(len(votes)):
                    if votes[j] > 0:
                        result_votes[j] = 1
                    else:
                        result_votes[j] = 0
                for j in range(len(result_votes)):
                    if expert_annotations[j] == result_votes[j]:
                        correct += 1
                    else:
                        incorrect += 1
            combination_accuracy = correct / (correct + incorrect)
            combination_accuracies.append(combination_accuracy)
        standard_deviation = np.std(combination_accuracies)
        standard_deviations.append(standard_deviation)
        accuracy = sum(combination_accuracies) / len(combination_accuracies)
        accuracies.append(accuracy)
    return accuracies, standard_deviations


def calculate_agreement_incremental(agreement_dictionary):
    """
    Calculates the agreement by incrementally adding additional turker rather than taking all the combinations of
    turkers
    Args:
         agreement_dictionary: holding sentence annotation records - 9 from non-experts and 1 expert
         sentence -> list of annotations (size settings.RESPONSE_COUNT + 1)

    Returns:
        The accuracies from combined agreement from one to nine non-experts with the expert
    """
    combinations = []
    for i in range(settings.RESPONSE_COUNT + 1):
        combinations.append([tuple(range(i))])
    print(combinations)
    accuracies = [0]
    for i in range(1, settings.RESPONSE_COUNT + 1):
        current_combinations = combinations[i]
        combination_accuracies = []
        for combination in current_combinations:
            correct = 0
            incorrect = 0
            for sentence in agreement_dictionary.keys():
                expert_annotations = agreement_dictionary[sentence][-1][1]
                chosen_annotations = [agreement_dictionary[sentence][x][1] for x in combination]
                votes = np.sum(chosen_annotations, axis=0)
                chair = 0
                if len(combination) > 0 and len(combination) % 2 == 0:
                    chair = combination[-1]
                result_votes = [0] * len(votes)
                for j in range(len(votes)):
                    if votes[j] < len(chosen_annotations) / 2:
                        result_votes[j] = 0
                    elif votes[j] > len(chosen_annotations) / 2:
                        result_votes[j] = 1
                    else:
                        result_votes[j] = agreement_dictionary[sentence][chair][1][j]
                for j in range(len(votes)):
                    if expert_annotations[j] == result_votes[j]:
                        correct += 1
                    else:
                        incorrect += 1
            combination_accuracy = correct / (correct + incorrect)
            combination_accuracies.append(combination_accuracy)
        accuracy = sum(combination_accuracies) / len(combination_accuracies)
        accuracies.append(accuracy)
    return accuracies


def calculate_agreement_second(agreement_dictionary):
    """
    Args:
         agreement_dictionary: holding sentence annotation records - 9 from non-experts and 1 expert
         sentence -> list of annotations (size settings.RESPONSE_COUNT + 1)

    Returns:
        The accuracies from combined agreement from one to nine non-experts with the expert
    """
    sequence = list(range(settings.RESPONSE_COUNT))
    combinations = []
    for i in range(settings.RESPONSE_COUNT + 1):
        combinations.append(list(itertools.combinations(sequence, i)))
    accuracies = [0]
    for i in range(1, settings.RESPONSE_COUNT + 1):
        current_combinations = combinations[i]
        sentence_votes = defaultdict(list)
        correct = 0
        incorrect = 0
        for combination in current_combinations:
            for sentence in agreement_dictionary.keys():
                chosen_annotations = [agreement_dictionary[sentence][x][1] for x in combination]
                votes = np.sum(chosen_annotations, axis=0)
                for j in range(len(votes)):
                    votes[j] = votes[j] / len(combination)
                sentence_votes[sentence].append(votes)
        for key in sentence_votes.keys():
            result_votes = [0] * len(sentence_votes[key][0])
            expert_annotations = agreement_dictionary[key][-1][1]
            for votes in sentence_votes[key]:
                for k in range(len(votes)):
                    result_votes[k] += votes[k]
            for j in range(len(result_votes)):
                aggregated_vote = result_votes[j] / len(result_votes)
                if aggregated_vote < 0.5:
                    result_votes[j] = 0
                elif aggregated_vote > 0.5:
                    result_votes[j] = 1
                else:
                    result_votes[j] = 0
                if result_votes[j] == expert_annotations[j]:
                    correct += 1
                else:
                    incorrect += 1
        accuracy = correct / (correct + incorrect)
        accuracies.append(accuracy)
    return accuracies


def extract_information_per_turker(filename):
    """
    Extracts dictionary with turker's details
    Returns:
        user_information: dictionary with user details
    """
    user_information = {}
    with open(filename) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            worker_id = row['WorkerId']
            location_info = json.loads(row['Answer.ClientLocation'])
            quality_info = json.loads(row['Answer.CountTries'])
            if worker_id not in user_information:
                if 'backUpLocation' in location_info:
                    user_information[worker_id] = location_info['backUpLocation']
                elif 'latitude' in location_info and 'longitude' in location_info:
                    user_information[worker_id] = {'latitude': location_info['latitude'],
                                                   'longitude': location_info['longitude']}
                else:
                    user_information[worker_id] = {'latitude': 'unknown', 'longitude': 'unknown'}
                user_information[worker_id]['responseTimes'] = []
                user_information[worker_id]['countTries'] = 0
                user_information[worker_id]['tasks'] = 0
            user_information[worker_id]['countTries'] += quality_info['countTries']
            user_information[worker_id]['tasks'] += 1
            user_information[worker_id]['triesPerTask'] = user_information[worker_id]['countTries'] / \
                                                          user_information[worker_id]['tasks']
            user_information[worker_id]['acceptTime'] = row['AcceptTime']
            user_information[worker_id]['submitTime'] = row['SubmitTime']
            time_of_start = row['AcceptTime'].split()[3]
            time_of_submit = row['SubmitTime'].split()[3]
            user_information[worker_id]['responseTimes'].append(
                get_time_difference(time_of_start, time_of_submit).total_seconds())
    for worker_id in user_information.keys():
        user_information[worker_id]['averageResponseTime'] = np.mean(user_information[worker_id]['responseTimes'])
    return user_information


def test_results_information(filename):
    """
    metrics from the test experiment
    Returns:
        user_information: dictionary with user details
    """
    user_information = {}
    with open(filename) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            worker_id = row['WorkerId']
            if worker_id not in user_information:
                user_information[worker_id] = {}
                user_information[worker_id]['responseTimes'] = []
                user_information[worker_id]['countTries'] = 0
                user_information[worker_id]['tasks'] = 0
            user_information[worker_id]['acceptTime'] = row['AcceptTime']
            user_information[worker_id]['submitTime'] = row['SubmitTime']
            time_of_start = row['AcceptTime'].split()[3]
            time_of_submit = row['SubmitTime'].split()[3]
            user_information[worker_id]['responseTimes'].append(
                get_time_difference(time_of_start, time_of_submit).total_seconds())
    for worker_id in user_information.keys():
        user_information[worker_id]['averageResponseTime'] = np.mean(user_information[worker_id]['responseTimes'])
    return user_information


def get_time_difference(start, submit):
    FMT = '%H:%M:%S'
    tdelta = datetime.strptime(submit, FMT) - datetime.strptime(start, FMT)
    return tdelta

def extract_coordinates(user_information):
    """
    Extracts the turker's coordinates: (latitude, longitude)
    Args:
        user_information: dict with the user information of the turkers (returned from @extract_information_per_turker
    Returns:
        locations: latitude, longitude) pairs
    """
    coordinates = []
    for turker_id in user_information.keys():
        latitude = user_information[turker_id]['latitude']
        longitude = user_information[turker_id]['longitude']
        coordinates.append((latitude, longitude))
    return coordinates


def extract_country_count(user_information):
    """
    Extracts the turker's coordinates: (latitude, longitude)
    Args:
        user_information: dict with the user information of the turkers (returned from @extract_information_per_turker)
    Returns:
        country_counts: dictionary with turker count per country
    """
    country_counts = defaultdict(int)
    for turker_id in user_information.keys():
        if 'country' in user_information[turker_id]:
            country = user_information[turker_id]['country']
            country_counts[country] += 1
        else:
            print(user_information[turker_id]['latitude'], ' , ', user_information[turker_id]['longitude'])
    return country_counts


def extract_state_count(user_information):
    """
    Extracts the turker's state:
    Args:
        user_information: dict with the user information of the turkers (returned from @extract_information_per_turker)
    Returns:
        country_counts: dictionary with turker count per country
    """
    state_counts = defaultdict(int)
    for turker_id in user_information.keys():
        if 'region' in user_information[turker_id] and 'country' in user_information[turker_id]:
            if user_information[turker_id]['country'] == 'US':
                state_counts[user_information[turker_id]['region']] += 1
    return state_counts


def agreement_minimum_hamming(agreement_dictionary, turker_accuracies):
    """
    Calculates the best pseudo turker based on the closest response to the expert annotations.
    Utilizing hamming distance.
    Args:
        agreement_dictionary: the agreement dictionary with turker IDs and annotations
        turker_accuracies: the dictionary with the accuracies of the turkers
    Return:
        the accuracy with expert
    """
    matched_turkers = {}
    correct = 0
    total = 0
    annotator_correlation_vector = []
    gold_correlation_vector = []
    for sentence in agreement_dictionary.keys():
        expert_annotations = agreement_dictionary[sentence][-1][1]
        min_distance = len(expert_annotations) + 1
        min_worker_id = ''
        min_index = -1
        for i in range(settings.RESPONSE_COUNT):
            distance = hamming(expert_annotations, agreement_dictionary[sentence][i][1])
            worker_id = agreement_dictionary[sentence][i][0]
            if distance < min_distance:
                min_distance = distance
                min_worker_id = worker_id
                min_index = i
        correct += len(expert_annotations) - min_distance
        total += len(expert_annotations)
        annotator_correlation_vector.extend(agreement_dictionary[sentence][min_index][1])
        gold_correlation_vector.extend(expert_annotations)
        worker_accuracy = turker_accuracies[min_worker_id][0][1]
        if min_worker_id not in matched_turkers.keys():
            matched_turkers[min_worker_id] = (worker_accuracy, 1)
        else:
            matched_turkers[min_worker_id] = (worker_accuracy, matched_turkers[min_worker_id][1] + 1)
    accuracy = correct / total
    return accuracy, matched_turkers, annotator_correlation_vector, gold_correlation_vector


def hamming(correct, observed):
    """
    Calculates hamming distance between correct code and observed code with possible errors
    Args:
        correct: the correct code as list (binary values)
        observed: the given code as list (binary values)
    Returns:
        distance: the hamming distance between correct and observed code
    """
    distance = 0
    for i in range(len(correct)):
        if correct[i] != observed[i]:
            distance += 1
    return distance


def accuracy_per_turker(agreement_dictionary, annotation_limit=sys.maxsize):
    """
    Calculates accuracy for each participated turker - unique workId
    Args:
        agreement_dictionary: the agreement dictionary with turker IDs and annotations
    Returns:
        dictionary relating the turker with their accuracy and HITs completed
    """
    turker_accuracies = defaultdict(list)
    guesses_count = defaultdict(int)
    for sentence in sorted(agreement_dictionary.keys()):
        expert_annotations = agreement_dictionary[sentence][-1][1]
        for i in range(settings.RESPONSE_COUNT):
            worker_id = agreement_dictionary[sentence][i][0]
            if len(turker_accuracies[worker_id]) == 0:
                turker_accuracies[worker_id].append((0, 0))
                turker_accuracies[worker_id].append(0)
            if guesses_count[worker_id] < annotation_limit:
                guesses = agreement_dictionary[sentence][i][1]
                for j in range(len(guesses)):
                    if guesses[j] == 1:
                        if guesses[j] == expert_annotations[j]:
                            turker_accuracies[worker_id][0] = (turker_accuracies[worker_id][0][0] + 1,
                                                               turker_accuracies[worker_id][0][1])
                        turker_accuracies[worker_id][0] = (turker_accuracies[worker_id][0][0],
                                                           turker_accuracies[worker_id][0][1] + 1)
                        guesses_count[worker_id] += 1
                        if guesses_count[worker_id] == annotation_limit:
                            break
                turker_accuracies[worker_id][1] += 1
    counter = 0
    for worker_id in sorted(turker_accuracies.keys()):
        counter += 1
        print(counter)
        turker_accuracies[worker_id][0] = turker_accuracies[worker_id][0][0] / turker_accuracies[worker_id][0][1]
        turker_accuracies[worker_id] = [(turker_accuracies[worker_id][1], turker_accuracies[worker_id][0])]
    return turker_accuracies


def run_display_confusion():
    """
    Execute this function to display the confusion matrix and the precision recall table.
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    annotations = get_annotations()
    gold, predicted = compare_annotations(gold_data, annotations)
    cm = create_confusion_matrix(gold, predicted)
    plot_confusion_matrix_dict(cm, classes=['Error', 'No Error'])
    precision_and_recall = full_evaluation_table(cm)
    print(precision_and_recall)


def plot_metrics(metric_data, x_scope, y_scope, labels, standard_deviations=list(),):
    """
    Plots the metric along with its' standard deviation
    Args:
        metric_data: agreement accuracies
        standard_deviations:  the standard deviations corresponding to the accuracies
        x_scope: the visible part of the x axis
        y_scope: the visible part of the y axis
    """
    plt.plot(metric_data)
    x_s = list(range(settings.RESPONSE_COUNT + 1))
    if len(standard_deviations) > 0:
        plt.errorbar(x_s[:len(metric_data)], metric_data, standard_deviations, linestyle='None', marker='^')
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
    plt.tight_layout()
    axes = plt.gca()
    axes.set_xlim(x_scope)
    axes.set_ylim(y_scope)
    plt.grid()
    plt.show()


def plot_multiple_accuracies(accuracies, x_scope, y_scope, legend_labels=None, plot_labels=settings.ACCURACY_LABELS,
                             standard_deviations=None, loc='upper right'):
    """
    Plots the accuracies with their standard deviation
    Args:
        accuracies: agreement accuracies
        x_scope: the visible part of the x axis
        y_scope: the visible part of the y axis
        legend_labels: the labels in the legend
        standard_deviations:  the standard deviations corresponding to the accuracies
    """

    x_s = list(range(settings.RESPONSE_COUNT + 1))
    if legend_labels is None:
        legend_labels = [str(x) for x in range(len(accuracies[0]))]
    legend_handles = []
    for i in range(len(accuracies)):
        legend_handles.append(plt.plot(accuracies[i], label=legend_labels[i]))
        if len(standard_deviations[i]) > 0 and standard_deviations is not None:
            plt.errorbar(x_s[:len(accuracies[i])], accuracies[i], standard_deviations[i], linestyle='None', marker='^')
    plt.xlabel(plot_labels['xlabel'])
    plt.ylabel(plot_labels['ylabel'])
    plt.tight_layout()
    plt.legend(loc=loc)
    axes = plt.gca()
    axes.set_xlim(x_scope)
    axes.set_ylim(y_scope)
    plt.grid()
    plt.show()


def inter_rater_f05_sentence(agreement_dictionary, golden=False):
    """
    Computes the inter-rater agreement in terms of f05
    Args:
        agreement_dictionary: the annotations used for agreement
    """
    add_golden = 0
    if golden:
        add_golden = 1
    sequence = list(range(settings.RESPONSE_COUNT + add_golden))
    combinations = []
    for i in range(1, settings.RESPONSE_COUNT + add_golden):
        combinations.append(list(itertools.combinations(sequence, i)))
    annotation_vectors = []
    # generate an annotation vector for each index
    for i in range(settings.RESPONSE_COUNT + add_golden):
        annotation_vectors.append(generate_annotation_vector(agreement_dictionary,i))
    f05_scores = [0]
    standard_deviations = [0]
    # combinations
    for i in range(settings.RESPONSE_COUNT - 1 + add_golden):
        current_combinations = combinations[i]
        # judgement
        f05_means = []
        for j in range(settings.RESPONSE_COUNT + add_golden):
            combination_index = [combination for combination in current_combinations if j not in combination]
            f05_array = []
            sentences_annotation_map = defaultdict(list)
            for combination in combination_index:
                max_f05_annotation_vector = []
                sentence_counter = 0
                for sentence in sorted(agreement_dictionary.keys()):
                    sentence_max = []
                    sentence_counter += 1
                    max_f05 = 0
                    for k in combination:
                        conf_matrix = create_confusion_matrix(agreement_dictionary[sentence][j][1],
                                                              agreement_dictionary[sentence][k][1])
                        _, _, _, f05 = evaluate_metrics(conf_matrix)
                        if f05 >= max_f05:
                            max_f05 = f05
                            sentence_max = agreement_dictionary[sentence][k][1]
                    sentences_annotation_map[sentence] = sentence_max
                    max_f05_annotation_vector.extend(sentence_max)
                conf_matrix = create_confusion_matrix(annotation_vectors[j], max_f05_annotation_vector)
                _ , _, _, f05 = evaluate_metrics(conf_matrix)
                f05_array.append(f05)
            f05_mean = np.mean(f05_array)
            f05_means.append(f05_mean)
        f05_score = np.mean(f05_means)
        f05_standard_deviation = np.std(f05_means)
        standard_deviations.append(f05_standard_deviation)
        f05_scores.append(f05_score)
    return f05_scores, standard_deviations, sentences_annotation_map


def rater_gold_f05(agreement_dictionary, golden=False):
    """
    Computes the inter-rater agreement vs gold in terms of f05
    Args:
        agreement_dictionary: the annotations used for agreement
    """
    sequence = list(range(settings.RESPONSE_COUNT))
    combinations = []
    for i in range(1, settings.RESPONSE_COUNT + 1):
        combinations.append(list(itertools.combinations(sequence, i)))
    annotation_vectors = []
    # generate an annotation vector for each index
    for i in range(settings.RESPONSE_COUNT + 1):
        annotation_vectors.append(generate_annotation_vector(agreement_dictionary,i))
    f05_scores = [0]
    standard_deviations = [0]
    # combinations
    for i in range(settings.RESPONSE_COUNT):
        f05_array = []
        for combination in combinations[i]:
            max_f05_annotation_vector = []
            sentence_counter = 0
            k_s = []
            sentences = []
            for sentence in sorted(agreement_dictionary.keys()):
                sentence_max = []
                sentence_counter += 1
                max_f05 = 0
                k_max = -1
                sentences.append(sentence)
                for k in combination:
                    conf_matrix = create_confusion_matrix(agreement_dictionary[sentence][9][1],
                                                          agreement_dictionary[sentence][k][1])
                    _, _, _, f05 = evaluate_metrics(conf_matrix)
                    if f05 >= max_f05:
                        max_f05 = f05
                        sentence_max = agreement_dictionary[sentence][k][1]
                        k_max = k
                k_s.append(k_max)
                max_f05_annotation_vector.extend(sentence_max)
            conf_matrix = create_confusion_matrix(annotation_vectors[9], max_f05_annotation_vector)
            _, _, _, f05 = evaluate_metrics(conf_matrix)
            f05_array.append(f05)
        f05_mean = np.mean(f05_array)
        f05_score = f05_mean
        f05_standard_deviation = np.std(f05_array)
        standard_deviations.append(f05_standard_deviation)
        f05_scores.append(f05_score)
    print(f05_scores)
    return f05_scores, standard_deviations


def inter_rater_f05(agreement_dictionary, golden=False):
    """
    Computes the inter-rater agreement in terms of f05
    Args:
        agreement_dictionary: the annotations used for agreement
    """
    add_golden = 0
    if golden:
        add_golden = 1
    sequence = list(range(settings.RESPONSE_COUNT + add_golden))
    combinations = []
    for i in range(1, settings.RESPONSE_COUNT + add_golden):
        combinations.append(list(itertools.combinations(sequence, i)))
    annotation_vectors = []
    # generate an annotation vector for each index
    for i in range(settings.RESPONSE_COUNT + add_golden):
        annotation_vectors.append(generate_annotation_vector(agreement_dictionary,i))
    f05_scores = [0]
    standard_deviations = [0]
    # combinations
    for i in range(settings.RESPONSE_COUNT - 1 + add_golden):
        current_combinations = combinations[i]
        # judgement
        f05_means = []
        for j in range(settings.RESPONSE_COUNT + add_golden):
            combination_index = [combination for combination in current_combinations if j not in combination]
            f05_array = []
            for combination in combination_index:
                max_f05 = 0
                for k in combination:
                    conf_matrix = create_confusion_matrix(annotation_vectors[j], annotation_vectors[k])
                    _ , _, _, f05 = evaluate_metrics(conf_matrix)
                    if f05 > max_f05:
                        max_f05 = f05
                f05_array.append(max_f05)
            f05_mean = np.mean(f05_array)
            f05_means.append(f05_mean)
        f05_score = np.mean(f05_means)
        f05_standard_deviation = np.std(f05_means)
        standard_deviations.append(f05_standard_deviation)
        f05_scores.append(f05_score)
    return f05_scores, standard_deviations


def generate_annotation_vector(agreement_dictionary, k):
    """
    Generates the annotation vector for specific response index
    Args:
        agreement_dictionary: the annotations used to combine the agreement vector
        k: the index
    """
    annotation_vector = []
    for sentence in sorted(agreement_dictionary.keys()):
        annotation_vector.extend(agreement_dictionary[sentence][k][1])
    return annotation_vector


def run_agreement_without_golden(filename):
    """
    Execute this function to get agreement without additional golden annotations.
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    annotations = get_annotations()
    gold_dict = create_gold_dict(gold_data)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    accuracies, standard_deviations = calculate_agreement(agreement_dictionary)
    print(accuracies)
    accuracies[0] = accuracies[1]  # pretty bug
    print(standard_deviations)
    plot_metrics(accuracies, [0.95, settings.RESPONSE_COUNT], [0.6, 0.8], settings.ACCURACY_LABELS,
                 standard_deviations=standard_deviations)
    save_metric_stdev(filename, accuracies, standard_deviations, 'Accuracy')


def run_agreement_without_golden_random(filename):
    """
    Execute this function to get agreement without additional golden annotations.
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    annotations = get_annotations()
    gold_dict = create_gold_dict(gold_data)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    accuracies, standard_deviations = calculate_agreement_random(agreement_dictionary)
    print(accuracies)
    accuracies[0] = accuracies[1]  # pretty bug
    print(standard_deviations)
    plot_metrics(accuracies, [0.95, settings.RESPONSE_COUNT], [0.6, 0.8], settings.ACCURACY_LABELS,
                 standard_deviations=standard_deviations)
    save_metric_stdev(filename, accuracies, standard_deviations, 'Accuracy')


def run_agreement_with_golden_random(filename):
    """
    Execute this function to get agreement without additional golden annotations.
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    accuracies, standard_deviations = calculate_agreement_random(agreement_dictionary)
    print(accuracies)
    accuracies[0] = accuracies[1]  # pretty bug
    print(standard_deviations)
    plot_metrics(accuracies, [0.95, settings.RESPONSE_COUNT], [0.495, 0.6], settings.ACCURACY_LABELS,
                 standard_deviations=standard_deviations)
    save_metric_stdev(filename, accuracies, standard_deviations, 'Accuracy')


def run_agreement_with_golden(filename):
    """
    Execute this function to get agreement with additional golden annotations.
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    accuracies, standard_deviations = calculate_agreement(agreement_dictionary)
    print(accuracies)
    accuracies[0] = accuracies[1]  # pretty bug
    print(standard_deviations)
    plot_metrics(accuracies, [0.95, settings.RESPONSE_COUNT], [0.495, 0.6], settings.ACCURACY_LABELS,
                 standard_deviations=standard_deviations)
    save_metric_stdev(filename, accuracies, standard_deviations, 'Accuracy')


def run_agreement_with_golden_and_shadow(filename):
    """
    Execute this function to get agreement without additional golden and bias (shadow) annotations.
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict, shadow=True)
    accuracies, standard_deviations = calculate_agreement(agreement_dictionary)
    print(accuracies)
    accuracies[0] = accuracies[1]  # pretty bug
    print(standard_deviations)
    plot_metrics(accuracies, [0.95, settings.RESPONSE_COUNT], [0.9, 0.95], settings.ACCURACY_LABELS,
                 standard_deviations=standard_deviations)
    save_metric_stdev(filename, accuracies, standard_deviations, 'Accuracy')


def run_binary_agreement():
    """
    Execute this function to get binary agreement on sentences.
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    agreement_dictionary = create_binary_agreement_dictionary(gold_dict)
    accuracies, standard_deviations = calculate_agreement(agreement_dictionary)
    print(accuracies)
    plot_metrics(accuracies, [1, settings.RESPONSE_COUNT], [0.4, 0.7], settings.ACCURACY_LABELS,
                 standard_deviations=standard_deviations)


def run_agreement_with_golden_incremental():
    """
    Execute this function to get incremental agreement on sentences with golden annotations.
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations()
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    accuracies = calculate_agreement_incremental(agreement_dictionary)
    print(accuracies)
    plt.plot(accuracies)
    plt.xlabel('judgements')
    plt.ylabel('accuracy')
    plt.tight_layout()
    axes = plt.gca()
    axes.set_xlim([1, settings.RESPONSE_COUNT])
    axes.set_ylim([0.6, 0.8])
    plt.show()


def run_create_location_csv():
    """
    Records the location information for turkers in terms of latitude and longitude
    """
    turker_details = extract_information_per_turker(settings.AMT_FILE)
    coordinates = extract_coordinates(turker_details)
    print('number of turker Ids:', len(turker_details))
    with open('locations.csv', 'w+') as location_file:
        fieldnames = ['lat/lon', 'latitude', 'longitude']
        csv_writer = csv.DictWriter(location_file,  lineterminator='\n', fieldnames=fieldnames)
        csv_writer.writeheader()
        for coordinate in coordinates:
            lat_lon = str(coordinate[0]) + ',' + str(coordinate[1])
            row_dict = {
                fieldnames[0]: lat_lon,
                fieldnames[1]: coordinate[0],
                fieldnames[2]: coordinate[1],
            }
            csv_writer.writerow(rowdict=row_dict)


def run_create_country_count_csv():
    """
    Computes and records the country distribution per turker
    """
    turker_details = extract_information_per_turker(settings.AMT_FILE)
    country_count = extract_country_count(turker_details)
    print('number of countries:', len(turker_details))
    with open('country_count.csv', 'w+') as location_file:
        fieldnames = ['Country', 'Count']
        csv_writer = csv.DictWriter(location_file,  lineterminator='\n', fieldnames=fieldnames)
        csv_writer.writeheader()
        for country in country_count.keys():
            row_dict = {
                fieldnames[0]: country,
                fieldnames[1]: country_count[country]
            }
            csv_writer.writerow(rowdict=row_dict)


def run_get_state_counts():
    """
    Produces and Displays the USA state turker counts dictionary
    """
    turker_details = extract_information_per_turker(settings.AMT_FILE)
    state_count = extract_state_count(turker_details)
    print(state_count)
    sorted(state_count.values())
    for state in state_count.keys():
        print('State: ', state, 'Count: ', state_count[state])
    with open('state_count.csv', 'w+') as location_file:
        fieldnames = ['State', 'Count']
        csv_writer = csv.DictWriter(location_file,  lineterminator='\n', fieldnames=fieldnames)
        csv_writer.writeheader()
        for state, value in sorted(state_count.items(), key=lambda x: -x[1]):
            row_dict = {
                fieldnames[0]: state,
                fieldnames[1]: state_count[state]
            }
            csv_writer.writerow(rowdict=row_dict)


def run_agreement_per_turker():
    """
    Execute this to get plot of turker agreement related to completed HITs
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    annotations = get_annotations()
    gold_dict = create_gold_dict(gold_data)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    turker_accuracies = accuracy_per_turker(agreement_dictionary)
    values_list = [value[0] for value in turker_accuracies.values()]
    plt.scatter(*zip(*values_list), marker='P')
    plt.xlabel('Number of HITs completed')
    plt.ylabel('Accuracy with expert')
    plt.grid()
    plt.show()
    max_number_of_HITs = max([y[0][0] for y in turker_accuracies.values()])
    accuracy_for_count = []
    for i in range(1, 7):
        accuracies = [y[0][1] for y in turker_accuracies.values() if (i - 1) * 10 < y[0][0] <= i * 10]
        if len(accuracies) > 0:
            accuracy_for_count.append(np.mean(accuracies))
    weight = sum(accuracy_for_count)
    uniform_arr = []
    for i in range(0, len(accuracy_for_count)):
        uniform_arr.append(accuracy_for_count[i] / weight)
    for i in range(len(uniform_arr)):
        plt.axhline(y=uniform_arr[i], xmin=i * 10/60, xmax=((i + 1) * 10)/60)
    axes = plt.gca()
    axes.set_xlim([0, 60])
    axes.set_ylim([0.0, 1.0])
    print(uniform_arr)
    plt.xlabel('Number of HITs completed')
    plt.ylabel('Accuracy weight')
    plt.grid()
    plt.show()
    #plot_metrics(uniform_arr,[0, len(uniform_arr)], [0,0.20], labels={'xlabel': 'number of HITS', 'ylabel': 'uniform distribution'})

def get_guessed_errors_counts():
    """
    Gets  dictionary with the counts of guessed errors types
    Returns:
        guessed_error_dict: error_type->count
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    annotations = get_annotations()
    guessed_error_dict = defaultdict(int)
    for sentence in gold_data:
        annotations_sentence = list(set(annotations[sentence[0][1:]]))
        if len(sentence[1]) < 1:
            guessed_error_dict['no error'] += len([x for x in annotations_sentence if x == -2])
        for span in sentence[1]:
            for annotation in annotations_sentence:
                if span[0] <= int(annotation) < int(span[1]):
                    guessed_error_dict[span[2]] += 1
                    break
    return guessed_error_dict


def run_create_error_count_csv():
    error_dict, count_error_dict = fd.count_error_types(settings.AMT_FCE_M2)
    count_tuples = count_error_dict.items()
    count_tuples = sorted(count_tuples, key=lambda x: -x[1])
    guessed_errors_counts = get_guessed_errors_counts()
    with open('error_count.csv', 'w+') as location_file:
        fieldnames = ['Error Type', 'Count', 'Guessed']
        csv_writer = csv.DictWriter(location_file,  lineterminator='\n', fieldnames=fieldnames)
        csv_writer.writeheader()
        for count_tuple in count_tuples:
            csv_writer.writerow({
                fieldnames[0]: count_tuple[0],
                fieldnames[1]: count_tuple[1],
                fieldnames[2]: guessed_errors_counts[count_tuple[0]]
            })


def run_best_pseudo_turker():
    """
    Run the function to get the accuracy of the best pseudo turker
    """
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    annotations = get_annotations()
    gold_dict = create_gold_dict(gold_data)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    individual_accuracies = accuracy_per_turker(agreement_dictionary)
    accuracy, turker_set, best_agreement_vector, gold_agreement_vector = agreement_minimum_hamming(agreement_dictionary,
                                                                                                   individual_accuracies)
    print(accuracy)
    for key in turker_set.keys():
        print(key, ' : (accuracy: ', turker_set[key][0], ', count: ', turker_set[key][1], ')')
    print('Number of turkers: ', len(turker_set))
    counts = sum([turker_set[key][1] for key in turker_set.keys()])
    print('Counts: ', counts)
    average_count = sum([turker_set[key][1] for key in turker_set.keys()]) / len(turker_set)
    print('Average Count: ', average_count)
    average_accuracy = sum([turker_set[key][0] for key in turker_set.keys()]) / len(turker_set)
    print('Average Accuracy: ', average_accuracy)
    cm = create_confusion_matrix(gold_agreement_vector, best_agreement_vector)
    plot_confusion_matrix_dict(cm)
    precision, recall, f1, f05 = evaluate_metrics(cm)
    print('F05 : ', f05)


def calculate_agreement_stv(agreement_dictionary, turker_accuracies):
    """
    Inter agreement with most accurate chair vote
    Args:
         agreement_dictionary: holding sentence annotation records - 9 from non-experts and 1 expert
         sentence -> list of annotations (size settings.RESPONSE_COUNT + 1)
         turker_accuracies: accuracy for each turker used for the chair vote
    Returns:
        The accuracies from combined agreement from one to nine non-experts with the expert
    """
    sequence = list(range(settings.RESPONSE_COUNT))
    combinations = []
    for i in range(settings.RESPONSE_COUNT + 1):
        combinations.append(list(itertools.combinations(sequence, i)))
    print(combinations)
    accuracies = [0]
    standard_deviations = [0]
    for i in range(1, settings.RESPONSE_COUNT + 1):
        current_combinations = combinations[i]
        combination_accuracies = []
        for combination in current_combinations:
            correct = 0
            incorrect = 0
            for sentence in agreement_dictionary.keys():
                expert_annotations = agreement_dictionary[sentence][-1][1]
                chosen_annotations = [agreement_dictionary[sentence][x][1] for x in combination]
                votes = np.sum(chosen_annotations, axis=0)
                chair = 0
                if len(combination) > 0 and len(combination) % 2 == 0:
                    max_accuracy = 0
                    for judgement_index in combination:
                        turker = agreement_dictionary[sentence][judgement_index][0]
                        turker_accuracy = turker_accuracies[turker][0][1]
                        if turker_accuracy > max_accuracy:
                            max_accuracy = turker_accuracy
                            chair = judgement_index
                result_votes = [0] * len(votes)
                for j in range(len(votes)):
                    if votes[j] < len(chosen_annotations) / 2:
                        result_votes[j] = 0
                    elif votes[j] > len(chosen_annotations) / 2:
                        result_votes[j] = 1
                    else:
                        result_votes[j] = agreement_dictionary[sentence][chair][1][j]
                for j in range(len(votes)):
                    if expert_annotations[j] == result_votes[j]:
                        correct += 1
                    else:
                        incorrect += 1
            combination_accuracy = correct / (correct + incorrect)
            combination_accuracies.append(combination_accuracy)
        standard_deviation = np.std(combination_accuracies)
        standard_deviations.append(standard_deviation)
        accuracy = sum(combination_accuracies) / len(combination_accuracies)
        accuracies.append(accuracy)
    return accuracies, standard_deviations


def calculate_agreement_weighting_votes(agreement_dictionary, turker_accuracies):
    """
    Inter agreement with most accurate chair vote
    Args:
         agreement_dictionary: holding sentence annotation records - 9 from non-experts and 1 expert
         sentence -> list of annotations (size settings.RESPONSE_COUNT + 1)
         turker_accuracies: accuracy for each turker used for the chair vote
    Returns:
        The accuracies from combined agreement from one to nine non-experts with the expert
    """
    sequence = list(range(settings.RESPONSE_COUNT))
    combinations = []
    for i in range(settings.RESPONSE_COUNT + 1):
        combinations.append(list(itertools.combinations(sequence, i)))
    print(combinations)
    accuracies = [0]
    standard_deviations = [0]
    for i in range(1, settings.RESPONSE_COUNT + 1):
        current_combinations = combinations[i]
        combination_accuracies = []
        for combination in current_combinations:
            correct = 0
            incorrect = 0
            for sentence in agreement_dictionary.keys():
                expert_annotations = agreement_dictionary[sentence][-1][1]
                chosen_annotations = [agreement_dictionary[sentence][x][1] for x in combination]
                chair = 0
                max_accuracy = 0
                for judgement_index in combination:
                    turker = agreement_dictionary[sentence][judgement_index][0]
                    turker_accuracy = turker_accuracies[turker][0][1]
                    if turker_accuracy > max_accuracy:
                        max_accuracy = turker_accuracy
                        chair = judgement_index
                result_votes = [0] * len(chosen_annotations[0])
                for j in range(len(chosen_annotations[0])):
                    for judgement_index in combination:
                        weighting_pairs = {0: 0, 1: 0}
                        turker = agreement_dictionary[sentence][judgement_index][0]
                        turker_accuracy = turker_accuracies[turker][0][1]
                        weighting_pairs[agreement_dictionary[sentence][judgement_index][1][j]] += turker_accuracy
                    if weighting_pairs[0] < weighting_pairs[1]:
                        result_votes[j] = 1
                    else:
                        result_votes[j] = agreement_dictionary[sentence][chair][1][j]
                for j in range(len(chosen_annotations[0])):
                    if expert_annotations[j] == result_votes[j]:
                        correct += 1
                    else:
                        incorrect += 1
            combination_accuracy = correct / (correct + incorrect)
            combination_accuracies.append(combination_accuracy)
        standard_deviation = np.std(combination_accuracies)
        standard_deviations.append(standard_deviation)
        accuracy = sum(combination_accuracies) / len(combination_accuracies)
        accuracies.append(accuracy)
    return accuracies, standard_deviations


def run_agreement_stv(filename):
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    turker_accuracies = accuracy_per_turker(agreement_dictionary)
    accuracies, standard_deviations = calculate_agreement_stv(agreement_dictionary, turker_accuracies)
    accuracies[0] = accuracies[1]
    print(accuracies)
    print(standard_deviations)
    plot_metrics(accuracies, [0.95, settings.RESPONSE_COUNT], [0.495, 0.6], settings.ACCURACY_LABELS, standard_deviations=standard_deviations)
    save_metric_stdev(filename, accuracies, standard_deviations, 'Accuracy')


def run_agreement_sum(filename):
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    accuracies, standard_deviations = calculate_agreement_sum(agreement_dictionary)
    accuracies[0] = accuracies[1]
    print(accuracies)
    plot_metrics(accuracies, [0.95, settings.RESPONSE_COUNT], [0.1, 0.7], settings.ACCURACY_LABELS,
                 standard_deviations=standard_deviations)
    save_metric_stdev(filename, accuracies, standard_deviations, 'Accuracy')


def run_agreement_weighted_votes(filenames, annotation_limit=sys.maxsize):
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    turker_accuracies_limit = accuracy_per_turker(agreement_dictionary, annotation_limit=annotation_limit)
    turker_accuracies_limit_2 = accuracy_per_turker(agreement_dictionary, annotation_limit=annotation_limit)

    turker_accuracies = accuracy_per_turker(agreement_dictionary, annotation_limit=sys.maxsize)
    accuracies, standard_deviations = calculate_agreement_weighting_votes(agreement_dictionary, turker_accuracies)
    accuracies_limit, standard_deviations_limit = calculate_agreement_weighting_votes(agreement_dictionary, turker_accuracies_limit)
    accuracies[0] = accuracies[1]
    accuracies_limit[0] = accuracies_limit[1]
    accuracies_set = [accuracies, accuracies_limit]
    standard_deviations_set = [standard_deviations, standard_deviations_limit]
    plot_multiple_accuracies(accuracies_set, [0.95, settings.RESPONSE_COUNT + 0.05], [0.495, 0.60],
                             legend_labels=['overall accuracy', 'accuracy over first 10 annotations'],
                             standard_deviations=standard_deviations_set, loc='upper right')
    # plot_metrics(accuracies, [0.95, settings.RESPONSE_COUNT], [0.495, 0.60], settings.ACCURACY_LABELS,
    #              standard_deviations=standard_deviations)
    for i in range(len(filenames)):
        save_metric_stdev(filenames[i], accuracies_set[i], standard_deviations_set[i], 'Accuracy')


def save_metric_stdev(filename, accuracies, stdev, metric_type):
    with open(filename, 'w+') as location_file:
        fieldnames = ['Judgements', metric_type, 'SD']
        csv_writer = csv.DictWriter(location_file,  lineterminator='\n', fieldnames=fieldnames)
        csv_writer.writeheader()
        for i in range(1, len(accuracies)):
            csv_writer.writerow({
                fieldnames[0]: i,
                fieldnames[1]: accuracies[i],
                fieldnames[2]: stdev[i]
            })


def calculate_fleiss_kappa(agreement_dictionary):
    """
       Calculates the Fleiss Kappa over annotations
       Args:
           agreement_dictionary: the dictionary with the annotations used in agreement
       Return:
           Fleiss kappa of the annotations
    """
    annotation_count = 1
    fleiss_input = []
    for key in agreement_dictionary.keys():
        for i in range(len(agreement_dictionary[key][0][1])):
            for j in range(len(agreement_dictionary[key]) - 1):
                fleiss_input.append((annotation_count, agreement_dictionary[key][j][1][i]))
            annotation_count += 1
    kappa = fleiss_kappa(fleiss_input, 9)
    return kappa


def run_fleiss_kappa():
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    f_kappa = calculate_fleiss_kappa(agreement_dictionary)
    print('Fleiss Kappa: ', f_kappa)


def run_test_with_count_tries():
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    turker_accuracies = accuracy_per_turker(agreement_dictionary)
    information_dict = extract_information_per_turker(settings.AMT_FILE)
    tries_vs_accuracies = []
    for key in turker_accuracies.keys():
        turker_accuracy = turker_accuracies[key][0][1]
        turker_tries = information_dict[key]['triesPerTask']
        try_vs_accuracy = (turker_accuracy, turker_tries)
        tries_vs_accuracies.append(try_vs_accuracy)
    tries_vs_accuracies.sort(key=lambda x: -x[0])
    cumulative_tries = 0
    for i in range(len(tries_vs_accuracies)):
        cumulative_tries += tries_vs_accuracies[i][1]
        tries_vs_accuracies[i] = (i + 1, tries_vs_accuracies[i][1])
    rank, tries = zip(*tries_vs_accuracies)
    plt.scatter(rank, tries)
    plt.show()


def run_stacked_plots():
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    turker_accuracies = accuracy_per_turker(agreement_dictionary)
    accuracies_set = [[x] for x in range(5)]
    standard_deviations_set = [[x] for x in range(5)]
    accuracies_set[0], standard_deviations_set[0] = calculate_agreement_sum(agreement_dictionary)
    accuracies_set[1], standard_deviations_set[1] = calculate_agreement_stv(agreement_dictionary, turker_accuracies)
    accuracies_set[2], standard_deviations_set[2] = calculate_agreement_random(agreement_dictionary)
    accuracies_set[3], standard_deviations_set[3] = calculate_agreement(agreement_dictionary)
    accuracies_set[4], standard_deviations_set[4] = calculate_agreement_weighting_votes(agreement_dictionary,
                                                                                        turker_accuracies)
    for i in range(len(accuracies_set)):
        print('Accuracies ', i, ' : ', accuracies_set[i])
    for i in range(len(accuracies_set)):
        accuracies_set[i][0] = accuracies_set[i][1]
    plot_labels = ['sum of judgements', 'most accurate tie-breaking', 'random tie-breaking',
                   'majority vote (Snow et al.)', 'weighted judgements']
    plot_multiple_accuracies(accuracies_set, [1, settings.RESPONSE_COUNT], [0.2,  0.8],
                             legend_labels=plot_labels, standard_deviations=standard_deviations_set)


def run_stacked_plots_different_annotations():
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    non_expert_judgements = get_annotations()
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(non_expert_judgements, gold_dict)
    agreement_dictionary_with_golden, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    agreement_dictionary_with_shadow, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict, shadow=True)
    accuracies_set = [[x] for x in range(3)]
    standard_deviations_set = [[x] for x in range(3)]
    accuracies_set[0], standard_deviations_set[0] = calculate_agreement(agreement_dictionary)
    accuracies_set[1], standard_deviations_set[1] = calculate_agreement(agreement_dictionary_with_golden)
    accuracies_set[2], standard_deviations_set[2] = calculate_agreement(agreement_dictionary_with_shadow)

    for i in range(len(accuracies_set)):
        print('Accuracies ', i, ' : ', accuracies_set[i])
    for i in range(len(accuracies_set)):
        accuracies_set[i][0] = accuracies_set[i][1]
    dir_prefix = 'agreement_methods/accuracy_stdevs_'
    file_names = [dir_prefix + '_explicit', dir_prefix + '_golden', dir_prefix + '_all_tokens']
    plot_labels = ['Method 1: selected tokens', 'Method 2: selected and gold tokens', 'Method 3: all sentence tokens']
    plot_multiple_accuracies(accuracies_set, [1, settings.RESPONSE_COUNT], [0.2,  1.0],
                             legend_labels=plot_labels, standard_deviations=standard_deviations_set, loc='lower right')
    for i in range(len(file_names)):
        save_metric_stdev(file_names[i], accuracies_set[i], standard_deviations_set[i], 'Accuracy')


def run_compute_inter_annotator_agreement_f05():
    gold_data = fd.extract_data(settings.AMT_FCE_M2)
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    scores_arr = []
    stdev_arr = []
    f05_scores, f05_standard_deviations, sentences_annotation_map = inter_rater_f05_sentence(agreement_dictionary)
    f05_scores_golden, f05_standard_deviations_golden, sentences_annotation_map = inter_rater_f05_sentence(agreement_dictionary, golden=True)
    f05_scores_gold_only, f05_standard_deviations_gold_only = rater_gold_f05(agreement_dictionary, golden=True)
    scores_arr.append(f05_scores)
    scores_arr.append(f05_scores_golden)
    scores_arr.append(f05_scores_gold_only)
    stdev_arr.append(f05_standard_deviations)
    stdev_arr.append(f05_standard_deviations_golden)
    stdev_arr.append(f05_standard_deviations_gold_only)
    print(f05_scores)
    f05_scores[0] = f05_scores[1]
    f05_scores_golden[0] = f05_scores_golden[1]
    f05_scores_gold_only[0] = f05_scores_gold_only[1]
    plot_multiple_accuracies(scores_arr, [0.95, 9.1], [0.49, 1.00], ['non-expert', 'non-expert and expert',
                                                                    'expert vs non-expert'],
                             {'xlabel': 'judgements', 'ylabel': 'F0.5'}, standard_deviations=stdev_arr)
    save_metric_stdev('agreement_methods/f05_full_inter_rater_agreement_2.csv', f05_scores, f05_standard_deviations,
                      'F0.5')
    save_metric_stdev('agreement_methods/f05_full_inter_rater_agreement_gold_2.csv', f05_scores_golden,
                      f05_standard_deviations_golden, 'F0.5')
    save_metric_stdev('agreement_methods/f05_full_inter_rater_agreement_gold_only.csv', f05_scores_gold_only,
                      f05_standard_deviations_gold_only, 'F0.5')


if __name__ == '__main__':
    # run_agreement_without_golden('agreement_methods/accuracy_stdevs_without_golden_snow_et_al_sorted.csv')
    #run_agreement_with_golden('agreement_methods/accuracy_stdevs_with_golden_snow_et_al.csv')
    # run_agreement_with_golden_and_shadow('agreement_methods/accuracy_stdevs_with_golden_and_shadow_snow_et_al.csv')
    # run_agreement_stv('agreement_methods/accuracy_stdevs_stv.csv')
    # run_agreement_sum('agreement_methods/accuracy_stdevs_sum.csv')
    # run_best_pseudo_turker()
    # run_stacked_plots()
    # run_agreement_per_turker()
    # run_agreement_weighted_votes(['accuracy_stdevs_weighting_votes.csv',
    # 'accuracy_stdevs_weighting_votes_limit_10.csv'], annotation_limit=10)
    # run_get_state_counts()
    # worker_information = extract_information_per_turker(settings.AMT_FILE)
    # worker_information = test_results_information(settings.AMT_SENTENCE_BATCH)
    # response_time_averages = [worker_information[worker_id]['averageResponseTime'] for worker_id in worker_information.keys()
    #                           if worker_information[worker_id]['averageResponseTime'] < 500]
    # print('Response Time Mean: ', np.mean(response_time_averages))
    # print('Response Time Standard Deviation', np.std(response_time_averages))
    # print('Max response time: ', np.max(response_time_averages))
    # print('Min response time: ', np.min(response_time_averages))
    # run_compute_inter_annotator_agreement_f05()
    # run_stacked_plots_different_annotations()
    # gold_data = fd.extract_data(settings.AMT_FCE_M2)
    #run_best_pseudo_turker()
    #run_binary_agreement()
    # gold_data = fd.extract_data(settings.AMT_FCE_M2)
    # gold_dict = create_gold_dict(gold_data)
    # annotations = get_annotations(golden=gold_dict)
    # agreement_dictionary, annotations_for_sentence = create_agreement_dictionary(annotations, gold_dict)
    # scores_arr = []
    # stdev_arr = []
    # f05_scores, f05_standard_deviations, sentences_annotation_map = inter_rater_f05_sentence(agreement_dictionary)
    # pairs = []
    # for sentence in annotations_for_sentence.keys():
    #     pairs.append((sentence, annotations_for_sentence[sentence], sentences_annotation_map[sentence]))
    # convert_m2.extract_to_m2('fce_amt_experiment_best_result.m2', pairs)
    run_agreement_per_turker()