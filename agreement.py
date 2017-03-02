from collections import defaultdict
import csv

import re

import fce_api as fd
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import settings


def full_evaluation_table(confusion_matrix, classes=list()):
    """
    Produce a pandas data-frame with Precision, F1 and Recall for all labels.
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


# if the number of annotations is at least number judgements
def test_annotation_dict(annot_dict, judgements):
    for key in annot_dict.keys():
        if len(annot_dict[key]) < judgements:
            return False
    return True


# create the confusion matrix
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


# plot the confusion matrix
def plot_confusion_matrix_dict(matrix_dict, classes=[], rotation=45, outside_label=''):
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
    if (len(classes) != len(sorted_labels)):
        classes = sorted_labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=rotation)
    plt.yticks(tick_marks, classes)
    plt.xlabel('predicted labels')
    plt.ylabel('gold labels')
    plt.tight_layout()
    plt.show()


# extracts the amt_sentences along with their errors
def extract_sentences_with_errors():
    # open source file
    with open(settings.TRAINING_DATA_FILE, 'r') as file:
        # read the lines
        readlines = file.readlines()
        with open('amt_sentence_batch.csv', 'r') as amt_batch:
            csv_reader = csv.DictReader(amt_batch)
            with open('fce_amt.experiment_two.max.rasp.m2', 'w+') as destination:
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


# get annotations from turkers
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
                if int(label) >= error_span[0] and int(label) < error_span[1]:
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
                counted = 1
    return gold, predicted


def create_agreement_dictionary(annotations, gold_labels, shadow=False):
    """
    Produces the agreement dictionary used for inter-rater agreement and accuracy scores
    Args:
        annotations: the annotations for each sentence
        gold_labels: the golden annotations
        shadow: apply no labeled annotations
    Returns:
        agreement_dictionary: a dictionary for inter-rater agreement
    """
    # turker ID -> annotations
    agreement_dictionary = defaultdict(list)
    with open(settings.AMT_FILE) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            sentence = row['Input.sentence']
            workerId = row['WorkerId']
            answer = json.loads(row['Answer.ChosenWord'])
            worker_annotations = [x['start'] for x in answer['selectedTokens']]
            annotations_for_sentence = list(set(annotations[sentence]))
            tokens = re.split(r'(\s+)', sentence)
            shadow_annotations = []
            if shadow:
                shadow_annotations = [0] * (len(tokens) + 2 - len(annotations_for_sentence))
            explicit_annotations = [1 if x in worker_annotations else 0 for x in annotations_for_sentence]
            agreement_dictionary[sentence].append((workerId, explicit_annotations + shadow_annotations))
    for key in agreement_dictionary.keys():
        annotations_for_sentence = list(set(annotations[key]))
        tokens = re.split(r'(\s+)', key)
        shadow_annotations = []
        if shadow:
            shadow_annotations = [0] * (len(tokens) + 2 - len(annotations_for_sentence))
        explicit_annotations = [1 if x in gold_labels[key] else 0 for x in annotations_for_sentence]
        agreement_dictionary[key].append(('expert', explicit_annotations + shadow_annotations))
    return agreement_dictionary


def create_binary_agreement_dictionary(gold_labels):
    """
    Produces the binary agreement dictionary used for inter-rater agreement and accuracy scores
    Binary puts only 1 annotation per sentence - without an error or with error
    Args:
        annotations: the annotations for each sentence
        gold_labels: the golden annotations
        shadow: apply no labeled annotations
    Returns:
        agreement_dictionary: a dictionary for inter-rater agreement
    """
    # turker ID -> annotations
    agreement_dictionary = defaultdict(list)
    with open(settings.AMT_FILE) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            sentence = row['Input.sentence']
            workerId = row['WorkerId']
            answer = json.loads(row['Answer.ChosenWord'])
            if answer['selectedTokens'][0]['start'] == -2:
                annotation = 0
            else:
                annotation = 1
            agreement_dictionary[sentence].append((workerId, [annotation]))
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
                start = span[0]
                if span[0] == span[1]:
                    start = str(span[0])
                gold_dict[sentence[1:]].append(span[0])
                for i in range(span[0] + 1, span[1]):
                    gold_dict[sentence[1:]].append(i)
    return gold_dict


def calculate_agreement(agreement_dictionary):
    """
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
                if (len(combination) % 2 == 0):
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
                    # chair = random.choice([x for x in list(combination)])
                    # chair =
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


def extract_information_per_turker():
    """
    Extracts dictionary with turker's details
    Returns:
        user_information: dictionary with user details
    """
    user_information = {}
    with open(settings.AMT_FILE) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            workerId = row['WorkerId']
            answer = json.loads(row['Answer.ClientLocation'])
            if workerId not in user_information:
                if 'backUpLocation' in answer:
                    user_information[workerId] = answer['backUpLocation']
                elif 'latitude' in answer and 'longitude' in answer:
                    user_information[workerId] = {'latitude': answer['latitude'], 'longitude': answer['longitude']}
                else:
                    user_information[workerId] = {'latitude': 'unknown', 'longitude': 'unknown'}

    return user_information


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
        user_information: dict with the user information of the turkers (returned from @extract_information_per_turker
    Returns:
        locations: latitude, longitude) pairs
    """
    country_counts = defaultdict(int)
    for turkerId in user_information.keys():
        if 'country' in user_information[turkerId]:
            country = user_information[turkerId]['country']
            country_counts[country] += 1
        else:
            print(user_information[turkerId]['latitude'],' , ', user_information[turkerId]['longitude'])
    return country_counts


def agreement_minimum_hamming(agreement_dictionary):
    correct = 0
    total = 0
    for sentence in agreement_dictionary.keys():
        expert_annotations = agreement_dictionary[sentence][-1][1]
        min_distance = len(expert_annotations) + 1
        for i in range(settings.RESPONSE_COUNT):
            distance = hamming(expert_annotations, agreement_dictionary[sentence][i][1])
            if distance < min_distance:
                min_distance = distance
        correct += len(expert_annotations) - min_distance
        total += len(expert_annotations)
    accuracy = correct / total
    return accuracy


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
        if (correct[i] != observed[i]):
            distance += 1
    return distance


def accuracy_per_turker(agreement_dictionary):
    """
    Calculates accuracy for each participated turker - unique workId
    Args:

    Returns:

    """
    turker_accuracies = defaultdict(list)
    for sentence in agreement_dictionary:
        expert_annotations = agreement_dictionary[sentence][-1][1]
        for i in range(settings.RESPONSE_COUNT):
            worker_id = agreement_dictionary[sentence][i][0]
            if len(turker_accuracies[worker_id]) == 0:
                turker_accuracies[worker_id].append((0, 0))
                turker_accuracies[worker_id].append(0)
            guesses = agreement_dictionary[sentence][i][1]
            for j in range(len(guesses)):
                if guesses[j] == 1:
                    if guesses[j] == expert_annotations[j]:
                        turker_accuracies[worker_id][0] = (turker_accuracies[worker_id][0][0] + 1, turker_accuracies[worker_id][0][1])
                    turker_accuracies[worker_id][0] = (turker_accuracies[worker_id][0][0], turker_accuracies[worker_id][0][1] + 1)
            turker_accuracies[worker_id][1] += 1

    for worker_id in turker_accuracies.keys():
        turker_accuracies[worker_id][0] = turker_accuracies[worker_id][0][0] / turker_accuracies[worker_id][0][1]
        turker_accuracies[worker_id] = [(turker_accuracies[worker_id][1], turker_accuracies[worker_id][0])]
    return turker_accuracies


def run_one_display_confusion():
    """
    Execute this function to display the confusion matrix and the precision recall table.

    """
    gold_data = fd.extract_data('fce_amt.experiment_two.max.rasp.m2')
    annotations = get_annotations()
    gold, predicted = compare_annotations(gold_data, annotations)
    cm = create_confusion_matrix(gold, predicted)
    plot_confusion_matrix_dict(cm, classes=['Error', 'No Error'])
    precision_and_recall = full_evaluation_table(cm)


def plot_accuracies(accuracies, standard_deviations, x_scope, y_scope):
    """
    Plots the accuracies with their standard deviation
    Args:
        accuracies: agreement accuracies
        standard_deviations:  the standard deviations corresponding to the accuracies
        x_scope: the visible part of the x axis
        y_scope: the visible part of the y axis
    """
    plt.plot(accuracies)
    x_s = list(range(settings.RESPONSE_COUNT + 1))
    plt.errorbar(x_s, accuracies, standard_deviations, linestyle='None', marker='^')
    plt.xlabel('judgements')
    plt.ylabel('accuracy')
    plt.tight_layout()
    axes = plt.gca()
    axes.set_xlim(x_scope)
    axes.set_ylim(y_scope)
    plt.show()


def run_two_agreement_without_golden():
    gold_data = fd.extract_data('fce_amt.experiment_two.max.rasp.m2')
    annotations = get_annotations()
    gold_dict = create_gold_dict(gold_data)
    agreement_dictionary = create_agreement_dictionary(annotations, gold_dict)
    accuracies, standard_deviations = calculate_agreement(agreement_dictionary)
    print(accuracies)
    print(standard_deviations)
    plot_accuracies(accuracies, standard_deviations, [1, settings.RESPONSE_COUNT], [0.6, 0.8])


def run_three_agreement_with_golden():
    gold_data = fd.extract_data('fce_amt.experiment_two.max.rasp.m2')
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary = create_agreement_dictionary(annotations, gold_dict)
    accuracies, standard_deviations = calculate_agreement(agreement_dictionary)
    print(accuracies)
    print(standard_deviations)
    plot_accuracies(accuracies, standard_deviations, [1, settings.RESPONSE_COUNT], [0.4, 0.7])


def run_four_binary_agreement():
    gold_data = fd.extract_data('fce_amt.experiment_two.max.rasp.m2')
    gold_dict = create_gold_dict(gold_data)
    agreement_dictionary = create_binary_agreement_dictionary(gold_dict)
    accuracies, standard_deviations = calculate_agreement(agreement_dictionary)
    print(accuracies)
    plot_accuracies(accuracies, standard_deviations, [1, settings.RESPONSE_COUNT], [0.4, 0.7])


def run_three_agreement_with_golden_and_shadow():
    gold_data = fd.extract_data('fce_amt.experiment_two.max.rasp.m2')
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary = create_agreement_dictionary(annotations, gold_dict, shadow=True)
    accuracies, standard_deviations = calculate_agreement(agreement_dictionary)
    print(accuracies)
    print(standard_deviations)
    plot_accuracies(accuracies, standard_deviations, [1, settings.RESPONSE_COUNT],[0.9, 0.95])


def run_three_agreement_with_golden_incremental():
    gold_data = fd.extract_data('fce_amt.experiment_two.max.rasp.m2')
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations()
    agreement_dictionary = create_agreement_dictionary(annotations, gold_dict)
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
    turker_details = extract_information_per_turker()
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
    turker_details = extract_information_per_turker()
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


def run_agreement_per_turker():
    gold_data = fd.extract_data('fce_amt.experiment_two.max.rasp.m2')
    annotations = get_annotations()
    gold_dict = create_gold_dict(gold_data)
    agreement_dictionary = create_agreement_dictionary(annotations, gold_dict)
    #accuracy = agreement_minimum_hamming(agreement_dictionary)
    accuracies = accuracy_per_turker(agreement_dictionary)
    print(accuracies)
    high = [(x, y) for x, y in accuracies.items() if y[0][1] > 0.98 ]
    print(high)
    print((len(high)))
    low = [(x, y) for x, y in accuracies.items() if y[0][1] < 0.01 ]
    print(low)
    print((len(low)))
    values_list = [value[0] for value in accuracies.values()]
    plt.scatter(*zip(*values_list), marker='P')
    plt.title('Agreement of individual Turkers v # HITS')
    plt.xlabel('Number of HITs completed')
    plt.ylabel('Accuracy with expert')
    plt.grid()
    plt.show()


def get_guessed_errors_counts():
    gold_data = fd.extract_data('fce_amt.experiment_two.max.rasp.m2')
    annotations = get_annotations()
    gold_dict = create_gold_dict(gold_data)
    guessed_error_dict = defaultdict(int)
    for sentence in gold_data:
        annotations_sentence = list(set(annotations[sentence[0][1:]]))
        if (len(sentence[1]) < 1):
            guessed_error_dict['no error'] += len([x for x in annotations_sentence if x == -2])
        for span in sentence[1]:
           for annotation in annotations_sentence:
                if span[0] <= int(annotation) < int(span[1]):
                    guessed_error_dict[span[2]] += 1
                    break
    return guessed_error_dict


def run_create_error_count_csv():
    error_dict, count_error_dict = fd.count_error_types('fce_amt.experiment_two.max.rasp.m2')
    count_tuples = count_error_dict.items()
    count_tuples = sorted(count_tuples, key=lambda x: -x[1])
    guessed_errors_counts = get_guessed_errors_counts()
    with open('error_count.csv', 'w+') as location_file:
        fieldnames = ['Error Type', 'Count', 'Guessed']
        csv_writer = csv.DictWriter(location_file,  lineterminator='\n', fieldnames=fieldnames)
        csv_writer.writeheader()
        for tuple in count_tuples:
            csv_writer.writerow({
                fieldnames[0]: tuple[0],
                fieldnames[1]: tuple[1],
                fieldnames[2]: guessed_errors_counts[tuple[0]]
            })


def run_best_pseudo_turker():
    gold_data = fd.extract_data('fce_amt.experiment_two.max.rasp.m2')
    annotations = get_annotations()
    gold_dict = create_gold_dict(gold_data)
    agreement_dictionary = create_agreement_dictionary(annotations, gold_dict)
    accuracy = agreement_minimum_hamming(agreement_dictionary)
    print(accuracy)


def calculate_agreement_in_progress(agreement_dictionary):
    """
    Calculates the agreement by incrementally adding additional turker rather than taking all the combinations of
    turkers
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
                    chair = combination[-1]
                    # chair = random.choice([x for x in list(combination)])
                    # chair =
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


def run_three_agreement_develop_new():
    gold_data = fd.extract_data('fce_amt.experiment_two.max.rasp.m2')
    gold_dict = create_gold_dict(gold_data)
    annotations = get_annotations(golden=gold_dict)
    agreement_dictionary = create_agreement_dictionary(annotations, gold_dict)
    accuracies, standard_deviations = calculate_agreement_in_progress(agreement_dictionary)
    print(accuracies)
    print(standard_deviations)
    plot_accuracies(accuracies, standard_deviations, [0.95, settings.RESPONSE_COUNT], [0.4, 0.7])


if __name__ == '__main__':
    run_two_agreement_without_golden()
    #run_three_agreement_with_golden()
    #run_three_agreement_with_golden_and_shadow()