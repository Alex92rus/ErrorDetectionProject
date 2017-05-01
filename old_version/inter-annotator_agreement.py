import csv
import json
from collections import defaultdict
from collections import Counter
import fce_api as fd
import pandas
import numpy as np

DEBUG = False

""" Computes the Fleiss' Kappa value as described in (Fleiss, 1971) """
def compute_kappa(mat):
    """ Computes the Kappa value
        @param n Number of rating per subjects (number of human raters)
        @param mat Matrix[subjects][categories]
        @return The Kappa value """
    n = check_each_line_count(mat)  # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])

    if DEBUG:
        print(n, 'raters.')
        print(N, 'subjects.')
        print(k, 'categories.')

    # Computing p[]
    p = [0.0] * k
    for j in range(k):
        p[j] = 0.0
        for i in range(N):
            p[j] += mat[i][j]
        p[j] /= N * n
    if DEBUG: print('p =', p)

    # Computing P[]
    P = [0.0] * N
    for i in range(N):
        P[i] = 0.0
        for j in range(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))
    if DEBUG: print('P =', P)

    # Computing Pbar
    Pbar = sum(P) / N
    if DEBUG:
        print('Pbar =', Pbar)
        print('Sum P =', sum(P))

    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj
    if DEBUG: print('PbarE =', PbarE)

    kappa = (Pbar - PbarE) / (1 - PbarE)
    if DEBUG: print('kappa =', kappa)

    return kappa


def compute_free_kappa(mat):
    """ Computes the Kappa value
        @param n Number of rating per subjects (number of human raters)
        @param mat Matrix[subjects][categories]
        @return The Kappa value """
    n = check_each_line_count(mat)  # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])

    if DEBUG:
        print(n, 'raters.')
        print(N, 'subjects.')
        print(k, 'categories.')

    # Computing Po
    sum_of_squares = 0
    for j in range(k):
        for i in range(N):
            sum_of_squares += mat[i][j] * mat[i][j]
    p_o = ((sum_of_squares - N * n) / ((N * n) * (n - 1)))
    if DEBUG: print('P_o =', p_o)

    # Computing P[]
    p_e = 1.0 / k

    kappa = (p_o - p_e) / (1.0 - p_e)
    if DEBUG: print('kappa =', kappa)

    return kappa

def compute_kraemer_kappa(mat):
    """ Computes the Kappa value
        @param n Number of rating per subjects (number of human raters)
        @param mat Matrix[subjects][categories]
        @return The Kappa value """
    n = check_each_line_count(mat)  # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])

    if DEBUG:
        print(n, 'raters.')
        print(N, 'subjects.')
        print(k, 'categories.')

    # Computing p[]
    p = [0.0] * k
    for j in range(k):
        p[j] = 0.0
        for i in range(N):
            p[j] += mat[i][j]
        p[j] /= N * n
    if DEBUG: print('p =', p)

    # Computing P[]
    P = [0.0] * N
    for i in range(N):
        P[i] = 0.0
        for j in range(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))
    if DEBUG: print('P =', P)

    # Computing Pbar
    Pbar = sum(P) / N
    if DEBUG:
        print('Pbar =', Pbar)
        print('Sum P =', sum(P))

    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj
    if DEBUG: print('PbarE =', PbarE)

    kappa = (Pbar - PbarE) / (1 - PbarE) + (1 - Pbar)/ ((N * n) * (1 - PbarE))
    if DEBUG: print('kappa =', kappa)
    return kappa


def check_each_line_count(mat):
    """ Assert that each line has a constant number of ratings
        @param mat The matrix checked
        @return The number of ratings
        @throws AssertionError If lines contain different number of ratings """
    n = sum(mat[0])

    assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
    return n


def extract_from_result(result, key):
    """ Returns the chosen ids from an answer
        :param result: the result from one answer
        :return: the values of the chosen key from the result tokens """
    values = []
    for token in result:
        values.append(token[key])
    return values

def extract_results(filename, value='id'):
    result_dict = defaultdict(list)
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            #print(row['Input.sentence'], row['Answer.chosenWord'])
            answer_json = json.loads(row['Answer.chosenWord'])
            result_dict[row['Input.sentence']] += extract_from_result(answer_json, value)
    return result_dict

def extract_agreement_table(filename, number_of_annotators, value='id'):
    result_dict = defaultdict(list)
    table_rows = np.full([500, number_of_annotators], 0)
    with open(filename) as file:
        reader = csv.DictReader(file)
        for i,row in enumerate(reader):
            answer_json = json.loads(row['Answer.chosenWord'])
            selection = extract_from_result(answer_json, value)
            for id in selection:
                key = row['Input.sentence'] + str(id)
                if result_dict[key] == []:
                    result_dict[key] = [row['Input.sentence']] + [id] + [0] * number_of_annotators
                result_dict[key][2 + i % number_of_annotators] += 1
    return result_dict

def test_extract_results():
    hundred_dict = extract_results('100_sentence_3_batch_results.csv')
    assert len(list(hundred_dict.keys())) == 100

def generate_matrix(result_dict, num_judges=3, num_classes=2):
    values_sets = [set(l) for l in result_dict.values()]
    set_lengths = [len(list(s)) for s in values_sets]
    num_events = sum(set_lengths)
    agreement_matrix = np.zeros([num_events, num_classes], dtype=np.int64)
    events_counter = 0
    for key in result_dict.keys():
        ids = Counter(result_dict[key])
        for key in ids.keys():
            agreement_matrix[events_counter, 1] = ids[key]
            agreement_matrix[events_counter, 0] = 3 - ids[key]
            events_counter += 1
    return agreement_matrix


def compute_agreement_dict(mat):
    agreement_dict = {}
    for row in mat:
        row_str = '[' + str(row[0]) + ',' + str(row[1]) + ']'
        agreement_dict.setdefault(row_str, 'no-value')
        if agreement_dict[row_str] == 'no-value':
            agreement_dict[row_str] = 1
        else:
            agreement_dict[row_str] += 1
    return agreement_dict

def naive_agreement(result_dict):
    matched = 0
    total = 0
    for key in result_dict.keys():
        ids = Counter(result_dict[key])
        for key in ids.keys():
            if (ids[key] == 3):
                matched += 3
            else:
                matched += 2
            total += 3
    ratio = matched / total
    return ratio

def record_fless_matrix(mat):
    with open('fleiss_mat.csv', 'w+') as csv_file:
        field_names = ['not_chosen', 'chosen']
        dict_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        dict_writer.writeheader()
        for event in mat:
            dict_writer.writerow({field_names[0]: event[0],
                                  field_names[1]: event[1]})

def test_agreement_table(agreement_table, number_of_annotators):
     for event in agreement_table.values():
         if (sum(event[2:]) < 1 or sum(event[2:]) > number_of_annotators):
             print('Invalid argument table')

def record_agreement_table(agreement_table):
    length = max([len(x) for x in agreement_table.values()])
    with open('agreement_table.csv', 'w+') as csv_file:
        field_names = ['sentence', 'id'] + [str(x) for x in list(range(1, length - 1))]
        dict_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        dict_writer.writeheader()
        for event in agreement_table.values():
            row_dict = {}
            for i in range(len(event)):
                row_dict[field_names[i]] = event[i]
            dict_writer.writerow(row_dict)

# TO DO: confusion matrix is better!!
def correct_count(result_dict):
    correct = 0
    no_error_correct = 0
    no_error_sentences = 0
    correct_spot = 0
    found = 0
    missed_spans = 0
    total_spans = 0
    counted_as_no_err = 0
    fce_data = fd.extract_data('fce_train.gold.max.rasp.old_cat.m2')
    for key in result_dict.keys():
        if -2  in result_dict[key]:
            counted_as_no_err += 1
        for sentence in fce_data:
            spans_selected = []
            if key == sentence[0][1:]:
                found += 1
                if len(sentence[1]) < 1:
                    no_error_sentences += 1
                for start in result_dict[key]:
                    if start == -2 and len(sentence[1]) == 0:
                        correct += 1
                        no_error_correct += 1
                    else:
                        for i, span in enumerate(sentence[1]):
                            if int(start) >= span[0] and int(start) < span[1]:
                                correct_spot += 1
                                correct += 1
                                spans_selected.append(i)
                missed_spans += len(sentence[1]) - len(list(set(spans_selected)))
                total_spans += len(sentence[1])
    print('No error match: ', no_error_correct)
    print('No error sentences: ', no_error_sentences)
    print('Has no error answer: ', counted_as_no_err)
    print('Error match: ', correct_spot)
    print('Error missed: ', missed_spans)
    print('Found sentences: ', found)
    print('Total spans: ', total_spans)
    return correct

def error_no_error(result_dict):
    errors = 0
    no_errors = 0
    for key in result_dict.keys():
        no_errors += len([x for x in result_dict[key] if x == -2])
        errors += len([x for x in result_dict[key] if x != -2])
    return errors, no_errors

if __name__ == "__main__":
    """ Example on this Wikipedia article data set """
    sentences_ids = extract_results('100_sentence_3_batch_results.csv')
    fleiss_matrix = generate_matrix(sentences_ids)

    free_kappa = compute_free_kappa(fleiss_matrix[0:3])
    print('free kappa: ' + str(free_kappa))
    fleiss_kappa = compute_kappa(fleiss_matrix)
    print('kappa: ' + str(fleiss_kappa))
    kraemer_kappa = compute_kraemer_kappa(fleiss_matrix)
    print('kraemer kappa: ' + str(kraemer_kappa))
    naive_ratio = naive_agreement(sentences_ids)
    print('naive ratio: ' + str(naive_ratio))
    agreement_classes = compute_agreement_dict(fleiss_matrix)
    print(agreement_classes)
    record_fless_matrix(fleiss_matrix)

    mat = \
         [
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7]
        ]
    #print('fleiss kappa: ' + str(compute_kappa(mat)))

    start_result_dict = extract_results('100_sentence_3_batch_results.csv', value='start')
    c_count = correct_count(start_result_dict)
    print('correct: ' + str(c_count))
    err, no_err = error_no_error(sentences_ids)
    print('error: ', err)
    print('no-error: ', no_err)
    err, no_err = error_no_error(start_result_dict)
    print('error: ', err)
    print('no-error: ', no_err)
    agreement_table = extract_agreement_table('100_sentence_3_batch_results.csv', 3,)
    for key in agreement_table.keys():
        print(key + str(agreement_table[key]))
    test_agreement_table(agreement_table, 3)
    expert_column = []
    fce_data = fd.extract_data('fce_train.gold.max.rasp.old_cat.m2')
    for event in agreement_table.values():
        for sentence in fce_data:
            # if the sentence is matched
            if event[0] == sentence[0][1:]:
                errors_num = len(sentence[1])
                if event[1] == -2:
                    # if the sentence does not have error and the annotation guessed it does not have error false negatives
                    if errors_num < 1:
                        event.append(1)
                    # if the sentence has an error but the annotation said there are no errors true negatives
                    else:
                        event.append(0)
                # if the sentence  does not have an error but the annotator said it has an errir
                elif errors_num < 1:
                    event.append(0)
                else:
                    for i in range(errors_num):
                        # if the annotated error fits in an error span
                        if int(sentence[1][i][0]) <= event[1] // 2 and event[1] // 2 < int(sentence[1][i][1]):
                            event.append(1)
                            break
                        # if there are no more errors.
                        if i == errors_num - 1:
                            event.append(0)
    print(len(expert_column))
    record_agreement_table(agreement_table)
