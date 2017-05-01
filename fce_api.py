import random
import re
from collections import defaultdict


def extract_data(filename):
    """
    Extracts the sentences from m2 file.
    Each sentence is formed into a tuple format (sentence, error_spans)
    Where error_spans is a list with the error spans in the sentence
    The error span is a tuple with the form (start_index, end_index, error_type, error_correction)
    Args:
        filename: the name of the m2 input file

    Returns:
        list of sentence tuples of format (sentence, error_spans)
    """

    # open file
    fce_file = open(filename)

    # read the lines
    iter_lines = fce_file.readlines()

    # array of tuples - (sentence, [error_spans])
    data = []

    # packet the error spans with their corresponding sentence
    for line in iter_lines:
        # appending the sentence to the data
        if line[0] == 'S':
            data.append((line[1:-1], []))
        elif line[0] == 'A':
            tokens = line.split(' ')
            start = int(tokens[1])
            error_details = tokens[2].split('|||')
            end = int(error_details[0])
            error_type = error_details[1]
            error_replace = error_details[2]
            data[-1][1].append((start, end, error_type, error_replace))
    
    # close file
    fce_file.close()
    
    return data


def extract_data_sentences(filename):
    # open file
    fce_file = open(filename)

    # read the lines
    iter_lines = fce_file.readlines()

    # array of tuples - (sentence, [error_spans])
    data = []

    # packet the error spans with their corresponding sentence
    for line in iter_lines:
        # appending the sentence to the data
        if re.search('[a-zA-z]+', line):
            data.append((line[:-1], []))
        elif re.search('[0-9][0-9]*\s,\s[0-9][0-9]*', line):
            tokens = line.split(' ')
            start = int(tokens[0])
            end = int(tokens[2])
            data[-1][1].append((start, end))

    # close file
    fce_file.close()

    return data


def extract_data_from_tsv(filename, limit=0):

    data = []

    with open(filename, 'r') as tsv:
        line_tokens = [line.strip().split('\t') for line in tsv]
        tokens = []
        errors = []
        for line in line_tokens:
            if len(line) == 1:
                if len(tokens) > limit:
                    data.append(((' '.join(tokens)), errors))
                tokens.clear()
                errors = []
            else:
                tokens.append(line[0])
                if line[1] == 'i':
                    errors.append((len(tokens) - 1, len(tokens), 'i'))
    return data


def dump_sentences_to_file(data, filename):

    with open(filename, 'w+') as output:
        for sentence_info in data:
            output.write(sentence_info[0] + '\n')
            for error_span in sentence_info[1]:
                output.write(str(error_span[0]) + ', ' + str(error_span[1]) + '\n')
            output.write('\n')


def matching_sentences():
    fce_train_data = extract_data_from_tsv('fce-public.train.original.tsv')
    fce_all_data = extract_data('fce_train.gold.max.rasp.old_cat.m2')
    fce_train_data.sort(key=lambda x: x[0])
    fce_all_data.sort(key=lambda x: x[0])
    #missing = 'Despite of the fact that I am a student , I could not get any discounts .'
    with open('sentences', 'w+') as file:
        for sent in fce_train_data:
            match = [x for x in fce_all_data if x[0][1:] == sent[0]]
            if len(match) > 0:
                file.write(match[0][0][1:] + '\n')
                for error in match[0][1]:
                    file.write(str(error[0]) + ', ' + str(error[1]) + ' ' + error[2] + '\n')
                file.write('\n')


def count_error_types(filename):
    no_error = 'no error'
    sentences = extract_data(filename)
    error_vocab = defaultdict(list)
    count_vocab = defaultdict(int)
    for sentence in sentences:
        if len(sentence[1]) < 1:
            error_vocab[no_error].append((sentence[0], -1, -1, -1))
            count_vocab[no_error] += 1
        for errors in sentence[1]:
            error_vocab[errors[2]].append((sentence[0], errors[0], errors[1], errors[3]))
            count_vocab[errors[2]] += 1
    return error_vocab, count_vocab


def convert_m2_to_tsv(m2_filename, destination_filename):
    """
    Converts CLC FCE m2 annotated file to tsv classifier input format
    Args:
        m2_filename: the name of the m2 input file
        destination_filename: the name of the destination tsv file
    """
    fce_sentence_data = extract_data(m2_filename)
    with open(destination_filename, 'w+') as destination_tsv:
        for sentence, errors in fce_sentence_data:
            tokens = sentence.split(' ')[1:]
            incorrect = []
            for error in errors:
                for i in range(error[0], error[1]):
                    if i not in incorrect:
                        incorrect.append(i)
                # add the word if the error is on a space
                if error[0] == error[1]:
                    incorrect.append(error[0])
            for tokenIndex in range(0, len(tokens)):
                line = ''
                if tokenIndex in incorrect:
                    line = '{}\t{}'.format(tokens[tokenIndex], 'i\n')
                else:
                    line = '{}\t{}'.format(tokens[tokenIndex], 'c\n')
                destination_tsv.write(line)
            destination_tsv.write('\n')
        destination_tsv.write('\n')


def generate_train_percentage_data(train_file, destination_dir):
    """
    Converts CLC FCE m2 annotated file to tsv classifier input format
    Args:
        m2_filename: the name of the m2 input file
        destination_filename: the name of the destination tsv file
    """
    fce_sentence_data = extract_data(train_file)
    one_tenth = len(fce_sentence_data) // 10
    for i in range(1, 11):
        indexes = random.sample(range(0, len(fce_sentence_data)), one_tenth * i)
        destination_file = destination_dir + '/' + 'train_' + str(i * 10) + 'percent'
        fce_sentence_data_batch = [sentence[1] for sentence in enumerate(fce_sentence_data) if sentence[0] in indexes]
        with open(destination_file, 'w+') as destination_tsv:
            for sentence, errors in fce_sentence_data_batch:
                tokens = sentence.split(' ')[1:]
                incorrect = []
                for error in errors:
                    for i in range(error[0], error[1]):
                        if i not in incorrect:
                            incorrect.append(i)
                    # add the word if the error is on a space
                    if error[0] == error[1]:
                        incorrect.append(error[0])
                for tokenIndex in range(0, len(tokens)):
                    line = ''
                    if tokenIndex in incorrect:
                        line = '{}\t{}'.format(tokens[tokenIndex], 'i\n')
                    else:
                        line = '{}\t{}'.format(tokens[tokenIndex], 'c\n')
                    destination_tsv.write(line)
                destination_tsv.write('\n')
            destination_tsv.write('\n')


if __name__ == '__main__':
    # fce_data = extract_data('fce_train.gold.max.rasp.old_cat.m2')
    # print("Sentence: {sentence} with list of error indecies {errors}".format(sentence=data[12][0], errors=data[12][1]))
    generate_train_percentage_data('fce_train.gold.max.rasp.old_cat.m2', 'train_data_chuncks_tsv')