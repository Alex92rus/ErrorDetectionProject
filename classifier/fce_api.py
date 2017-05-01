import re
import csv
import numpy as np

def extract_data(filename):
    #open file
    fce_file = open(filename)

    #read the lines
    iter_lines = fce_file.readlines()
    
    #array of tuples - (sentence, [error_spans])
    data = []
    
    #packet the error spans with their corresponding sentence
    for line in iter_lines:
        #appending the sentence to the data
        if line[0] == 'S':
            data.append((line[1:-1], []))
        elif line[0] == 'A':
            tokens = line.split(' ')
            start = int(tokens[1])
            error_details = tokens[2].split('|||')
            end = int(error_details[0])
            error_type = error_details[1]
            data[-1][1].append((start, end, error_type))

    for i in range(10):
        print(data[i])
    #close file
    fce_file.close()
    
    return data

def extract_data_from_tsv(filename, limit=0):

    data = []

    with open(filename, 'r') as tsv:
        line_tokens = [line.strip().split('\t') for line in tsv]
        sentence = ''
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
                if (line[1] == 'i'):
                    errors.append((len(tokens) - 1, len(tokens), 'i'))
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
    missing = 'Despite of the fact that I am a student , I could not get any discounts .'
    with open('sentences', 'w+') as file:
        for sent in fce_train_data:
            match = [x for x in fce_all_data if x[0][1:] == sent[0]]
            if (len(match) > 0):
                file.write(match[0][0][1:] + '\n')
                for error in match[0][1]:
                    file.write(str(error[0]) + ', ' + str(error[1]) + ' ' + error[2] + '\n')
                file.write('\n')


def retrieve_random_batch(data, batch_size=100):
    set_sentences = list(set([tuple(x[0].split(' ')) for x in data]))
    indices = np.random.choice(len(set_sentences), batch_size, replace=False)
    random_sentences = [set_sentences[index] for index in indices]
    random_batch = [' '.join(sentence) for sentence in random_sentences]
    errors = []
    for sentence in random_batch:
     for sentence_tuple in data:
         if sentence == sentence_tuple[0]:
           errors.append(sentence_tuple[1])
    return random_batch, errors

def dump_to_csv(filename, sentences):
    with open(filename, 'w', newline='') as csv_file:
        field_names = ['sentence']
        dict_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        dict_writer.writeheader()
        for sentence in sentences:
            dict_writer.writerow({field_names[0]: sentence})
    line_tokens = [line.strip().split('\t') for line in tsv]

def extract_data_from_csv(filename, limit=0):

    data = []

    with open(filename, 'r') as csv_file:
        iter_lines = csv_file.readlines()
        for line in iter_lines:
            data.append(line[:-1])
    return data

if __name__ == '__main__':
    # fce_train_data = extract_data_from_tsv('fce-public.train.original.tsv', limit=5)
    # batch, errors_batch  = retrieve_random_batch(fce_train_data)
    # dump_to_csv('amt_sentence_batch.csv', batch)
    data = extract_data_from_csv('amt_sentence_batch.csv')
    fce_with_errors = extract_data('fce_train.gold.max.rasp.old_cat.m2')
    errors = []
    unmatched = []
    counter = 0
    counter_2 = 0
    for sentence in data:
        found = False
        for fce_sent, errors_sent in fce_with_errors:
            if sentence[1:-1] == fce_sent[1:] or sentence == fce_sent[1:]:
                counter += 1
                errors.append(errors_sent)
                found = True
        if not found:
            unmatched.append(sentence)
        else:
            counter_2 += 1

    for unmatched_sentence in unmatched:
        print(unmatched_sentence)
    print(len(unmatched_sentence))
    print(len(errors))
    print(counter)
    print(counter_2)
    #dump_sentences_to_file(fce_train_data, 'train_sentences.txt')