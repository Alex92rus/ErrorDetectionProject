

def extract_to_m2(filename, annot_triples):
    """
    Extracts error detection annotations in m2 file format
    Args:
       filename: the output m2 file
       annot_triples: the annotations of form (sentence, indexes, selections)
    """
    with open(filename, 'w+') as m2_file:
        for triple in annot_triples:
            s_line = 'S ' + triple[0] + '\n'
            m2_file.write(s_line)
            for i in range(len(triple[1])):
                if triple[2][i] == 1:
                    a_line = 'A '
                    if isinstance(triple[1][i], int):
                        a_line += str(triple[1][i]) + ' ' + str(triple[1][i] + 1)
                    else:
                        a_line += triple[1][i] + ' ' + triple[1][i]
                    a_line += '|||IG|||IG|||REQUIRED|||-NONE-|||1\n'
                    m2_file.write(a_line)
            m2_file.write('\n')