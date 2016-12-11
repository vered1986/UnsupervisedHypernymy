import os

from common import *
from docopt import docopt
from collections import defaultdict

MIN_FREQ = 100


def main():
    """
    Create dependency-based co-occurence file from Wackypedia and UKWac
    """

    # Get the arguments
    args = docopt("""Create a co-occurence file in the format w1 w2 freq, in which the context type is dependency-based,
                    i.e. w1 is a target word and w2='dependency:w2' where w1, w2 are connected in a dependency tree.

    Usage:
        create_dep_based_cooc_file.py <corpus_dir> <output_prefix> <frequency_file>

        <corpus_dir> = the corpus directory
        <frequency_file> = the file containing lemmas frequencies
        <output_prefix> = the prefix for the output files: .sm sparse matrix output file, .rows and .cols
    """)

    corpus_dir = args['<corpus_dir>']
    freq_file = args['<frequency_file>']
    output_prefix = args['<output_prefix>']

    # Load the frequent words file
    with open(freq_file) as f_in:
        freq_words = set([line.strip() for line in f_in])

    cooc_mat = defaultdict(lambda : defaultdict(int))

    corpus_files = sorted([corpus_dir + '/' + file for file in os.listdir(corpus_dir) if file.endswith('.gz')])

    for file_num, corpus_file in enumerate(corpus_files):

        print 'Processing corpus file %s (%d/%d)...' % (corpus_file, file_num + 1, len(corpus_files))

        for sentence in get_sentences(corpus_file):
            update_dep_based_cooc_matrix(cooc_mat, freq_words, sentence)

    # Filter contexts
    frequent_contexts = filter_contexts(cooc_mat, MIN_FREQ)

    # Save the files
    save_files(cooc_mat, frequent_contexts, output_prefix)


def update_dep_based_cooc_matrix(cooc_mat, freq_words, sentence):
    """
    Updates the co-occurrence matrix with the current sentence
    :param cooc_mat: the co-occurrence matrix
    :param freq_words: the list of frequent words
    :param sentence: the current sentence
    :return: the update co-occurrence matrix
    """

    for (word, lemma, pos, index, parent, dep) in sentence:

        # Make sure the target is either a noun, verb or adjective, and it is a frequent enough word
        if lemma not in freq_words or \
                not (pos.startswith('N') or pos.startswith('V') or pos.startswith('J')):
            continue

        # Not root
        if parent != 0:

            # Get context token and make sure it is either a noun, verb or adjective, and it is
            # a frequent enough word

            # Can't take sentence[parent - 1] because some malformatted tokens might have been skipped!
            parents = [token for token in sentence if token[-2] == parent]

            if len(parents) > 0:
                _, c_lemma, c_pos, _, _, _ = parents[0]

                if c_lemma not in freq_words or \
                        not (c_pos.startswith('N') or c_pos.startswith('V') or c_pos.startswith('J')):
                    continue

                target = lemma + '-' + pos[0].lower()  # lemma + first char of POS, e.g. run-v / run-n

                context = dep + ':' + c_lemma + '-' + c_pos[0].lower()  # dependency label : parent lemma
                cooc_mat[target][context] = cooc_mat[target][context] + 1

                # Add the reversed edge
                reversed_target = c_lemma + '-' + c_pos[0].lower()
                reversed_context = dep + '-1:' + lemma + '-' + pos[0].lower()
                cooc_mat[reversed_target][reversed_context] = cooc_mat[reversed_target][reversed_context] + 1

    return cooc_mat


if __name__ == '__main__':
    main()
