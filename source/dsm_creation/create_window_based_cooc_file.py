import os

from common import *
from docopt import docopt
from itertools import islice
from collections import defaultdict

MIN_FREQ = 100


def main():
    """
    Create window-based co-occurence file from Wackypedia and UKWac
    """

    # Get the arguments
    args = docopt("""Create a co-occurence file in the format w1 w2 freq, in which the context type is window-based.

    Usage:
        create_window_based_cooc_file.py <corpus_dir> <output_prefix> <frequency_file> <window_size> <directional>

        <corpus_dir> = the corpus directory
        <frequency_file> = the file containing lemmas frequencies
        <output_prefix> = the prefix for the output files: .sm sparse matrix output file, .rows and .cols
        <window_size> = the number of words on each side of the target
        <directional> = whether (dir) or not (nodir) the contexts should be directional
    """)

    corpus_dir = args['<corpus_dir>']
    freq_file = args['<frequency_file>']
    output_prefix = args['<output_prefix>']
    window_size = int(args['<window_size>'])
    directional = True if args['<directional>'] == 'dir' else False

    # Load the frequent words file
    with open(freq_file) as f_in:
        freq_words = set([line.strip() for line in f_in])

    cooc_mat = defaultdict(lambda: defaultdict(int))

    corpus_files = sorted([corpus_dir + '/' + file for file in os.listdir(corpus_dir) if file.endswith('.gz')])

    for file_num, corpus_file in enumerate(corpus_files):

        print 'Processing corpus file %s (%d/%d)...' % (corpus_file, file_num + 1, len(corpus_files))

        for sentence in get_sentences(corpus_file):
            update_window_based_cooc_matrix(cooc_mat, freq_words, sentence, window_size, directional)

    # Filter contexts
    frequent_contexts = filter_contexts(cooc_mat, MIN_FREQ)

    # Save the files
    save_files(cooc_mat, frequent_contexts, output_prefix)


def update_window_based_cooc_matrix(cooc_mat, freq_words, sentence, window_size, directional):
    """
    Updates the co-occurrence matrix with the current sentence
    :param cooc_mat: the co-occurrence matrix
    :param freq_words: the list of frequent words
    :param sentence: the current sentence
    :param window_size: the number of words on each side of the target
    :param directional: whether to distinguish between contexts before and after the target
    :return: the update co-occurrence matrix
    """

    # Remove all the non relevant words, keeping only NN, JJ and VB
    strip_sentence = [(w_word, w_lemma, w_pos, w_index, w_parent, w_dep) for
                      (w_word, w_lemma, w_pos, w_index, w_parent, w_dep) in sentence
                      if w_pos.startswith('N') or w_pos.startswith('V') or w_pos.startswith('J')]

    # Add 1 for content words in window_size, either differentiating or not the contexts by direction
    for i, (t_word, t_lemma, t_pos, t_index, t_parent, t_dep) in enumerate(strip_sentence):

        # Make sure the target is a frequent enough word
        if t_lemma not in freq_words:
            continue

        # print type(t_pos)
        target_lemma = t_lemma + '-' + t_pos[0].decode('utf8').lower()  # lemma + first char of POS, e.g. run-v / run-n

        # Update left contexts if they are inside the window and after BOS (and frequent enough)
        if i > 0:
            for l in range(max(0, i-window_size), i):

                _, c_lemma, c_pos, _, _, _ = strip_sentence[l]

                if c_lemma not in freq_words:
                    continue

                prefix = '-l-' if directional else '-'
                context = c_lemma + prefix + c_pos[0].decode('utf8').lower()  # context lemma + left + lower pos
                cooc_mat[target_lemma][context] = cooc_mat[target_lemma][context] + 1

        # Update right contexts if they are inside the window and before EOS (and frequent enough)
        for r in range(i + 1, min(len(strip_sentence), i + window_size + 1)):

            _, c_lemma, c_pos, _, _, _ = strip_sentence[r]

            if c_lemma not in freq_words:
                continue

            prefix = '-r-' if directional else '-'
            context = c_lemma + prefix + c_pos[0].decode('utf8').lower()
            cooc_mat[target_lemma][context] = cooc_mat[target_lemma][context] + 1

    return cooc_mat


def n_grams(a, n):
    z = (islice(a, i, None) for i in range(n))
    return zip(*z)


if __name__ == '__main__':
    main()
