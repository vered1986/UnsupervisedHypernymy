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
                    i.e. w1 is a target word and w2='wi#wj' where w1 is connected to wi in a dependency tree,
                    and wi is also connected to wj, e.g.: cat : eat#food

    Usage:
        create_joint_dep_based_cooc_file.py <corpus_dir> <output_prefix> <frequency_file>

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
            update_joint_dep_based_cooc_matrix(cooc_mat, freq_words, sentence)

    # Filter contexts
    frequent_contexts = filter_contexts(cooc_mat, MIN_FREQ)

    # Save the files
    save_files(cooc_mat, frequent_contexts, output_prefix)


def update_joint_dep_based_cooc_matrix(cooc_mat, freq_words, sentence):
    """
    Updates the co-occurrence matrix with the current sentence
    :param cooc_mat: the co-occurrence matrix
    :param freq_words: the list of frequent words
    :param sentence: the current sentence
    :return: the update co-occurrence matrix
    """

    # Look for c1->p<-c2 structure, and use c1 as the target and p-c2 as the context
    for (p_word, p_lemma, p_pos, p_index, p_parent, p_dep) in sentence:

        # Make sure the parent is a frequent enough word
        if p_lemma not in freq_words:
            continue

        # Look for child nodes of this current parent and make sure they are either a noun,
        # verb or adjective, and frequent enough
        child_nodes = [(c_word, c_lemma, c_pos, c_index, c_parent, c_dep) for
                       (c_word, c_lemma, c_pos, c_index, c_parent, c_dep) in sentence
                       if c_parent == p_index and c_lemma in freq_words and
                       (c_pos.startswith('N') or c_pos.startswith('V') or c_pos.startswith('J'))]

        pairs = [((c1[1], c1[2]), (c2[1], c2[2]))
                 for i, c1 in enumerate(child_nodes)
                 for j, c2 in enumerate(child_nodes) if i < j]

        for ((c1_lemma, c1_pos), (c2_lemma, c2_pos)) in pairs:

            target_lemma = c1_lemma + '-' + c1_pos[0].lower() # lemma + first char of POS, e.g. run-v / run-n

            context = p_lemma + '-' + p_pos[0].lower() + '#' + c2_lemma + '-' + c2_pos[0].lower() # e.g. eat-food
            cooc_mat[target_lemma][context] = cooc_mat[target_lemma][context] + 1

    return cooc_mat


if __name__ == '__main__':
    main()
