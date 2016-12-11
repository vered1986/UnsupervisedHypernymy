import sys
sys.path.append('../')

import codecs

from docopt import docopt
from dsm_creation.common import *
from composes.similarity.cos import CosSimilarity


def main():
    """
    Cosine similarity
    """

    # Get the arguments
    args = docopt("""Compute cosine for a lis of (x, y) pairs and save their scores.

    Usage:
        cosine.py <testset_file> <dsm_prefix> <output_file>

        <testset_file> = a file containing term-pairs and labels, each line in the form of x\ty\tlabel
        <dsm_prefix> = the prefix for the pkl files for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tscore, where the
        score is cosine (simmetric measure).
    """)

    testset_file = args['<testset_file>']
    dsm_prefix = args['<dsm_prefix>']
    output_file = args['<output_file>']

    # Load the term-pairs
    with codecs.open(testset_file) as f_in:
        test_set = [tuple(line.strip().split('\t')) for line in f_in]

    # Load the vector space
    vector_space = load_pkl_files(dsm_prefix)

    target_index = {w: i for i, w in enumerate(vector_space.id2row)}

    # Compute the score for each term
    with codecs.open(output_file, 'w', 'utf-8') as f_out:

        for (x, y, label, relation) in test_set:

            x_index, y_index = target_index.get(x, -1), target_index.get(y, -1)
            cosine = 0.0

            if x_index > -1 and y_index > -1:
                cosine = vector_space.get_sim(x, y, CosSimilarity())

            print >> f_out, '\t'.join((x, y, label, '%.5f' % cosine))


if __name__ == '__main__':
    main()
