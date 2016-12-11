import sys
sys.path.append('../')

import codecs
import numpy as np

from docopt import docopt
from dsm_creation.common import *


def main():
    """
    clarkeDE - as described in:
    Daoud Clarke. 2009. Context-theoretic semantics for natural language: an overview. In EACL.
    """

    # Get the arguments
    args = docopt("""Compute clarkeDE for a list of (x, y) pairs and save their scores.

    Usage:
        clarkeDE.py <testset_file> <dsm_prefix> <output_file>

        <testset_file> = a file containing term-pairs and labels, each line in the form of x\ty\tlabel
        <dsm_prefix> = the prefix for the pkl files for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tscore, where the
        score is ClarkeDE (for y as the hypernym of x).
    """)

    testset_file = args['<testset_file>']
    dsm_prefix = args['<dsm_prefix>']
    output_file = args['<output_file>']

    # Load the term-pairs
    with codecs.open(testset_file) as f_in:
        test_set = [tuple(line.strip().split('\t')) for line in f_in]

    # Load the vector space
    vector_space = load_pkl_files(dsm_prefix)

    target_index = { w : i for i, w in enumerate(vector_space.id2row) }

    cooc_mat = vector_space.cooccurrence_matrix

    # Compute the score for each term
    with codecs.open(output_file, 'w', 'utf-8') as f_out:

        for (x, y, label, relation) in test_set:

            x_index, y_index = target_index.get(x, -1), target_index.get(y, -1)
            score = 0.0

            if x_index > -1 and y_index > -1:

                x_row, y_row = cooc_mat[x_index, :], cooc_mat[y_index, :]
                score = clarkeDE(x_row, y_row)

            print >> f_out, '\t'.join((x, y, label, '%.5f' % score))


def clarkeDE(x_row, y_row):
    """
    clarkeDE(x -> y) = (\sigma_c \in Fx,y min(w_x(c), w_y(c))) / (\sigma_c \in Fx w_x(c))
    Fx,y is the mutual contexts (non-zero entries) of x and y rows in the ppmi matrix
    w_x(c) is the weight of feature c in x's feature vector, i.e. ppmi(x, c)
    Fx is the row of x in the ppmi matrix
    :param x_row: x's row in the co-occurrence matrix
    :param y_row: y's row in the co-occurrence matrix
    :return:
    """

    # Get the sum of the minimum for each context. Only the mutual contexts will yield values > 0
    numerator = np.minimum(x_row.to_dense_matrix().get_mat(), y_row.to_dense_matrix().get_mat()).sum()

    # The sum of x's contexts (for ppmi) is the sum of x_row.
    denominator = x_row.sum()

    return 0.0 if denominator == 0 else numerator * (1.0 / denominator)


if __name__ == '__main__':
    main()
