import sys
sys.path.append('../')

import codecs

from docopt import docopt
from dsm_creation.common import *


def main():
    """
    Weeds Precision - as described in:
    J. Weeds and D. Weir. 2003. A general framework for distributional similarity. In EMNLP.
    """

    # Get the arguments
    args = docopt("""Compute Weeds Precision for a list of (x, y) pairs and save their scores.

    Usage:
        weeds.py <testset_file> <dsm_prefix> <output_file>

        <testset_file> = a file containing term-pairs and labels, each line in the form of x\ty\tlabel
        <dsm_prefix> = the prefix for the pkl files for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tscore, where the
        score is Weeds Precision (for y as the hypernym of x).
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
                score = weeds_prec(x_row, y_row)

            print >> f_out, '\t'.join((x, y, label, '%.5f' % score))


def weeds_prec(x_row, y_row):
    """
    WeedsPrec(x -> y) = (\sigma_c \in Fx,y w_x(c)) / (\sigma_c \in Fx w_x(c))
    Fx,y is the mutual contexts (non-zero entries) of x and y rows in the ppmi matrix
    w_x(c) is the weight of feature c in x's feature vector, i.e. ppmi(x, c)
    Fx is the row of x in the ppmi matrix
    :param x_row: x's row in the co-occurrence matrix
    :param y_row: y's row in the co-occurrence matrix
    :return:
    """

    # Get the mutual contexts: use y as a binary vector and apply dot product with x:
    # If c is a mutual context, it is 1 in y_non_zero and the value ppmi(x, c) is added to the sum
    # Otherwise, if it is 0 in either x or y, it adds 0 to the sum.
    y_row.to_ones()
    numerator = x_row.multiply(y_row).sum() # dot-product

    # The sum of x's contexts (for ppmi) is the sum of x_row.
    denominator = x_row.sum()

    return 0.0 if denominator == 0 else numerator * (1.0 / denominator)


if __name__ == '__main__':
    main()
