import sys
sys.path.append('../')

import codecs

from docopt import docopt
from dsm_creation.common import *


def main():
    """
    lin - as described in:
    Dekang Lin. 1998. An information-theoretic definition of similarity.
    """

    # Get the arguments
    args = docopt("""Lin similarity

    Usage:
        lin.py <testset_file> <dsm_prefix> <output_file>

        <testset_file> = a file containing term-pairs, labels and relations, each line in the form 
                         of x\ty\tlabel\trelation
        <dsm_prefix> = the prefix for the pkl files for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tlabel\trelation\tscore,
                        where the score is Lin (for y as the hypernym of x).
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
                score = lin(x_row, y_row)

            print >> f_out, '\t'.join((x, y, label, '%.5f' % score))


def lin(x_row, y_row):
    """
    Lin(x, y) = (\sigma_c \in Fx,y [w_x(c) + w_y(c)]) / (\sigma_c \in Fx w_x(c) + \sigma_c \in Fy w_y(c))
    Fx,y is the mutual contexts (non-zero entries) of x and y rows in the ppmi matrix
    w_x(c) is the weight of feature c in x's feature vector, i.e. ppmi(x, c)
    Fx is the row of x in the ppmi matrix
    :param x_row: x's row in the co-occurrence matrix
    :param y_row: y's row in the co-occurrence matrix
    :return:
    """

    y_ones = y_row.get_non_negative() # Just to copy
    y_ones.to_ones()
    x_ones = x_row.get_non_negative() # Just to copy
    x_ones.to_ones()
    numerator = x_row.multiply(y_ones).sum() + y_row.multiply(x_ones).sum() # dot-product

    denominator = x_row.sum() + y_row.sum()

    return 0.0 if denominator == 0 else numerator * (1.0 / denominator)

if __name__ == '__main__':
    main()
