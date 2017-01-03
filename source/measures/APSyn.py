import sys
sys.path.append('../')

import codecs
import scipy.sparse

from docopt import docopt
from itertools import izip
from dsm_creation.common import *


def main():
    """
    APSyn - as described in:
    E. Santus, T.-S. Chiu, A. Lenci, Q. Lu, C.-R. Huang. 2016. 
    What a Nerd! Beating Students and Vector Cosine in the ESL and TOEFL Datasets. In LREC 2016.
    """

    # Get the arguments
    args = docopt("""Compute APSyn for a list of (x, y) pairs on the N most relevant contexts, and save their scores.

    Usage:
        APSyn.py <testset_file> <N> <dsm_prefix> <output_file>

	<N> = the number of relevant contexts to be considered
	<testset_file> = a file containing term-pairs, labels and relations, each line in the form 
			 of x\ty\tlabel\trelation
 	<dsm_prefix> = the prefix for the pkl files for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tlabel\trelation\tscore,
			where the score is APSyn.
    """)

    testset_file = args['<testset_file>']
    N = int(args['<N>'])
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
                score = APSyn(x_row, y_row, N)

            print >> f_out, '\t'.join((x, y, label, '%.5f' % score))


def APSyn(x_row, y_row, N):
    """
    APSyn(x, y) = (\sum_{f\epsilon N(f_{x})\bigcap N(f_{y})))} \frac{1}{(rank(f_{x})+rank(f_{y})/2)})
    :param x_row:
    :param y_row:
    :return:
    """

    # Sort y's contexts
    y_contexts_cols = sort_by_value_get_col(scipy.sparse.coo_matrix(y_row.mat)) # tuples of (row, col, value)
    y_contexts_cols = y_contexts_cols[:N]
    y_context_rank = { c : i + 1 for i, c in enumerate(y_contexts_cols) }

    # Sort x's contexts
    x_contexts_cols = sort_by_value_get_col(scipy.sparse.coo_matrix(x_row.mat))
    x_contexts_cols = x_contexts_cols[:N]
    x_context_rank = { c : i + 1 for i, c in enumerate(x_contexts_cols) }

    # Average of 1/(rank(w1)+rank(w2)/2) for every intersected feature among the top N contexts
    intersected_context = set(y_contexts_cols).intersection(set(x_contexts_cols))
    score = sum([1.0 / ((x_context_rank[c] + y_context_rank[c]) / 2.0) for c in intersected_context])
    #score *= (1.0 / N)

    return score


def sort_by_value_get_col(mat):
    """
    Sort a sparse coo_matrix by values and returns the columns (the matrix has 1 row)
    :param mat: the matrix
    :return: a sorted list of tuples columns by descending values
    """
    tuples = izip(mat.row, mat.col, mat.data)
    sorted_tuples = sorted(tuples, key=lambda x: x[2], reverse=True)

    if len(sorted_tuples) == 0:
        return []

    rows, columns, values = zip(*sorted_tuples)
    return columns


if __name__ == '__main__':
    main()
