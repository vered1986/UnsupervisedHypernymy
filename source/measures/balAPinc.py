import sys
sys.path.append('../')

import math
import codecs
import numpy as np
import scipy.sparse

from lin import lin
from docopt import docopt
from itertools import izip
from dsm_creation.common import *


def main():
    """
    balAPinc - as described in:
    L. Kotlerman, I. Dagan, I. Szpektor and M. Zhitomirsky-Geffet. 2009.
    Directional Distributional Similarity for Lexical Expansion. In ACL.
    """

    # Get the arguments
    args = docopt("""Compute balAPinc for a list of (x, y) pairs and save their scores.

    Usage:
        balapinc.py <testset_file> <dsm_prefix> <output_file>

        <testset_file> = a file containing term-pairs and labels, each line in the form of x\ty\tlabel
        <dsm_prefix> = the prefix for the pkl files for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tscore, where the
        score is balAPinc (for y as the hypernym of x).
    """)

    testset_file = args['<testset_file>']
    dsm_prefix = args['<dsm_prefix>']
    output_file = args['<output_file>']

    # Load the term-pairs
    with codecs.open(testset_file) as f_in:
        test_set = [tuple(line.strip().split('\t')) for line in f_in]

    xs, ys, labels, relations = zip(*test_set)
    targets = set(xs).union(set(ys))

    # Load the vector space
    vector_space = load_pkl_files(dsm_prefix)

    target_index = { w : i for i, w in enumerate(vector_space.id2row) }

    cooc_mat = vector_space.cooccurrence_matrix

    # Get ranked contexts for each target - this is a time-consuming operation that should
    # be performed only once for every word
    contexts_rank = get_contexts_rank(targets, cooc_mat, target_index)

    # Compute the score for each term
    with codecs.open(output_file, 'w', 'utf-8') as f_out:

        for (x, y, label, relation) in test_set:

            print >> f_out, '\t'.join((x, y, label, str(balAPinc(x, y, contexts_rank, target_index, cooc_mat))))


def get_contexts_rank(targets, cooc_mat, target_index):
    """
    A dictionary in which each key is a target word and the value is a sorted list of
    context columns in descending order
    :param targets: the words
    :return:
    """
    contexts_rank = {}

    for target in targets:
        index = target_index.get(target, -1)

        if index == -1:
            contexts_rank[target] = []

        row = cooc_mat[index, :]
        contexts_rank[target] = sort_by_value_get_col(scipy.sparse.coo_matrix(row.mat)) # tuples of (row, col, value)

    return contexts_rank


def balAPinc(x, y, contexts_rank, target_index, cooc_mat):
    """
    balAPinc(x, y) = sqrt(lin(x, y) * APinc(x -> y))
    :return:
    """
    x_index, y_index = target_index.get(x, -1), target_index.get(y, -1)
    balapinc = 0.0

    if x_index > -1 and y_index > -1:

        x_row, y_row = cooc_mat[x_index, :], cooc_mat[y_index, :]
        balapinc = math.sqrt(lin(x_row, y_row) * apinc(x, y, contexts_rank))

    return balapinc


def apinc(x, y, contexts_rank):
    """
    APinc(x, y) = (\sigma c \in row_x (p(c) * rel(c))) / num_x_contexts
    :param x:
    :param y:
    :param contexts_rank
    :return:
    """

    # Get y's sorted contexts
    y_contexts_cols = contexts_rank[y]
    num_y_contexts = len(y_contexts_cols)

    # Get x's sorted contexts
    x_contexts_cols = contexts_rank[x]

    # No contexts or no mutual contexts
    if num_y_contexts == 0:
        return 0.0

    y_contexts_cols_set = set(y_contexts_cols)
    y_context_rank = { c : i + 1 for i, c in enumerate(y_contexts_cols) }

    # rel(c) = 0 if c is not a context of y, and 1 - (rank(c, row_y) / (num_y_contexts + 1))
    rel_c = [0.0 if c not in y_contexts_cols_set else 1.0 - y_context_rank[c] / (num_y_contexts + 1.0)
             for c in x_contexts_cols]

    # p(c) = |included features in ranks 1 to r| / r
    p_c = []
    curr_intersection = 0

    for r in range(len(x_contexts_cols)):
        if x_contexts_cols[r] in y_contexts_cols_set:
            curr_intersection += 1
        p_c.append(curr_intersection / (1.0 * (r + 1)))

    score = np.dot(np.array(rel_c), np.array(p_c)) / (1.0 * num_y_contexts)

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
