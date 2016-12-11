import sys
sys.path.append('../')

import codecs
import numpy as np

from docopt import docopt
from dsm_creation.common import *
from balAPinc import get_contexts_rank
from composes.transformation.scaling.epmi_weighting import EpmiWeighting


def main():
    """
    RCTC - as described in:
    Laura Rimell. 2014. Distributional lexical entailment by topic coherence. In EACL.
    """

    # Get the arguments
    args = docopt("""Compute RCTC for a list of (x, y) pairs and save their scores.

    Usage:
        rctc.py <testset_file> <N> <dsm_prefix> <output_file>

        <testset_file> = a file containing term-pairs and labels, each line in the form of x\ty\tlabel
        <N> = the number of contexts.
        <dsm_prefix> = the prefix for the pkl files for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tscore.
    """)

    testset_file = args['<testset_file>']
    N = int(args['<N>'])
    dsm_prefix = args['<dsm_prefix>']
    output_file = args['<output_file>']

    # Load the term-pairs
    with codecs.open(testset_file) as f_in:
        test_set = [tuple(line.strip().split('\t')) for line in f_in]

    # Load the frequency-based vector space and apply PMI weighting
    vector_space = load_pkl_files(dsm_prefix)
    cooc_mat = vector_space.cooccurrence_matrix.copy()

    vector_space = vector_space.apply(EpmiWeighting())
    ppmi_mat = vector_space.cooccurrence_matrix
    ppmi_mat.mat.data = np.log(cooc_mat.mat.data)

    target_index = {w: i for i, w in enumerate(vector_space.id2row)}

    xs, ys, labels, relations = zip(*test_set)
    targets = set(xs).union(set(ys))

    # Get ranked contexts for each target - this is a time-consuming operation that should
    # be performed only once for every word
    contexts_rank = get_contexts_rank(targets, cooc_mat, target_index)

    # Compute the score for each term
    with codecs.open(output_file, 'w', 'utf-8') as f_out:

        for (x, y, label, relation) in test_set:

            score = rctc(x, y, target_index, vector_space.id2column, N, contexts_rank, ppmi_mat)
            print >> f_out, '\t'.join((x, y, label, str(score)))


def topic_coherence(topic_words, target_index, contexts, ppmi_mat):
    """
    Computes topic coherence by the average pairwise PMI score
    :return:
    """

    # Convert context indices to target indices
    target_indices = [target_index.get(contexts[w], -1) for w in topic_words]

    # Compute the median of pairwise scores
    pairwise_scores = [ppmi_mat[w1, w2] if w1 > -1 and w2 > -1 else 0.0 for w1 in target_indices for w2 in topic_words]
    median = np.median(pairwise_scores)
    return median


def rctc(x, y, target_index, contexts, N, contexts_rank, ppmi_mat):
    """
    rctc(x, y) = (TC(topic x)/TC(topic x not y))/(TC(topic y)/TC(topic~y~not~x))
    :return:
    """
    x_index, y_index = target_index.get(x, -1), target_index.get(y, -1)
    rctc = 0.0

    if x_index > -1 and y_index > -1:

        # Get the top N contexts of each word
        topic_x = contexts_rank[x][:N+1]
        topic_y = contexts_rank[y][:N+1]

        # Get the x-not-y and y-not-x contexts
        set_contexts_y = set(contexts_rank[y])
        topic_x_not_y = []

        for c in contexts_rank[x]:
            if c not in set_contexts_y:
                topic_x_not_y.append(c)
                if len(topic_x_not_y) == N:
                    break

        set_contexts_x = set(contexts_rank[x])
        topic_y_not_x = []

        for c in contexts_rank[y]:
            if c not in set_contexts_x:
                topic_y_not_x.append(c)
                if len(topic_y_not_x) == N:
                    break

        # Get topics coherence
        tc_x = topic_coherence(topic_x, target_index, contexts, ppmi_mat)
        tc_y = topic_coherence(topic_y, target_index, contexts, ppmi_mat)
        tc_x_not_y = topic_coherence(topic_x_not_y, target_index, contexts, ppmi_mat)
        tc_y_not_x = topic_coherence(topic_y_not_x, target_index, contexts, ppmi_mat)

        numerator = 0.0 if tc_x_not_y == 0.0 else tc_x / (1.0 * tc_x_not_y)
        denominator = 0.0 if tc_y_not_x == 0.0 else tc_y / (1.0 * tc_y_not_x)

        rctc = 0.0 if denominator == 0 else numerator * (1.0 / denominator)

    return rctc


if __name__ == '__main__':
    main()
