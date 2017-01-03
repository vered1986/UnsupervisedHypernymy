import sys
sys.path.append('../')

import math
import codecs

from common import *
from docopt import docopt
from clarkeDE import clarkeDE
from dsm_creation.common import *


def main():
    """
    invCL - as described in:
    A. Lenci and G. Benotto. 2012. Identifying hypernyms in distributional semantic spaces. In *SEM
    Weeds Precision - as described in:
    J. Weeds and D. Weir. 2003. A general framework for distributional similarity. In EMNLP.
    """

    # Get the arguments
    args = docopt("""Compute invCL for a list of (x, y) pairs and save their scores.

    Usage:
        invCL.py <testset_file> <dsm_prefix> <output_file>

        <testset_file> = a file containing term-pairs, labels and relations, each line in the form 
                         of x\ty\tlabel\trelation
        <dsm_prefix> = the prefix for the pkl files for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tlabel\trelation\tscore,
                        where the score is invCL (for y as the hypernym of x).
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
                score = invCL(x_row, y_row)

            print >> f_out, '\t'.join((x, y, label, '%.5f' % score))


def invCL(x_row, y_row):
    """
    invCL(x --> y) = (\sqrt{clarkeDE(x -> y) \cdot (1-clarkeDE(y -> x)))})
    clarkeDE(x -> y) = (\sigma_c \in Fx,y min(w_x(c), w_y(c))) / (\sigma_c \in Fx w_x(c))
    """
    try:
        score = math.sqrt(clarkeDE(x_row, y_row)*(1-clarkeDE(y_row, x_row)))
    except:
        score = 0.0
    return score


if __name__ == '__main__':
    main()
