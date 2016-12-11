import os
import gzip
import pickle
import numpy as np

from collections import defaultdict
from composes.utils import io_utils
from composes.semantic_space.space import Space
from scipy.sparse import coo_matrix, csr_matrix
from composes.matrix.sparse_matrix import SparseMatrix


def get_sentences(corpus_file):
    """
    Returns all the (content) sentences in a corpus file
    :param corpus_file: the corpus file
    :return: the next sentence (yield)
    """

    # Read all the sentences in the file
    with gzip.open(corpus_file, 'r') as f_in:

        s = []

        for line in f_in:
            line = line.decode('ISO-8859-2')

            # Ignore start and end of doc
            if '<text' in line or '</text' in line or '<s>' in line:
                continue
            # End of sentence
            elif '</s>' in line:
                yield s
                s = []
            else:
                try:
                    word, lemma, pos, index, parent, dep = line.split()
                    s.append((word, lemma, pos, int(index), int(parent), dep))
                # One of the items is a space - ignore this token
                except:
                    continue


def save_files(cooc_mat, frequent_contexts, output_prefix):
    """
    Saves the .sm, .rows and .cols files
    :param cooc_mat: the co-occurrence matrix
    :param frequent_contexts: the frequent contexts
    :param output_prefix: the prefix for the output files: .sm sparse matrix output file, .rows and .cols
    :return:
    """

    # Print in a sparse matrix format
    with open(output_prefix + '.sm', 'w') as f_out:
        for target, contexts in cooc_mat.iteritems():
            for context, freq in contexts.iteritems():
                if context in frequent_contexts:
                    print >> f_out, ' '.join((target, context, str(freq)))

    # Save the contexts as columns
    with open(output_prefix + '.cols', 'w') as f_out:
        for context in frequent_contexts:
            print >> f_out, context

    # Save the targets as rows
    with open(output_prefix + '.rows', 'w') as f_out:
        for target in cooc_mat.keys():
            print >> f_out, target


def filter_contexts(cooc_mat, min_occurrences):
    """
    Returns the contexts that occurred at least min occurrences times
    :param cooc_mat: the co-occurrence matrix
    :param min_occurrences: the minimum number of occurrences
    :return: the frequent contexts
    """

    context_freq = defaultdict(int)
    for target, contexts in cooc_mat.iteritems():
        for context, freq in contexts.iteritems():
            context_freq[context] = context_freq[context] + freq

    frequent_contexts = set([context for context, frequency in context_freq.iteritems() if frequency >= min_occurrences])
    return frequent_contexts


def save_pkl_files(dsm_prefix, dsm, save_in_one_file=False):
    """
    Save the space to separate pkl files
    :param dsm_prefix:
    :param dsm:
    :return:
    """
    # Save in a single file (for small spaces)
    if save_in_one_file:
        io_utils.save(dsm, dsm_prefix + '.pkl')

    # Save in multiple files: npz for the matrix and pkl for the other data members of Space
    else:
        mat = coo_matrix(dsm.cooccurrence_matrix.get_mat())
        np.savez_compressed(dsm_prefix + 'cooc.npz', data=mat.data, row=mat.row, col=mat.col, shape=mat.shape)

        with open(dsm_prefix + '_row2id.pkl', 'wb') as f_out:
            pickle.dump(dsm._row2id, f_out, 2)

        with open(dsm_prefix + '_id2row.pkl', 'wb') as f_out:
            pickle.dump(dsm._id2row, f_out, 2)

        with open(dsm_prefix + '_column2id.pkl', 'wb') as f_out:
            pickle.dump(dsm._column2id, f_out, 2)

        with open(dsm_prefix + '_id2column.pkl', 'wb') as f_out:
            pickle.dump(dsm._id2column, f_out, 2)


def load_pkl_files(dsm_prefix):
    """
    Load the space from either a single pkl file or numerous files
    :param dsm_prefix:
    :param dsm:
    :return:
    """
    # Check whether there is a single pickle file for the Space object
    if os.path.isfile(dsm_prefix + '.pkl'):
        return io_utils.load(dsm_prefix + '.pkl')

    # Load the multiple files: npz for the matrix and pkl for the other data members of Space
    with np.load(dsm_prefix + 'cooc.npz') as loader:
        coo = coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])

    cooccurrence_matrix = SparseMatrix(csr_matrix(coo))

    with open(dsm_prefix + '_row2id.pkl', 'rb') as f_in:
        row2id = pickle.load(f_in)

    with open(dsm_prefix + '_id2row.pkl', 'rb') as f_in:
        id2row = pickle.load(f_in)

    with open(dsm_prefix + '_column2id.pkl', 'rb') as f_in:
        column2id = pickle.load(f_in)

    with open(dsm_prefix + '_id2column.pkl', 'rb') as f_in:
        id2column = pickle.load(f_in)

    return Space(cooccurrence_matrix, id2row, id2column, row2id=row2id, column2id=column2id)
