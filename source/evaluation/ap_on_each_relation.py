import codecs

import numpy as np

from docopt import docopt
from sklearn.metrics import average_precision_score


def main():
    """
    Train a classifier based on all the measures, to discriminate hypernymy from one other single relation.
    """

    # Get the arguments
    args = docopt("""Calculate the Average Precision (AP) at k for every hyper-other relation in the dataset.

    Usage:
        ap_on_each_relation.py <test_results_file> <test_set_file> <k>

        <test_results_file> = the test set result file.
        <test_set_file> = the test set containing the original relations.
        <k> = the cutoff; if it is equal to zero, all the rank is considered.
    """)

    test_set_file = args['<test_set_file>']
    test_results_file = args['<test_results_file>']
    cutoff = int(args['<k>'])

    # Load the test set
    print 'Loading the dataset...'
    test_set, relations = load_dataset(test_set_file + '.test')
    hyper_relation = 'hyper'

    for other_relation in [relation for relation in relations if relation != hyper_relation]:

        curr_relations = [other_relation, hyper_relation]
        print '=================================================='
        print 'Testing', hyper_relation, 'vs.', other_relation, '...'

        # Filter out the dataset to contain only these two relations
        relation_index = { relation : index for index, relation in enumerate(curr_relations) }
        curr_test_set = { (x, y) : relation for (x, y), relation in test_set.iteritems() if relation in curr_relations }

        # Sort the lines in the file in descending order according to the score
        with codecs.open(test_results_file, 'r', 'utf-8') as f_in:
            dataset = [tuple(line.strip().split('\t')) for line in f_in]
            dataset = [(x, y, label, float(score)) for (x, y, label, score) in dataset if (x, y) in curr_test_set]

        dataset = sorted(dataset, key=lambda line: line[-1], reverse=True)

        # relevance: rel(i) is an indicator function equaling 1 if the item at rank i is a hypernym
        gold = np.array([1 if label == 'True' else 0 for (x, y, label, score) in dataset])
        scores = np.array([score for (x, y, label, score) in dataset])

        for i in range(1, min(cutoff + 1, len(dataset))):
            score = average_precision_score(gold[:i], scores[:i])
            print 'Average Precision at %d is %.3f' % (i, 0 if score == -1 else score)

        print 'FINAL: Average Precision at %d is %.3f' % (len(dataset), average_precision_score(gold, scores))


def load_dataset(dataset_file, relations=None):
    """
    Loads a dataset file
    :param dataset_file: the file path
    :return: a list of dataset instances, (x, y, relation)
    """
    with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
        dataset = [tuple(line.strip().split('\t')) for line in f_in]

        if not relations:
            if len(dataset[0]) == 4:
                xs, ys, labels, relations = zip(*dataset)
            else:
                xs, ys, labels, _, relations = zip(*dataset)

        dataset = { (item[0], item[1]) : item[-1] for item in dataset if item[-1] in relations }

    return dataset, set(relations)


if __name__ == '__main__':
    main()