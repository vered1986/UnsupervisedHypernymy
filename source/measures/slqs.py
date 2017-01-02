import os

from docopt import docopt
from os.path import basename
from slqs_module import *
               
            
def main():
    """
    SLQS - as described in:
    Santus, Enrico; Lu, Qin; Lenci, Alessandro; Schulte im Walde, Sabine. 2014. Chasing Hypernyms in Vector Spaces with Entropy. 
        Proceedings of the 14th Conference of the European Chapter of the Association of Computational Linguistics. 38-42.
    """
    args = docopt("""Compute SLQS for a list of (x, y) pairs and save their scores.

    Usage:
        slqs.py (-p | -l) (-f | -w) (-a | -m) <testset_file> <model> <N> <output_file>
        
    Arguments:
        <testset_file> = a file containing term-pairs, labels and relations, each line in the form of x\ty\tlabel\trelation
        <model> = the pkl file of the vector space
        <N> = for a target word the entropy of the N most associated contexts will be computed
        <output_file> = where to save the results: a tab separated file with x\ty\tlabel\trelation\tscore, where the
                        score is SLQS (for y as the hypernym of x).
        
    Options:
        -p, --ppmi  weight matrice with Ppmi
        -l, --plmi  weight matrice with Plmi
        -f, --freq  calculate context entropies from frequency matrice
        -w, --weighted  calculate context entropies from weighted matrice (with Ppmi, Plmi)
        -a, --average  calculate average of context entropies for target entropy
        -m, --median   calculate median of context entropies for target entropy
        
    """)
    
    # Get the arguments
    matrice_pkl = args['<model>']
    testset_file = args['<testset_file>']
    N = int(args['<N>'])
    output_file = args['<output_file>']
    is_freq = args['--freq']
    is_weighted = args['--weighted']    
    is_pmi = args['--ppmi']
    is_lmi = args['--plmi']
    is_average = args['--average']
    is_median = args['--median']
    #TODO: make is_save_weighted an argument of the script
    is_save_weighted = False
        
    matrice_name = os.path.splitext(basename(matrice_pkl))[0]
    matrice_folder = os.path.dirname(matrice_pkl) + "/"

    # Load the term-pairs
    targets, test_set = load_test_pairs(testset_file)

    # Receive a .pkl file
    cooc_space, mi_space, vocab_map, vocab_size, column_map, id2column_map = get_space(matrice_folder, matrice_name, is_pmi, is_lmi, is_save_weighted)
            
    # Get most associated columns for all targets
    most_associated_cols_dict, union_m_a_c = get_all_most_assoc_cols(mi_space, targets, vocab_map, N)

    # Assign context entropy file
    c_entrop_file = assign_c_entr_file(matrice_name, is_pmi, is_lmi, is_weighted)

    #TODO: transmit arguments more elegantly
    # Get context entropies
    c_entropies_dict, c_entr_ranked = get_c_entropies(targets, cooc_space, mi_space, N, c_entrop_file, vocab_map, id2column_map, most_associated_cols_dict, union_m_a_c, is_freq, is_weighted)

    # Scale the context entropy values to a value between 0 and 1 
    c_entropies = Min_max_scaling().scale(c_entropies_dict)     

    # Make relative target entropies
    unscored_output = make_relative_target_entropies(output_file, vocab_map, id2column_map, test_set, most_associated_cols_dict, c_entropies, N, is_average, is_median)
    
    # Compute SLQS for test tuples
    scored_output = score_slqs(unscored_output)

    # Save results
    save_results(scored_output, output_file)
 

if __name__ == '__main__':
    main()