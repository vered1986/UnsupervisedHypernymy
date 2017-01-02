import os

from docopt import docopt
from os.path import basename
from slqs_module import *
  
            
def main():
    """
    SLQS Row - as described in:
    Vered Shwartz, Enrico Santus, Dominik Schlechtweg. 2017. Hypernyms under Siege: Linguistically-motivated Artillery for Hypernymy Detection.
        Proceedings of the 15th Conference of the European Chapter of the Association of Computational Linguistics.
    """

    args = docopt("""Compute SLQS Row for a list of (x, y) pairs and save their scores.

    Usage:
        slqsrow.py (-f | -p | -l) <testset_file> <model> <output_file>
        
    Arguments:
        <testset_file> = a file containing term-pairs, labels and relations, each line in the form of x\ty\tlabel\trelation
        <model> = the pkl file for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tlabel\trelation\tscore, where the
                        score is SLQS Row (for y as the hypernym of x).
        
    Options:
        -f, --freq  calculate row entropies from frequency matrice
        -p, --ppmi  calculate row entropies from PPMI weighted matrice
        -l, --plmi  calculate row entropies from PLMI weighted matrice
        
    """)
    
    # Get the arguments
    matrice_pkl = args['<model>']
    testset_file = args['<testset_file>']
    output_file = args['<output_file>']
    is_freq = args['--freq']
    is_pmi = args['--ppmi']
    is_lmi = args['--plmi']
    is_save_weighted = False
    
    matrice_name = os.path.splitext(basename(matrice_pkl))[0]
    matrice_folder = os.path.dirname(matrice_pkl) + "/"
    

    # Load the term-pairs
    targets, test_set = load_test_pairs(testset_file)

    # Receive a .pkl file
    cooc_space, mi_space, vocab_map, vocab_size, column_map, id2column_map = get_space(matrice_folder, matrice_name, is_pmi, is_lmi, is_save_weighted)

    # Get row entropies    
    r_entropies_dict, r_entr_ranked = get_r_entropies(targets, cooc_space, mi_space, is_freq)

    # Scale the context entropy values to a value between 0 and 1 
    r_entropies = Min_max_scaling().scale(r_entropies_dict)     

    # Make unscored output
    unscored_output = make_unscored_output(r_entropies, test_set, vocab_map)
    
    # Compute SLQS for test tuples
    scored_output = score_slqs(unscored_output)

    # Save results
    save_results(scored_output, output_file)
    

if __name__ == '__main__':
    main()