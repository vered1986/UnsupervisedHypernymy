import sys
sys.path.append('../')

from slqs import *
from dsm_creation.common import *
  
            
def main():
    """
    SLQS modified: SLQS_sub(x,y) = E(y) - E(x)
    """

    # Get the arguments
    args = docopt("""Compute entropies for a list of (x, y) pairs and save their scores. Each word's entropy corresponds to
                        the entropy of its row (instead of the median of the entropies of its most associated columns as
                        in the paper mentioned above).

    Usage:
        slqsrow_sub.py (-f | -p | -l) <testset_file> <model> <output_file>
        
    Arguments:
        <testset_file> = a file containing term-pairs and labels, each line in the form of x\ty\tlabel
        <model> = the pkl file for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tscore, where the
                        score is SLQS_sub (for y as the hypernym of x).
        
    Options:
        -f, --freq  calculate row entropies from frequency matrice
        -p, --ppmi  calculate row entropies from PPMI weighted matrice
        -l, --plmi  calculate row entropies from PLMI weighted matrice
        
    """)
    
    matrice_pkl = args['<model>']
    testset_file = args['<testset_file>']
    output_file = args['<output_file>']
    is_freq = args['--freq']
    is_pmi = args['--ppmi']
    is_lmi = args['--plmi']
    is_save_weighted = False
    
    matrice_name = os.path.splitext(basename(matrice_pkl))[0]
    matrice_prefix = matrice_name.split("_")[0]
    args['matrice_prefix'] = matrice_prefix
    matrice_folder = os.path.dirname(matrice_pkl) + "/"
    args['matrice_folder'] = matrice_folder
    output_file_prefix = os.path.splitext(basename(output_file))[0]
    args['output_file_prefix'] = output_file_prefix
    testset_file_prefix = os.path.splitext(basename(testset_file))[0].split("_")[0]
    args['testset_file_prefix'] = testset_file_prefix
    testset_file_postfix = testset_file.split(".")[1]
    args['testset_file_postfix'] = testset_file_postfix
    target_output = "measures/entropies/target_entropies/" + output_file_prefix + "_" + testset_file_prefix + \
                    "_" + testset_file_postfix + "_absolute_target_entropies" + ".txt"
    args['target_output'] = target_output
    

    # Load the term-pairs
    targets, test_set = load_test_pairs(testset_file)

    # Receive a .pkl file
    cooc_space, mi_space, vocab_map, vocab_size, column_map, id2column_map = \
        get_space(matrice_folder, matrice_name, matrice_prefix, is_pmi, is_lmi, is_save_weighted)

    # Get row entropies    
    r_entropies_dict, r_entr_ranked = get_r_entropies(targets, cooc_space, mi_space, target_output, is_freq)

    # Normalize the context entropy values to a value between 0 and 1 
    r_entropies = Min_max_scaling().scale(r_entropies_dict)     

    # Make unscored output
    unscored_output = make_unscored_output(r_entropies, test_set, vocab_map)
    
    # Compute target SLQS for test tuples
    scored_output = score_slqs_sub(unscored_output)

    # Save results
    save_results(scored_output, output_file)
    

if __name__ == '__main__':
    main()