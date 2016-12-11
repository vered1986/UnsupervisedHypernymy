import sys
sys.path.append('../')

from slqs import *
from dsm_creation.common import *
       
 
def main():
    """
    SLQS modified: SLQS_sub(x,y) = E(y) - E(x)
    """

    # Get the arguments
    args = docopt("""Compute SLQS_sub for a list of (x, y) pairs and save their scores.

    Usage:
        slqs_sub.py (-p | -l) (-f | -w) (-a | -m) <testset_file> <model> <N> <output_file>
        
    Arguments:
        <testset_file> = a file containing term-pairs and labels, each line in the form of x\ty\tlabel
        <model> = the pkl file for the vector space
        <N> = for a target word the entropy of the N most associated contexts will be computed
        <output_file> = where to save the results: a tab separated file with x\ty\tscore, where the
                        score is SLQS_sub (for y as the hypernym of x).
        
    Options:
        -p, --ppmi  weight matrice with Ppmi
        -l, --plmi  weight matrice with Plmi
        -f, --freq  calculate context entropies from frequency matrice
        -w, --weighted  calculate context entropies from weighted matrice (with Ppmi, Plmi)
        -a, --average  calculate average of context entropies for target entropy
        -m, --median   calculate median of context entropies for target entropy
        
    """)
    
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
    target_output = "measures/entropies/target_entropies/" + output_file_prefix + "_" + testset_file_prefix + "_" + \
                    testset_file_postfix + "_absolute_target_entropies" + ".txt"
    args['target_output'] = target_output


   # Load the term-pairs
    targets, test_set = load_test_pairs(testset_file)

    # Receive a .pkl file
    cooc_space, mi_space, vocab_map, vocab_size, column_map, id2column_map = \
        get_space(matrice_folder, matrice_name, matrice_prefix, is_pmi, is_lmi, is_save_weighted)
            
    # Get most associated columns for all targets
    most_associated_cols_dict, union_m_a_c = get_all_most_assoc_cols(mi_space, targets, vocab_map, N)

    # Assign context entropy file
    c_entrop_file = assign_c_entr_file(matrice_name, is_pmi, is_lmi, is_weighted)

    #TODO: transmit arguments more elegantly
    # Get context entropies
    c_entropies_dict, c_entr_ranked = get_c_entropies(targets, cooc_space, mi_space, N, c_entrop_file,
                                                      vocab_map, id2column_map, most_associated_cols_dict, union_m_a_c,
                                                      is_freq, is_weighted)

    # Normalize the context entropy values to a value between 0 and 1 
    c_entropies = Min_max_scaling().scale(c_entropies_dict)     

    # Make and save absolute target entropies
    cooc_space.make_absolute_target_entropies(args, targets, most_associated_cols_dict, c_entropies, test_set)
    
    # Make relative target entropies
    unscored_output = make_relative_target_entropies(output_file, vocab_map, id2column_map, test_set,
                                                     most_associated_cols_dict, c_entropies, N, is_average, is_median)
    
    # Compute target SLQS_sub for test tuples
    scored_output = score_slqs_sub(unscored_output)

    # Save results
    save_results(scored_output, output_file)
 

if __name__ == '__main__':
    main()