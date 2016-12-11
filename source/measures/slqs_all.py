import sys
sys.path.append('../')

from slqs import *
from dsm_creation.common import *

 
def main():
    """
     SLQS - as described in:
     Santus, Enrico; Lu, Qin; Lenci, Alessandro; Schulte im Walde, Sabine. 2014. Chasing Hypernyms in Vector Spaces with Entropy. 
     Proceedings of the 14th Conference of the European Chapter of the Association of Computational Linguistics. 38-42.
     """

    # Get the arguments
    args = docopt("""Compute second order entropies for all rows of a vector space and save their scores.

    Usage:
        slqs_all.py (-p | -l) (-f | -w) (-a | -m) <model> <N> <output_file>
        
    Arguments:
        <model> = the pkl file for the vector space
        <N> = for a target word the entropy of the N most associated contexts will be computed
        <output_file> = where to save the results: a tab separated file with x\ty\tscore, where the
                        score is SLQS (for y as the hypernym of x).
        
    Options:
        -p, --ppmi  weight matrice with Ppmi
        -l, --plmi  weight matrice with Plmi
        -f, --freq  calculate context entropies from frequency matrice
        -w, --weighted  calculate context entropies from weighted matrice (with Ppmi, Plmi)
        -a, --average  calculate average of context entropies for target entropy
        -m, --median   calculate median of context entropies for target entropy
        
    """)
    
    matrice_pkl = args['<model>']
    N = int(args['<N>'])
    output_file = args['<output_file>']
    is_freq = args['--freq']
    is_weighted = args['--weighted']    
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
    target_output = "measures/entropies/target_entropies/" + output_file_prefix + "_absolute_target_entropies" + ".txt"   
    args['target_output'] = target_output
    

    # Receive a .pkl file
    cooc_space, mi_space, vocab_map, vocab_size, column_map, id2column_map = get_space(matrice_folder, matrice_name,
                                                                                       matrice_prefix, is_pmi, is_lmi,
                                                                                       is_save_weighted)
     
   # Load the vocab
    targets = vocab_map
       
    # Get most associated columns for all targets
    most_associated_cols_dict, union_m_a_c = get_all_most_assoc_cols(mi_space, targets, vocab_map, N)

    # Assign context entropy file
    c_entrop_file = assign_c_entr_file(matrice_name, is_pmi, is_lmi, is_weighted)

    # Get context entropies
    c_entropies_dict, c_entr_ranked = get_c_entropies(targets, cooc_space, mi_space, N, c_entrop_file, vocab_map,
                                                      id2column_map, most_associated_cols_dict, union_m_a_c, is_freq,
                                                      is_weighted)

    # Normalize the context entropy values to a value between 0 and 1 
    c_entropies = Min_max_scaling().scale(c_entropies_dict)

    # Make and save absolute target entropies
    cooc_space.make_absolute_target_entropies(args, targets, most_associated_cols_dict, c_entropies, [])
 

if __name__ == '__main__':
    main()