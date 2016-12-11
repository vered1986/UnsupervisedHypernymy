import sys
sys.path.append('../')

from slqs import *
from dsm_creation.common import *
  
            
def main():
    """
    Modification of SLQS - as described in:
    Santus, Enrico; Lu, Qin; Lenci, Alessandro; Schulte im Walde, Sabine. 2014. Chasing Hypernyms in Vector Spaces with Entropy. 
       Proceedings of the 14th Conference of the European Chapter of the Association of Computational Linguistics. 38-42.
      """

    # Get the arguments
    args = docopt("""Compute first order entropies for all rows of a vector space and save their scores.

    Usage:
        slqsrow_all.py (-f | -p | -l) <model> <output_file>
        
    Arguments:
        <testset_file> = a file containing term-pairs and labels, each line in the form of x\ty\tlabel
        <model> = the pkl file for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tscore, where the
                        score is SLQS (for y as the hypernym of x).
        
    Options:
        -f, --freq  calculate row entropies from frequency matrice
        -p, --ppmi  calculate row entropies from PPMI weighted matrice
        -l, --plmi  calculate row entropies from PLMI weighted matrice
        
    """)
    
    matrice_pkl = args['<model>']
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
    target_output = "measures/entropies/target_entropies/" + output_file_prefix + "_absolute_target_entropies" + ".txt"   
    args['target_output'] = target_output
    scaled_output = output_file_prefix + "_ranked_target_entropies" + ".txt"   

    

    # Receive a .pkl file
    cooc_space, mi_space, vocab_map, vocab_size, column_map, id2column_map = \
        get_space(matrice_folder, matrice_name, matrice_prefix, is_pmi, is_lmi, is_save_weighted)

    # Load the term-pairs
    targets = vocab_map

    # Get row entropies    
    r_entropies_dict, r_entr_ranked = get_r_entropies(targets, cooc_space, mi_space, target_output, is_freq)

    # Normalize the context entropy values to a value between 0 and 1 
    r_entropies = Min_max_scaling().scale(r_entropies_dict)     

    # Save the row entropies      
    save_entropies(r_entr_ranked, r_entropies, scaled_output)
    

if __name__ == '__main__':
    main()