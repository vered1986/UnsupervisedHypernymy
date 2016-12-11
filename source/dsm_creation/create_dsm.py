from common import *
from docopt import docopt
from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.plmi_weighting import PlmiWeighting


def main():
    """
    Compute the FREQ/PPMI/PLMI matrix from a co-occurrence matrix, as default pickle the raw matrix
    """

    # Get the arguments
    args = docopt('''Compute the FREQ/PPMI/PLMI matrix from a co-occurrence matrix, as default pickle the raw matrix

    Usage:
        create_dsm.py <dsm_prefix> [-p | -l]

        <dsm_prefix> = the prefix for the input files (.sm for the matrix, .rows and .cols) and output files (.ppmi)
    
    Options:
    <none>      weight the matrice entries via FREQUENCY
    -p, --ppmi  weight the matrice entries via PPMI
    -l, --plmi  weight the matrice entries via PLMI
    
    ''')

    dsm_prefix = args['<dsm_prefix>']
    is_ppmi = args['--ppmi']
    is_plmi = args['--plmi']
    
    postfix = "_freq"


    # Create a space from co-occurrence counts in sparse format
    dsm = Space.build(data=dsm_prefix + '.sm',
                      rows=dsm_prefix + '.rows',
                      cols=dsm_prefix + '.cols',
                      format='sm')

    if is_ppmi:
        # Apply ppmi weighting
        dsm = dsm.apply(PpmiWeighting())
        postfix = "_ppmi"
    elif is_plmi:
        # Apply plmi weighting
        dsm = dsm.apply(PlmiWeighting())
        postfix = "_plmi"

    # Save the Space object in pickle format
    save_pkl_files(dsm_prefix + postfix, dsm)


if __name__ == '__main__':
    main()
