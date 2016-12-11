import sys
sys.path.append('../')

import os
import math
import codecs

from docopt import docopt
from os.path import basename
from dsm_creation.common import *
from statistics import median, mean
from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.plmi_weighting import PlmiWeighting


class Min_max_scaling(object):
    
    def scale(self, raw_dict):

        def min_max(minimum, maximum, X):
            X_norm = (X - minimum) / (maximum - minimum)
            return X_norm
        
        print "Scaling context entropy values..."
        
        # Get the minimum and maximum value
        min_key, min_value = min(raw_dict.iteritems(), key=lambda x:x[1])
        max_key, max_value = max(raw_dict.iteritems(), key=lambda x:x[1])

        # Create the scaled dictionary
        scaled_dict = {key : min_max(min_value, max_value, value) for key, value in raw_dict.items()}

        return scaled_dict 


class Space_extension(Space):
    
    def get_vocab(self):
        
        vocab_map = self.get_row2id()
            
        return vocab_map
        
        
    def get_columns(self):
        
        column_map = self.get_column2id()
            
        return column_map
        
        
    def get_id2column_map(self):
        
        id_map = self.get_id2column()
            
        return id_map
        

    def get_most_associated_cols(self, target, N):
        
        row = self.get_row(target)

        # Data returns the non-zero elements in the row and indices returns the indices of the non-zero elements
        data = row.get_mat().data
        indices = row.get_mat().indices

        most_associated_cols_indices = data.argsort()[-N:]
        most_associated_cols = { indices[index] : data[index] for index in most_associated_cols_indices }

        return most_associated_cols
        

    def compute_row_entropies(self, targets):

        targets_size = len(targets)
        vocab_map = self.get_row2id()
        
        # Iterate over rows
        r_entropies_dict = {}
        print "Iterating over rows..."
        for j, w in enumerate(targets):
            
            if w not in vocab_map:
                continue

            print "%d%% done..." % (j*100/targets_size)            
            row = self.get_row(w)

            # Get all counts in column (non-zero elements)
            counts = row.get_mat().data

            # Get sum of column (total count of context)
            r_total_freq = row.get_mat().sum()

            # Compute entropy of context
            H = -sum([((count/r_total_freq) * math.log((count/r_total_freq),2)) for count in counts])

            r_entropies_dict[w] = H
        
        return r_entropies_dict        
        

    def compute_context_entropies(self, N, union_m_a_c):

        union_m_a_c_size = len(union_m_a_c) 
        id2row = self.id2row
        id2column = self.id2column

        # Transpose matrix to iterate over columns
        print "Transposing the matrix..."
        matrix_transposed = self.cooccurrence_matrix.transpose()
        # Instantiate a new space from transposed matrix
        space_transposed = Space_extension(matrix_transposed, id2column, id2row)

        
        c_entropies_dict = {} 
        # Iterate over columns (contexts) 
        print "Iterating over columns..."
        for j, column_id in enumerate(union_m_a_c):
            context = id2column[column_id]

            print "%d%% done..." % (j*100/union_m_a_c_size)            
            col = space_transposed.get_row(context)

            # Get all counts in column (non-zero elements)
            counts = col.get_mat().data

            # Get sum of column (total count of context)
            c_total_freq = col.get_mat().sum()

            # Compute entropy of context
            H = -sum([((count/c_total_freq) * math.log((count/c_total_freq),2)) for count in counts])

            c_entropies_dict[context] = H
        
        return c_entropies_dict
        
        
    def make_absolute_target_entropies(self, args, targets, most_associated_cols_dict, c_entropies, test_set):
    
        N = int(args['<N>'])    
    
        is_average = args['--average']
        is_median = args['--median']         
        target_output = args['target_output']
        
        targets_size = len(targets)     
        vocab_map = self.get_vocab()
        id2column_map = self.get_id2column_map()
        
        print "Computing target entropies..."
        i = 0
        target_entropies = {}    
        # Compute target entropies for all target rows
        for target in targets:
            if target not in vocab_map:
                continue
            
            print "%d%% done..." % (i*100/targets_size)
            
            # N most associated contexts of target
            most_associated_cs = [id2column_map[mapping[0]] for mapping in most_associated_cols_dict[target][:N]]
                                      
            print "Most associated contexts of " + target
            print most_associated_cs
            
            if not len(most_associated_cs) > 0:
                target_entropies[target] = -999.0
                continue
    
            # Get the entropies of the most associated contexts
            entr_of_most_assoc_cs = [float(c_entropies[context]) for context in most_associated_cs]
            
            # Compute the average or median of the entropies of the most associated contexts (target entropy)
            if is_average:
                target_entropy = float(mean(entr_of_most_assoc_cs))
            elif is_median:
                target_entropy = float(median(entr_of_most_assoc_cs))
                   
            target_entropies[target] = target_entropy
            
            i += 1
        
        # Rank the target entropies
        target_entropies_ranked = sorted(target_entropies, key=lambda x: -(target_entropies[x]))

        # Save the target entropies for maximal number of associated columns <= N
        print "Writing target entropies to %s..." % target_output
        with open(target_output, 'w') as f_out:
            for target in target_entropies_ranked:
                H = target_entropies[target]
                print >> f_out, "\t".join((target, str(H)))
        

        # Prepare output            
        unscored_output = []        
        for (x, y, label, relation) in test_set:
            if x not in vocab_map or y not in vocab_map:
                # Assign a special score to out-of-vocab pairs
                unscored_output.append((x, y, label, relation, -999.0, -999.0))
                continue 
        
            unscored_output.append((x, y, label, relation, target_entropies[x], target_entropies[y]))
            
                    
        return unscored_output
         
         
            
def main():
    """
    SLQS - as described in:
    Santus, Enrico; Lu, Qin; Lenci, Alessandro; Schulte im Walde, Sabine. 2014. Chasing Hypernyms in Vector Spaces with Entropy. 
       Proceedings of the 14th Conference of the European Chapter of the Association of Computational Linguistics. 38-42.
    """

    # Get the arguments
    args = docopt("""Compute SLQS for a list of (x, y) pairs and save their scores.

    Usage:
        slqs.py (-p | -l) (-f | -w) (-a | -m) <testset_file> <model> <N> <output_file>
        
    Arguments:
        <testset_file> = a file containing term-pairs and labels, each line in the form of x\ty\tlabel
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
    target_output = "measures/entropies/target_entropies/" + output_file_prefix + \
                    "_" + testset_file_prefix + "_" + testset_file_postfix + "_absolute_target_entropies" + ".txt"
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
    c_entropies_dict, c_entr_ranked = get_c_entropies(targets, cooc_space, mi_space, N, c_entrop_file, vocab_map,
                                                      id2column_map, most_associated_cols_dict, union_m_a_c,
                                                      is_freq, is_weighted)

    # Normalize the context entropy values to a value between 0 and 1 
    c_entropies = Min_max_scaling().scale(c_entropies_dict)     

    #TODO: split calc of entropies and preparing the output
    # Make and save absolute target entropies
    cooc_space.make_absolute_target_entropies(args, targets, most_associated_cols_dict, c_entropies, test_set)
    
    # Make relative target entropies
    unscored_output = make_relative_target_entropies(output_file, vocab_map, id2column_map,
                                                     test_set, most_associated_cols_dict, c_entropies, N,
                                                     is_average, is_median)
    
    # Compute target SLQS for test tuples
    scored_output = score_slqs(unscored_output)

    # Save results
    save_results(scored_output, output_file)
 

def slqs(x_entr, y_entr):
    
    score = 1 - (x_entr/y_entr) if y_entr != 0.0 else -1.0  
    
    return score


def slqs_sub(x_entr, y_entr):
    
    score = y_entr - x_entr
    
    return score
    
    
def load_test_pairs(testset_file):
    
    print "Loading test pairs..."
    with codecs.open(testset_file) as f_in:
        test_set = [tuple(line.strip().split('\t')) for line in f_in]
        
    targets = get_targets(test_set)
    
    return targets, test_set
    

def get_targets(test_set):
    
    xs, ys, labels, relation = zip(*test_set)
    union = set(xs) | set(ys)
    return union
    

def get_space(matrice_folder, matrice_name, matrice_prefix, is_pmi, is_lmi, is_save_weighted):
    
    
    try:
        print "Loading frequency matrice..."
        cooc_space = load_pkl_files(matrice_folder + matrice_name)
        cooc_space.__class__ = Space_extension
    except IOError:
        print "Format not suitable or file does not exist"
       
    mi_space = []   
       
    if is_pmi:
        try:
            mi_space = load_pkl_files(matrice_folder + matrice_prefix + "_ppmi")
            print "Found Ppmi weighted matrice."
        except:
            print "No Ppmi weighted matrice found."
            print "Building Ppmi weighted matrice..."
            mi_space = cooc_space.apply(PpmiWeighting())
            if is_save_weighted:
                print "Saving Ppmi weighted matrice..."
                save_pkl_files(matrice_folder + matrice_prefix + "_ppmi", mi_space, False)
            
        mi_space.__class__ = Space_extension

                
    if is_lmi:
        try:
            mi_space = load_pkl_files(matrice_folder + matrice_prefix + "_plmi")
            print "Found Plmi weighted matrice."
        except:
            print "No Plmi weighted matrice found."
            print "Building Plmi weighted matrice..."
            mi_space = cooc_space.apply(PlmiWeighting())
            if is_save_weighted:
                print "Saving Plmi weighted matrice..."
                save_pkl_files(matrice_folder + matrice_prefix + "_plmi", mi_space, False)
        
        mi_space.__class__ = Space_extension

    
    vocab_map = cooc_space.get_vocab()
    vocab_size = len(vocab_map)
    column_map = cooc_space.get_columns()
    id2column_map = cooc_space.get_id2column_map()
    
    print "The vocabulary has size: " + str(vocab_size)
    
    return cooc_space, mi_space, vocab_map, vocab_size, column_map, id2column_map
    
    
def get_all_most_assoc_cols(mi_space, targets, vocab_map, N):
    
    print "Getting most associated columns for all targets..."
    most_associated_cols_dict = {}
    union_m_a_c = set()
    for target in targets:
        if target not in vocab_map:
            continue
        
        # Get most associated columns for target
        most_associated_cols = mi_space.get_most_associated_cols(target, N) 
        union_m_a_c = union_m_a_c | set(most_associated_cols)
        most_associated_cols_sorted = sorted(most_associated_cols.iteritems(), key=lambda (k,v): (v,k), reverse=True)
        most_associated_cols_dict[target] = most_associated_cols_sorted
    
    return most_associated_cols_dict, union_m_a_c
    
    
def assign_c_entr_file(matrice_name, is_pmi, is_lmi, is_weighted):
    
    if is_weighted:
        if is_pmi:
            c_entrop_file = "measures/entropies/context_entropies/" + matrice_name + "_ppmi" + "_context_entropies" + ".txt" 
        if is_lmi:
            c_entrop_file = "measures/entropies/context_entropies/" + matrice_name + "_plmi" + "_context_entropies" + ".txt" 
    if not is_weighted:
        c_entrop_file = "measures/entropies/context_entropies/" + matrice_name  + "_freq" + "_context_entropies" + ".txt"

    return c_entrop_file
    

def save_entropies(entr_ranked, entropies_dict, entrop_file):
    print "Writing raw entropies to %s..." % entrop_file                
    with open(entrop_file, 'w') as f_out:
        for context in entr_ranked:
            H = entropies_dict[context]
            print >> f_out, "\t".join((context, str(H)))
                
                
def get_r_entropies(targets, cooc_space, mi_space, target_output, is_freq):
    
    # Build row entropies file if non-existent
    if is_freq:
        print "Computing row entropies from co-occurence matrice..."
        r_entropies_dict = cooc_space.compute_row_entropies(targets)
        print "Calculated entropies for %d rows." % len(r_entropies_dict)
    else:
        print "Computing row entropies from weighted matrice..."
        r_entropies_dict = mi_space.compute_row_entropies(targets)
        print "Calculated entropies for %d rows." % len(r_entropies_dict)
            
    # Rank the row entropies
    r_entr_ranked = sorted(r_entropies_dict, key=lambda x: -(float(r_entropies_dict[x])))
    
    # Save the row entropies      
    save_entropies(r_entr_ranked, r_entropies_dict, target_output)
            
    return r_entropies_dict, r_entr_ranked
    

def get_c_entropies(targets, cooc_space, mi_space, N, c_entrop_file, vocab_map, id2column_map,
                    most_associated_cols_dict, union_m_a_c, is_freq, is_weighted):
    
    # Try to get context entropy file
    try:
        with open(c_entrop_file) as f_in:
            c_entropies_dict = dict([[line.strip().split("\t")[0], float(line.strip().split("\t")[1])] for line in f_in])
            print "Found context entropy file: " + c_entrop_file

            # Get new contexts
            new_union_m_a_c = set()
            for target in targets:
                if target not in vocab_map:
                    continue
                for mapping in most_associated_cols_dict[target]:
                    col_id = int(mapping[0]) 
                    context = id2column_map[col_id]
                    if not context in c_entropies_dict:
                        new_union_m_a_c = new_union_m_a_c | set([col_id])
                         
            if len(new_union_m_a_c) > 0:
                if is_freq:
                    print "Computing new context entropies from co-occurence matrice..."
                    new_c_entropies_dict = cooc_space.compute_context_entropies(N, new_union_m_a_c)
                elif is_weighted:
                    print "Computing new context entropies from weighted matrice..."
                    new_c_entropies_dict = mi_space.compute_context_entropies(N, new_union_m_a_c)               
                # Add the new context entropies to the old ones
                print "Calculated entropies for %d new contexts." % len(new_c_entropies_dict)
                c_entropies_dict.update(new_c_entropies_dict)  
                
    except IOError:
        print "No context entropy file found."
        # Build context entropy file if non-existent
        if is_freq:
            print "Computing context entropies instead from co-occurence matrice..."
            c_entropies_dict = cooc_space.compute_context_entropies(N, union_m_a_c)
            print "Calculated entropies for %d contexts." % len(c_entropies_dict)
        elif is_weighted:
            print "Computing context entropies instead from weighted matrice..."
            c_entropies_dict = mi_space.compute_context_entropies(N, union_m_a_c)
            print "Calculated entropies for %d contexts." % len(c_entropies_dict)
            
    # Rank the context entropies
    c_entr_ranked = sorted(c_entropies_dict, key=lambda x: -(float(c_entropies_dict[x])))
    
    # Save the (updated) context entropies      
    save_entropies(c_entr_ranked, c_entropies_dict, c_entrop_file)
            
    return c_entropies_dict, c_entr_ranked
        

def make_relative_target_entropies(output_file, vocab_map, id2column_map, test_set,
                                   most_associated_cols_dict, c_entropies, N, is_average, is_median):
    
    unscored_output = []    
    
    for (x, y, label, relation) in test_set:
        if x not in vocab_map or y not in vocab_map:
            # Assign a special score to out-of-vocab pairs
            unscored_output.append((x, y, label, relation, -999.0, -999.0))
            continue
        
        print x, y
        
        # Get smaller number M of associated columns
        M = min([len(most_associated_cols_dict[x]), len(most_associated_cols_dict[y])])
        
        if M == 0:
            unscored_output.append((x, y, label, relation, -999.0, -999.0))
            continue
 
        target_entropies = {}
        
        # Compute Generality Index for x and y
        for var in (x, y): 

            m_most_assoc_cs = {}

            # M Most associated contexts of x and y
            m_most_assoc_cs[var] = [id2column_map[mapping[0]] for mapping in most_associated_cols_dict[var][:M]]
                                      
            #print "M Most associated contexts of " + var
            #print m_most_assoc_cs[var]

            entr_of_m_most_assoc_cs = {}    

            # Get the M entropies of the most associated contexts of x and y
            entr_of_m_most_assoc_cs[var] = [float(c_entropies[context]) for context in m_most_assoc_cs[var]]
            
            # Compute the average or median of the entropies of the M most associated contexts (target entropy)
            if is_average:
                target_entropies[var] = float(mean(entr_of_m_most_assoc_cs[var]))
            elif is_median:
                target_entropies[var] = float(median(entr_of_m_most_assoc_cs[var]))
                
            print float(median(entr_of_m_most_assoc_cs[var]))
        
        unscored_output.append((x, y, label, relation, target_entropies[x], target_entropies[y]))
    
    return unscored_output


def make_unscored_output(entropies, test_set, vocab_map):
    
    unscored_output = []    
    
    for (x, y, label, relation) in test_set:
        if x not in vocab_map or y not in vocab_map:
            # Assign a special score to out-of-vocab pairs
            unscored_output.append((x, y, label, relation, -999.0, -999.0))
            continue
        
        print (x, y, label, relation, entropies[x], entropies[y])
    
        unscored_output.append((x, y, label, relation, entropies[x], entropies[y]))
    
    return unscored_output


def score_slqs(rel_tar_entrs):

    scored_output = []

    print "Computing target SLQS for test tuples..."
    
    for (x, y, label, relation, xentr, yentr) in rel_tar_entrs:
        
        if xentr == -999.0 or yentr == -999.0:
            scored_output.append((x, y, label, relation, -999.0))
            continue
        # Compute slqs for y being the hypernym of x
        score = slqs(xentr, yentr)            
    
        scored_output.append((x, y, label, relation, score))
        
    return scored_output
    

def score_slqs_sub(rel_tar_entrs):

    scored_output = []

    print "Computing target SLQS_sub for test tuples..."
    
    for (x, y, label, relation, xentr, yentr) in rel_tar_entrs:
        
        if xentr == -999.0 or yentr == -999.0:
            scored_output.append((x, y, label, relation, -999.0))
            continue
        
        # Compute slqs_sub for y being the hypernym of x
        score = slqs_sub(xentr, yentr)            
    
        scored_output.append((x, y, label, relation, score))
        
    return scored_output
    

def save_results(scored_output, output_file):
    
    with codecs.open(output_file, 'w') as f_out:
            
        for (x, y, label, relation, score) in scored_output:
    
            print >> f_out, '\t'.join((x, y, label, relation, '%.5f' % score))
    
    print "Saved the results to " + output_file
    
    

if __name__ == '__main__':
    main()