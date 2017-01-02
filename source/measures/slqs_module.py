"""
This module is a collection of classes and methods useful for calculating, saving and
evaluating information-theoretic measures such as entropy (SLQS and variations) as 
described in:
    
    Vered Shwartz, Enrico Santus, Dominik Schlechtweg. 2017. Hypernyms under Siege: 
        Linguistically-motivated Artillery for Hypernymy Detection. Proceedings of 
        the 15th Conference of the European Chapter of the Association of Computational 
        Linguistics.
"""

import sys
sys.path.append('../')

import math
import codecs

from dsm_creation.common import *
from statistics import median, mean
from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.plmi_weighting import PlmiWeighting
       

#TODO: make  this class a method
class Min_max_scaling(object):    
    """
    This class implements Min-Max-Scaling.
    """
    
    def scale(self, raw_dict):
        """
        Scales values of a dictionary using Min-Max-Scaling
        :param raw_dict: the dictionary
        :return: scaled dictionary
        """

        def min_max(minimum, maximum, X):
            """
            Scales value using Min-Max-Scaling
            :param minimum: scale minimum
            :param maximum: scale maximum
            :param X: value
            :return: scaled value
            """
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
    """
    This class extends the Space class implementing semantic spaces.
    
    The extension mainly consists in methods useful for calculating
    information-theoretic measures such as entropy (SLQS and variations).
    """
    
    def get_vocab(self):
        """
        Gets the mapping of row strings to integer ids for a semantic space.
        :param self: a semantic space
        :return: dictionary that maps row strings to integer ids
        """
        
        vocab_map = self.get_row2id()
            
        return vocab_map
        
        
    def get_columns(self):
        """
        Gets the mapping of column strings to integer ids for a semantic space.
        :param self: a semantic space
        :return: dictionary that maps column strings to integer ids
        """
        
        column_map = self.get_column2id()
            
        return column_map
        
        
    def get_id2column_map(self):
        """
        Gets the column elements for a semantic space.
        :param self: a semantic space
        :return: list of strings, the column elements
        """
        
        id_map = self.get_id2column()
            
        return id_map
        

    def get_most_associated_cols(self, target, N):
        """
        Gets N columns with highest values for a row string (target).
        :param self: a semantic space
        :param target: target row string
        :param N: int, number of columns to extract
        :return: dictionary, mapping column ids to their values
        """
        
        row = self.get_row(target)

        # Data returns the non-zero elements in the row and indices returns the indices of the non-zero elements
        data = row.get_mat().data
        indices = row.get_mat().indices

        most_associated_cols_indices = data.argsort()[-N:]
        most_associated_cols = { indices[index] : data[index] for index in most_associated_cols_indices }

        return most_associated_cols
        

    def compute_row_entropies(self, targets):
        """
        Computes row entropy for a list of target row strings.
        :param self: a semantic space
        :param targets: list of strings, the targets
        :return: dictionary, mapping targets to entropy values
        """

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
        

    def compute_context_entropies(self, union_m_a_c):
        """
        Computes entropies for a set of column strings.
        :param self: a semantic space
        :param union_m_a_c: set of column ids
        :return: dictionary, mapping column strings to entropy values
        """

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
        
        
         
def slqs(x_entr, y_entr):
    """
    Computes SLQS score from two entropy values.
    :param x_entr: entropy value of x
    :param y_entr: entropy value of y
    :return: SLQS score
    """
    
    score = 1 - (x_entr/y_entr) if y_entr != 0.0 else -1.0  
    
    return score


def slqs_sub(x_entr, y_entr):
    """
    Computes SLQS Sub score from two entropy values.
    :param x_entr: entropy value of x
    :param y_entr: entropy value of y
    :return: SLQS Sub score
    """
    
    score = y_entr - x_entr
    
    return score
    
    
def load_test_pairs(testset_file):
    """
    Loads target tuples from a test file.
    :param testset_file: test file with each line a test item
    :return targets: set of target strings
    :return test_set: list of tuples, each tuple a test item
    """
    
    print "Loading test pairs..."
    with codecs.open(testset_file) as f_in:
        test_set = [tuple(line.strip().split('\t')) for line in f_in]
        
    targets = get_targets(test_set)
    
    return targets, test_set
    

def get_targets(test_set):
    """
    Get set of elements in index position 0 and 1 from each tuple from list.
    :param test_set: list of tuples, each tuple a test item
    :return union: set, union of targets in index position 0 and 1
    """
    
    xs, ys, labels, relation = zip(*test_set)
    union = set(xs) | set(ys)
    return union
    

def get_space(matrice_folder, matrice_name, is_pmi, is_lmi, is_save_weighted):
    """
    Loads semantic space from matrix file.
    :param matrice_folder: string, path of matrice folder
    :param matrice_name: string, name of matrice file
    :param is_pmi: boolean, whether to weight matrice with PPMI values
    :param is_lmi: boolean, whether to weight matrice with PLMI values
    :param is_save_weighted: boolean, whether to save weighted matrice
    :return cooc_space: unweighted semantic space
    :return mi_space: weighted semantic space
    :return vocab_map: dictionary that maps row strings to integer ids
    :return vocab_size: int, number of rows
    :return column_map: dictionary that maps column strings to integer ids
    :return id2column_map: list of strings, the column elements
    """    
    
    try:
        print "Loading frequency matrice..."
        cooc_space = load_pkl_files(matrice_folder + matrice_name)
        # Change class of space to apply information-theoretic measures
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
        
        # Change class of space to apply information-theoretic measures
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
        
        # Change class of space to apply information-theoretic measures
        mi_space.__class__ = Space_extension

    
    vocab_map = cooc_space.get_vocab()
    vocab_size = len(vocab_map)
    column_map = cooc_space.get_columns()
    id2column_map = cooc_space.get_id2column_map()
    
    print "The vocabulary has size: " + str(vocab_size)
    
    return cooc_space, mi_space, vocab_map, vocab_size, column_map, id2column_map
    
    
def get_all_most_assoc_cols(mi_space, targets, vocab_map, N):
    """
    Gets columns with highest values for each target row.
    :param mi_space: a semantic space
    :param targets: set of target strings
    :param vocab_map: dictionary that maps row strings to integer ids
    :param N: int, number of columns to extract
    :return most_associated_cols_dict: dictionary, mapping target strings to
                                        column ids
    :return union_m_a_c: set, union of column ids with highest values
    """
    
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
    """
    Assigns the path of the context entropy file.
    :param matrice_name: string, name of matrice file
    :param is_pmi: boolean, whether matrice is weighted with PPMI values
    :param is_lmi: boolean, whether matrice is weighted with PLMI values
    :param is_weighted: boolean, whether matrice is weighted in the first place
    :return c_entrop_file: string, path of context_entropy file
    """
    
    if is_weighted:
        if is_pmi:
            c_entrop_file = matrice_name + "_ppmi" + "_context_entropies" + ".txt" 
        if is_lmi:
            c_entrop_file = matrice_name + "_plmi" + "_context_entropies" + ".txt" 
    if not is_weighted:
        c_entrop_file = matrice_name  + "_freq" + "_context_entropies" + ".txt"

    return c_entrop_file
    

def save_entropies(entr_ranked, entropies_dict, entrop_file):
    """
    Saves entropy rank to disk.
    :param entr_ranked: list of floats, ranked values
    :param entropies_dict: dictionary, mapping strings to values
    :param entrop_file: string, path of output file
    """
    
    print "Writing raw entropies to %s..." % entrop_file                
    with open(entrop_file, 'w') as f_out:
        for context in entr_ranked:
            H = entropies_dict[context]
            print >> f_out, "\t".join((context, str(H)))
                
                
def get_r_entropies(targets, cooc_space, mi_space, is_freq):
    """
    Get row entropies from unweighted or weighted space for a set of targets.
    :param targets: set of target strings
    :return cooc_space: unweighted semantic space
    :param mi_space: weighted semantic space
    :param is_freq: boolean, whether to compute row entropies from unweighted 
                        or weighted space    
    :return r_entropies_dict: dictionary, mapping targets to entropy values
    :return r_entr_ranked: list of ranked entropy values
    """
    
    if is_freq:
        # Compute row entropies from unweighted matrice
        print "Computing row entropies from co-occurence matrice..."
        r_entropies_dict = cooc_space.compute_row_entropies(targets)
        print "Calculated entropies for %d rows." % len(r_entropies_dict)
    else:
        # Compute row entropies from weighted matrice
        print "Computing row entropies from weighted matrice..."
        r_entropies_dict = mi_space.compute_row_entropies(targets)
        print "Calculated entropies for %d rows." % len(r_entropies_dict)
            
    # Rank the row entropies
    r_entr_ranked = sorted(r_entropies_dict, key=lambda x: -(float(r_entropies_dict[x])))
            
    return r_entropies_dict, r_entr_ranked
    

def get_c_entropies(targets, cooc_space, mi_space, N, c_entrop_file, vocab_map, id2column_map, most_associated_cols_dict, union_m_a_c, is_freq, is_weighted):
    """
    Get context entropies from unweighted or weighted space for a set of targets.
    :param targets: set of target strings
    :param cooc_space: unweighted semantic space
    :param mi_space: weighted semantic space
    :param N: int, number of columns to extract
    :param c_entrop_file: string, path of context_entropy file
    :param vocab_map: dictionary that maps row strings to integer ids
    :param id2column_map: list of strings, the column elements
    :param most_associated_cols_dict: dictionary, mapping target strings to
                                        column ids
    :param union_m_a_c: set, union of column ids with highest values
    :param is_freq: boolean, whether to compute entropies from unweighted space
    :param is_weighted: boolean, whether to compute entropies from weighted space
    :return c_entropies_dict: dictionary, mapping contexts to entropy values
    :return c_entr_ranked: list of ranked entropy values
    """
    
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
                    new_c_entropies_dict = cooc_space.compute_context_entropies(new_union_m_a_c)
                elif is_weighted:
                    print "Computing new context entropies from weighted matrice..."
                    new_c_entropies_dict = mi_space.compute_context_entropies(new_union_m_a_c)               
                # Add the new context entropies to the old ones
                print "Calculated entropies for %d new contexts." % len(new_c_entropies_dict)
                c_entropies_dict.update(new_c_entropies_dict)  
                
    except IOError:
        print "No context entropy file found."
        # Build context entropy file if non-existent
        if is_freq:
            print "Computing context entropies instead from co-occurence matrice..."
            c_entropies_dict = cooc_space.compute_context_entropies(union_m_a_c)
            print "Calculated entropies for %d contexts." % len(c_entropies_dict)
        elif is_weighted:
            print "Computing context entropies instead from weighted matrice..."
            c_entropies_dict = mi_space.compute_context_entropies(union_m_a_c)
            print "Calculated entropies for %d contexts." % len(c_entropies_dict)
            
    # Rank the context entropies
    c_entr_ranked = sorted(c_entropies_dict, key=lambda x: -(float(c_entropies_dict[x])))
    
    # Save the (updated) context entropies      
    save_entropies(c_entr_ranked, c_entropies_dict, c_entrop_file)
            
    return c_entropies_dict, c_entr_ranked
        

def make_relative_target_entropies(output_file, vocab_map, id2column_map, test_set, most_associated_cols_dict, c_entropies, N, is_average, is_median):
    """
    Get relative entropy values for x and y for each test pair (x,y) of each test item.
    :param output_file: string, path of output file
    :param vocab_map: dictionary that maps row strings to integer ids
    :param id2column_map: list of strings, the column elements
    :param test_set: list of tuples, each tuple a test item
    :param most_associated_cols_dict: dictionary, mapping target strings to
                                        column ids
    :param c_entropies: dictionary, mapping contexts to entropy values
    :param N: int, number of columns to extract

    :param is_average: boolean, whether to calculate average of context entropies
    :param is_median: boolean, whether to calculate median of context entropies
    :return unscored_output: list of tuples of test items plus their entropy values
                                for x and y for each test pair (x,y).
    """
    
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
    """
    Append test items and their entropy values.
    :param entropies: dictionary, mapping targets to entropy values
    :param test_set: list of tuples, each tuple a test item
    :param vocab_map: dictionary that maps row strings to integer ids
    :return unscored_output: list of tuples of test items plus their entropy values
                                for x and y for each test pair (x,y).
    """
    
    unscored_output = []    
    
    for (x, y, label, relation) in test_set:
        if x not in vocab_map or y not in vocab_map:
            # Assign a special score to out-of-vocab pairs
            unscored_output.append((x, y, label, relation, -999.0, -999.0))
            continue
        
        print (x, y, label, relation, entropies[x], entropies[y])
    
        unscored_output.append((x, y, label, relation, entropies[x], entropies[y]))
    
    return unscored_output


def score_slqs(unscored_output):
    """
    Make scored SLQS output from individual values for test pairs.
    :param unscored_output: list of tuples of test items plus their entropy values
                                for x and y for each test pair (x,y).    
    :return scored_output: list of tuples of test items plus their SLQS score
                                for x and y for each test pair (x,y).    
    """

    scored_output = []

    print "Computing target SLQS for test tuples..."
    
    for (x, y, label, relation, xentr, yentr) in unscored_output:
        
        if xentr == -999.0 or yentr == -999.0:
            scored_output.append((x, y, label, relation, -999.0))
            continue
        # Compute SLQS for y being the hypernym of x
        score = slqs(xentr, yentr)            
    
        scored_output.append((x, y, label, relation, score))
        
    return scored_output
    

def score_slqs_sub(unscored_output):
    """
    Make scored SLQS Sub output from individual values for test pairs.
    :param unscored_output: list of tuples of test items plus their entropy values
                                for x and y for each test pair (x,y).    
    :return scored_output: list of tuples of test items plus their SLQS Sub score
                                for x and y for each test pair (x,y). 
    """

    scored_output = []

    print "Computing target SLQS Sub for test tuples..."
    
    for (x, y, label, relation, xentr, yentr) in unscored_output:
        
        if xentr == -999.0 or yentr == -999.0:
            scored_output.append((x, y, label, relation, -999.0))
            continue
        
        # Compute SLQS Sub for y being the hypernym of x
        score = slqs_sub(xentr, yentr)            
    
        scored_output.append((x, y, label, relation, score))
        
    return scored_output
    

def save_results(scored_output, output_file):
    """
    Saves scored output to disk.
    :param scored_output: list of tuples of test items plus their score
                                for x and y for each test pair (x,y). 
    :param output_file: string, path of output file
    """
    
    with codecs.open(output_file, 'w') as f_out:
            
        for (x, y, label, relation, score) in scored_output:
    
            print >> f_out, '\t'.join((x, y, label, relation, '%.5f' % score))
    
    print "Saved the results to " + output_file