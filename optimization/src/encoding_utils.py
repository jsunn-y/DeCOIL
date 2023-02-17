import numpy as np
from dclo.src.seqtools import *
from scipy.spatial import distance

index2base_dict = {
    0: "A",
    1: "C",
    2: "G",
    3: "T"
}

seq2encoding_dict = {
    "A": (1, 0, 0, 0),
    "C": (0, 1, 0, 0),
    "G": (0, 0, 1, 0),
    "T": (0, 0, 0, 1),
    "R": (1, 0, 1, 0),
    "Y": (0, 1, 0, 1),
    "M": (1, 1, 0, 0),
    "K": (0, 0, 1, 1),
    "S": (0, 1, 1, 0),
    "W": (1, 0, 0, 1),
    "H": (1, 1, 0, 1),
    "B": (0, 1, 1, 1),
    "V": (1, 1, 1, 0),
    "D": (1, 0, 1, 1),
    "N": (1, 1, 1, 1)
    }

encoding2choices_dict = {
    (1, 0, 0, 0): ('A',),
    (0, 1, 0, 0): ('C',),
    (0, 0, 1, 0): ('G',),
    (0, 0, 0, 1): ('T',),
    (1, 0, 1, 0): ('A', 'G'),
    (0, 1, 0, 1): ('C', 'T'),
    (1, 1, 0, 0): ('A', 'C'),
    (0, 0, 1, 1): ('G', 'T'),
    (0, 1, 1, 0): ('C', 'G'),
    (1, 0, 0, 1): ('A', 'T'),
    (1, 1, 0, 1): ('A', 'C', 'T'),
    (0, 1, 1, 1): ('C', 'G', 'T'),
    (1, 1, 1, 0): ('A', 'C', 'G'),
    (1, 0, 1, 1): ('A', 'G', 'T'),
    (1, 1, 1, 1): ('A', 'C', 'G', 'T')
    }

encoding2seq_dict = {v: k for k, v in seq2encoding_dict.items()}

def encoding2seq(encoding):
    encoding = encoding.reshape((-1, 4))
    seq = ""
    for row in encoding:
        seq += encoding2seq_dict[tuple(row)]
    return seq

def generate_mixedcodon2aaprobs_dict():
    full_dict = {}
    encoding_choices = seq2encoding_dict.values()
    for choice1 in encoding_choices:
        for choice2 in encoding_choices:
            for choice3 in encoding_choices:
                encoding = choice1 + choice2 + choice3
                aaprobs_dict = mixedcodon2aaprobs(np.array(encoding))
                full_dict[encoding] = aaprobs_dict
    return full_dict

def mixedcodon2aaprobs(encoding):
    '''
    Input: 12 bit encoding, corresponding to a single codon
    Output: dictionary from AAs to their corresponding probabilities
    '''
    encoding = encoding.reshape((-1, 4))

    for i, row in enumerate(encoding):
        encoding[i, :] = row * 12 / sum(row) #normalized to probabilities (but scaled by 12 to remain as integer)
    encoding = encoding.astype(int)

    #print(encoding)
    aa_probs = {el:0 for el in ALL_AAS}

    for i, base1 in enumerate(['A', 'C', 'G', 'T']):
         for j, base2 in enumerate(['A', 'C', 'G', 'T']):
             for k, base3 in enumerate(['A', 'C', 'G', 'T']):
                codon = base1 + base2 + base3
                aa = SequenceTools.codon2protein_[codon]
                prob = encoding[0,i]*encoding[1,j]*encoding[2,k]
                aa_probs[aa] += prob 

    return aa_probs

def seq2encoding(seq):
    encoding = np.zeros((1, 4*len(seq)))
    for i, let in enumerate(seq):
        encoding[0, 4*i:4*(i+1)] = seq2encoding_dict[let]
    return encoding

def hammingdistance(seq1, seq2):
    return sum(s1 != s2 for s1, s2 in zip(seq1, seq2))

def manhattandistance(a, b):
    return np.sum(np.absolute(a-b))

def euclideandistance(a, b):
    return distance.euclidean(a, b)

def allowed(encoding):
    encoding2 = np.copy(encoding)
    encoding2 = encoding2.reshape((-1, 4))
    for i, row in enumerate(encoding2):
        if sum(row) == 1:
            encoding2[i, np.where(row == 1)] = 2

    encoding2 = encoding2.flatten()
    return np.where(encoding2 != 2)[0]

def get_library_size(encoding):
    encoding = encoding.reshape(12,4)
    sums = np.sum(encoding, axis = 1)
    return np.product(sums)

def get_AA_encodings(all_encodings):
    all_encodings2 = np.copy(all_encodings)
    AA_encodings = np.zeros((all_encodings.shape[0], 21*int(all_encodings.shape[1]/12), all_encodings.shape[2]))
    #AA library sizes
    library_sizes = []

    for i, encodings in enumerate(all_encodings):
        #options = np.zeros((all_encodings.shape[2], 4))

        for k, encoding in enumerate(encodings.T):
            
            encoding = encoding.reshape(4, -1)

            for j, row in enumerate(encoding):
                AA_encoding = np.array(list(mixedcodon2aaprobs(row).values()))
                AA_encodings[i, 21*j : 21*(j+1), k] = AA_encoding
        
        
        AA_encodings_sum = np.sum(AA_encodings[i], axis = 1)

        AA_encodings_sum = AA_encodings_sum.reshape(4, 21)
        options = np.count_nonzero(AA_encodings_sum, axis = 1)
        
        #options = np.max(options, axis=1)
        library_sizes.append(np.product(options))

    unique_AA_encodings, indices = np.unique(AA_encodings, axis = 0, return_index=True)

    unique_encodings = all_encodings2[indices]
    print(unique_encodings.shape)
    return library_sizes, indices, AA_encodings, unique_encodings, unique_AA_encodings
    
ALL_AAS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "*")
num_tokens = len(ALL_AAS)

# Copied from ProFET (Ofer & Linial, DOI: 10.1093/bioinformatics/btv345)
# Original comment by the ProFET authors: 'Acquired from georgiev's paper of
# AAscales using helper script "GetTextData.py". + RegEx cleaning DOI: 10.1089/cmb.2008.0173'
# gg_1 = {'Q': -2.54, 'L': 2.72, 'T': -0.65, 'C': 2.66, 'I': 3.1, 'G': 0.15, 'V': 2.64, 'K': -3.89, 'M': 1.89, 'F': 3.12, 'N': -2.02, 'R': -2.8, 'H': -0.39, 'E': -3.08, 'W': 1.89, 'A': 0.57, 'D': -2.46, 'Y': 0.79, 'S': -1.1, 'P': -0.58}
# gg_2 = {'Q': 1.82, 'L': 1.88, 'T': -1.6, 'C': -1.52, 'I': 0.37, 'G': -3.49, 'V': 0.03, 'K': 1.47, 'M': 3.88, 'F': 0.68, 'N': -1.92, 'R': 0.31, 'H': 1, 'E': 3.45, 'W': -0.09, 'A': 3.37, 'D': -0.66, 'Y': -2.62, 'S': -2.05, 'P': -4.33}
# gg_3 = {'Q': -0.82, 'L': 1.92, 'T': -1.39, 'C': -3.29, 'I': 0.26, 'G': -2.97, 'V': -0.67, 'K': 1.95, 'M': -1.57, 'F': 2.4, 'N': 0.04, 'R': 2.84, 'H': -0.63, 'E': 0.05, 'W': 4.21, 'A': -3.66, 'D': -0.57, 'Y': 4.11, 'S': -2.19, 'P': -0.02}
# gg_4 = {'Q': -1.85, 'L': 5.33, 'T': 0.63, 'C': -3.77, 'I': 1.04, 'G': 2.06, 'V': 2.34, 'K': 1.17, 'M': -3.58, 'F': -0.35, 'N': -0.65, 'R': 0.25, 'H': -3.49, 'E': 0.62, 'W': -2.77, 'A': 2.34, 'D': 0.14, 'Y': -0.63, 'S': 1.36, 'P': -0.21}
# gg_5 = {'Q': 0.09, 'L': 0.08, 'T': 1.35, 'C': 2.96, 'I': -0.05, 'G': 0.7, 'V': 0.64, 'K': 0.53, 'M': -2.55, 'F': -0.88, 'N': 1.61, 'R': 0.2, 'H': 0.05, 'E': -0.49, 'W': 0.72, 'A': -1.07, 'D': 0.75, 'Y': 1.89, 'S': 1.78, 'P': -8.31}
# gg_6 = {'Q': 0.6, 'L': 0.09, 'T': -2.45, 'C': -2.23, 'I': -1.18, 'G': 7.47, 'V': -2.01, 'K': 0.1, 'M': 2.07, 'F': 1.62, 'N': 2.08, 'R': -0.37, 'H': 0.41, 'E': 0, 'W': 0.86, 'A': -0.4, 'D': 0.24, 'Y': -0.53, 'S': -3.36, 'P': -1.82}
# gg_7 = {'Q': 0.25, 'L': 0.27, 'T': -0.65, 'C': 0.44, 'I': -0.21, 'G': 0.41, 'V': -0.33, 'K': 4.01, 'M': 0.84, 'F': -0.15, 'N': 0.4, 'R': 3.81, 'H': 1.61, 'E': -5.66, 'W': -1.07, 'A': 1.23, 'D': -5.15, 'Y': -1.3, 'S': 1.39, 'P': -0.12}
# gg_8 = {'Q': 2.11, 'L': -4.06, 'T': 3.43, 'C': -3.49, 'I': 3.45, 'G': 1.62, 'V': 3.93, 'K': -0.01, 'M': 1.85, 'F': -0.41, 'N': -2.47, 'R': 0.98, 'H': -0.6, 'E': -0.11, 'W': -1.66, 'A': -2.32, 'D': -1.17, 'Y': 1.31, 'S': -1.21, 'P': -1.18}
# gg_9 = {'Q': -1.92, 'L': 0.43, 'T': 0.34, 'C': 2.22, 'I': 0.86, 'G': -0.47, 'V': -0.21, 'K': -0.26, 'M': -2.05, 'F': 4.2, 'N': -0.07, 'R': 2.43, 'H': 3.55, 'E': 1.49, 'W': -5.87, 'A': -2.01, 'D': 0.73, 'Y': -0.56, 'S': -2.83, 'P': 0}
# gg_10 = {'Q': -1.67, 'L': -1.2, 'T': 0.24, 'C': -3.78, 'I': 1.98, 'G': -2.9, 'V': 1.27, 'K': -1.66, 'M': 0.78, 'F': 0.73, 'N': 7.02, 'R': -0.99, 'H': 1.52, 'E': -2.26, 'W': -0.66, 'A': 1.31, 'D': 1.5, 'Y': -0.95, 'S': 0.39, 'P': -0.66}
# gg_11 = {'Q': 0.7, 'L': 0.67, 'T': -0.53, 'C': 1.98, 'I': 0.89, 'G': -0.98, 'V': 0.43, 'K': 5.86, 'M': 1.53, 'F': -0.56, 'N': 1.32, 'R': -4.9, 'H': -2.28, 'E': -1.62, 'W': -2.49, 'A': -1.14, 'D': 1.51, 'Y': 1.91, 'S': -2.92, 'P': 0.64}
# gg_12 = {'Q': -0.27, 'L': -0.29, 'T': 1.91, 'C': -0.43, 'I': -1.67, 'G': -0.62, 'V': -1.71, 'K': -0.06, 'M': 2.44, 'F': 3.54, 'N': -2.44, 'R': 2.09, 'H': -3.12, 'E': -3.97, 'W': -0.3, 'A': 0.19, 'D': 5.61, 'Y': -1.26, 'S': 1.27, 'P': -0.92}
# gg_13 = {'Q': -0.99, 'L': -2.47, 'T': 2.66, 'C': -1.03, 'I': -1.02, 'G': -0.11, 'V': -2.93, 'K': 1.38, 'M': -0.26, 'F': 5.25, 'N': 0.37, 'R': -3.08, 'H': -1.45, 'E': 2.3, 'W': -0.5, 'A': 1.66, 'D': -3.85, 'Y': 1.57, 'S': 2.86, 'P': -0.37}
# gg_14 = {'Q': -1.56, 'L': -4.79, 'T': -3.07, 'C': 0.93, 'I': -1.21, 'G': 0.15, 'V': 4.22, 'K': 1.78, 'M': -3.09, 'F': 1.73, 'N': -0.89, 'R': 0.82, 'H': -0.77, 'E': -0.06, 'W': 1.64, 'A': 4.39, 'D': 1.28, 'Y': 0.2, 'S': -1.88, 'P': 0.17}
# gg_15 = {'Q': 6.22, 'L': 0.8, 'T': 0.2, 'C': 1.43, 'I': -1.78, 'G': -0.53, 'V': 1.06, 'K': -2.71, 'M': -1.39, 'F': 2.14, 'N': 3.13, 'R': 1.32, 'H': -4.18, 'E': -0.35, 'W': -0.72, 'A': 0.18, 'D': -1.98, 'Y': -0.76, 'S': -2.42, 'P': 0.36}
# gg_16 = {'Q': -0.18, 'L': -1.43, 'T': -2.2, 'C': 1.45, 'I': 5.71, 'G': 0.35, 'V': -1.31, 'K': 1.62, 'M': -1.02, 'F': 1.1, 'N': 0.79, 'R': 0.69, 'H': -2.91, 'E': 1.51, 'W': 1.75, 'A': -2.6, 'D': 0.05, 'Y': -5.19, 'S': 1.75, 'P': 0.08}
# gg_17 = {'Q': 2.72, 'L': 0.63, 'T': 3.73, 'C': -1.15, 'I': 1.54, 'G': 0.3, 'V': -1.97, 'K': 0.96, 'M': -4.32, 'F': 0.68, 'N': -1.54, 'R': -2.62, 'H': 3.37, 'E': -2.29, 'W': 2.73, 'A': 1.49, 'D': 0.9, 'Y': -2.56, 'S': -2.77, 'P': 0.16}
# gg_18 = {'Q': 4.35, 'L': -0.24, 'T': -5.46, 'C': -1.64, 'I': 2.11, 'G': 0.32, 'V': -1.21, 'K': -1.09, 'M': -1.34, 'F': 1.46, 'N': -1.71, 'R': -1.49, 'H': 1.87, 'E': -1.47, 'W': -2.2, 'A': 0.46, 'D': 1.38, 'Y': 2.87, 'S': 3.36, 'P': -0.34}
# gg_19 = {'Q': 0.92, 'L': 1.01, 'T': -0.73, 'C': -1.05, 'I': -4.18, 'G': 0.05, 'V': 4.77, 'K': 1.36, 'M': 0.09, 'F': 2.33, 'N': -0.25, 'R': -2.57, 'H': 2.17, 'E': 0.15, 'W': 0.9, 'A': -4.22, 'D': -0.03, 'Y': -3.43, 'S': 2.67, 'P': 0.04}

# # Package all georgiev parameters
# georgiev_parameters = [gg_1, gg_2, gg_3, gg_4, gg_5, gg_6, gg_7, gg_8, gg_9,
#                        gg_10, gg_11, gg_12, gg_13, gg_14, gg_15, gg_16, gg_17,
#                        gg_18, gg_19]


# def get_georgiev_params_for_aa(aa):
#     return [gg[aa] for gg in georgiev_parameters]

# def get_georgiev_params_for_seq(s):
#     return np.concatenate([get_georgiev_params_for_aa(aa) for aa in s])

# def seqs_to_georgiev(seqs):
#     return np.stack([get_georgiev_params_for_seq(s) for s in seqs])

# def generate_onehot(seqs):
#     """
#     Builds a onehot encoding for a given combinatorial space.
#     """
#     # Make a dictionary that links amino acid to index
#     one_hot_dict = {aa: i for i, aa in enumerate(ALL_AAS)}

#     # Build an array of zeros
#     onehot_array = np.zeros([len(seqs), len(seqs[0]), len(ALL_AAS)])
    
#     # Loop over all combos. This should all be vectorized at some point.
#     for i, combo in enumerate(seqs):
        
#         # Loop over the combo and add ones as appropriate
#         for j, character in enumerate(combo):
            
#             # Add a 1 to the appropriate position
#             onehot_ind = one_hot_dict[character]
#             onehot_array[i, j, onehot_ind] = 1
            
#     # Return the array
#     X = onehot_array
#     return X

# def generate_georgiev(seqs):
#     """
#     Builds a georgiev encoding for a given combinatorial space.
#     """
#     X = seqs_to_georgiev(seqs)
#     return X

#print(mixedcodon2aaprobs(np.tile([1,1,1,0], 3)))