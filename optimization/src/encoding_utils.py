import numpy as np
from optimization.src.seqtools import *

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