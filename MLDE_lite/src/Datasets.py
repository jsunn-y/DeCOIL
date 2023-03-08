from .encoding_utils import *

encoding_dict = {
    'one-hot' : generate_onehot,
    'georgiev' : generate_georgiev
}

class Dataset():
    """Base class for labeled datasets."""

    def __init__(self, dataframe):

        self.data = dataframe
        self.N = len(self.data)
        self.sorted_data = self.data.sort_values(by='zs score', ascending=False)
        self.all_combos = self.data['Combo'].values
        #normalize fitness
        self.data['fit'] = self.data['fit']/np.max(self.data['fit'].values)
        self.y = self.data['fit']

    def sample_top(self, cutoff, n_samples, seed):
        '''
        Samples n_samples from the top triad scores based on a given cutoff in the ranking and a seed.
        '''
        if self.N <= cutoff:
            sorted = self.data
        else:
            sorted = self.sorted_data[:cutoff]
        
        options = sorted.Combo.values
        np.random.seed(seed)
        return np.random.choice(options, n_samples, replace=False)

    def encode_X(self, encoding):
        if encoding == 'one-hot':
            self.X = np.array(encoding_dict[encoding](self.all_combos)) 
            self.X = self.X.reshape(self.X.shape[0],-1) 

        self.input_dim = self.X.shape[1]
        self.n_residues = self.input_dim/len(ALL_AAS)
    
    def get_mask(self, seqs):
        return list(self.data[self.data['Combo'].isin(seqs)].index)