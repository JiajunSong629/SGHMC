import numpy as np

def create_batch( dataset, batch_size, doshuffle=True, random_seed=42):
    '''
    Create minibatch samples
    
    Parameters
    --------
    dataset: 2-d array, shape = (n_samples, n_features)
        define the dataset for the problem
    
    batch_size: int
        minibatch size, specifies amount of dataset to use at each leap-frog steps
    

    Optional Parameters:
    -------
    doshuffle: boolean
        if shffule the dataset, default set to True
    
    random_seed: int
        seed for random number generation, default set to 42


    Return
    -------
    dataset: 3-d array, shape = (batch_size, n_params, n_batches)
        dataset separated into batches, ith batch is dataset[:, :, i]
    
    n_batches: int
        the number of batches created within the dataset

    '''

    RSTATE = np.random.RandomState(int(random_seed))
    
    n_samples, n_params = dataset.shape
    if n_samples % batch_size != 0:
        print ('%d data will be dropped during batching' % (dataset.shape[0] % batch_size))
    batch_sample_size = int(dataset.shape[0] / batch_size * batch_size)
    n_batches = int(dataset.shape[0]) // batch_size

    if doshuffle:
        ind = list(range( dataset.shape[0] ))
        RSTATE.shuffle( ind )
        dataset = dataset[ind]
    
    dataset = dataset[:batch_sample_size].reshape(batch_size, n_params, n_batches)
    
    return dataset, n_batches