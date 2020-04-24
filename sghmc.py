import numpy as np
from preprocess import create_batch

def sghmc(gradU, dataset, batch_size, start_states, alpha=None, eta=None, n_epochs=500, n_burn=50, random_seed=42):
    '''
    SGHMC cleaner version, core implementation
    Details about the algorithm and paramters settings are in chapter four of the report.

    author:
    jiajun.song@duke.edu; yipin.song@duke.edu
    
    Parameters
    --------
    gradU : callable
        gradient of the posterior probability with respect to the parameter
        vector, `gradU(params, data, scale)`
    
    dataset: 2-d array, shape = (n_samples, n_features)
        define the dataset for the problem
    
    batch_size: int
        minibatch size, specifies amount of dataset to use at each leap-frog steps
    
    start_states: 1-d array, shape = (n_params, )
        the start state for the parameter vector
    

    Optional Parameters
    --------------------
    alpha: float
        momentum decay, default set to 0.1
    
    eta: float
        learning rate, default set to 2 * 0.01 / dataset.shape[0]
    
    n_epochs: int
        specifies number of epochs to perform, default set to 500
    
    n_burn: int
        specifies number of epochs at the start to remove, default set to 50
    
    random_seed: int
        seed for random number generation, default set to 42


    Return
    -------
    proposal_samples: 2-d array, shape = (n_epochs - n_burn, n_params)
        the parameter vector samples obtained by SGHMC following the target posterior distribution
    '''

    # check some options
    assert batch_size >= 1, 'batch size has to be >= 1'
    assert n_burn >= 0, 'n_burn has to be >= 0'
    assert alpha >= 0, 'momentum decay has to be >= 0'
    assert alpha <= 1, 'momentum decay has to be <= 1'

    ## random state
    RSTATE = np.random.RandomState(int(random_seed))
    
    ## parameter vector dimension and number of samples
    n_samples = dataset.shape[0]
    n_params = start_states.shape[0]
    assert batch_size <= n_samples, 'batch_size cannot be bigger than number of samples'
    
    ## placeholder for proposal_samples
    proposal_samples = np.zeros((n_epochs, n_params))
    proposal_samples[0] = start_states
    
    ## hyperparameter
    ## please refer to report chapter 4 for details about tunning
    alpha = 0.1 if alpha is None else alpha
    eta = 2 * 0.01 / n_samples if eta is None else eta
    bhat = 0
    Sigma = np.linalg.cholesky(2 * (alpha-bhat) * eta * np.eye(n_params))
    
    ## data
    minibatch_data, n_batches = create_batch(dataset, batch_size)
    
    ## main loop
    for i in range(n_epochs-1):
        cur_states = proposal_samples[i]
        
        ## resample momentum at each new epoch
        momentum = np.sqrt(eta) * RSTATE.randn(n_params)
        
        ## sghmc sampler
        for j in range(n_batches):
            cur_states = cur_states + momentum
            gradU_batch = gradU(cur_states, minibatch_data[:,:,j], n_samples/batch_size)
            momentum = (1-alpha) * momentum - eta * gradU_batch.flatten() \
                                            + np.dot(Sigma, RSTATE.randn(n_params))
        
        ## store results
        proposal_samples[i+1] = cur_states
    
    return proposal_samples[n_burn: , :]