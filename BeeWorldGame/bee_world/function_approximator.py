import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    x[x < 0] = 0
    return x

class ESN():
    def __init__(self, n_inputs:int, n_outputs:int, n_reservoir:int=300, 
                 activation=None, two_norm:float=1, sparsity:float=0.0, 
                 random_state:int=2022, L2:float=10**-9, 
                 gamma:float=0.5, sampling_bounds:float=0.05):
        """
        The class for the ESN model.

        Parameters
        ----------
        n_inputs : int
            The dimensionality of the input.
        n_outputs : int
            The dimensionality of the output.
        n_reservoir : int, optional
            The size of the reservoir. The default is 300.
        activation : TYPE, optional
            The activation function. The default is None.
        two_norm : float, optional
            The 2-norm of the recurrent weight matrix. The default is 1.
        sparsity : float, optional
            The proportion of recurrent weights set to zero. The default is 0.0.
        random_state : int, optional
            A positive integer seed, np.rand.RandomState object,
            or None to use numpy's builting RandomState. The default is 2022.
        L2 : float, optional
            The regularisation parameter of the ridge regression. The default 
            is 10**-9.
        gamma : float, optional
            The discount factor of the value function. The default is 0.5.
        sampling_bounds : float, optional
            The uniform bounds of the interval from which to sample the random 
            matrices. The default is 0.05.

        Raises
        ------
        Exception
            Exception for the invalid random seed.

        Returns
        -------
        None.

        """
        
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.two_norm = two_norm
        self.sparsity = sparsity
        if activation is None:
            self.activation = relu
        else:
            self.activation = activation
        self.L2 = L2
        self.gamma = gamma
        self.sampling_bounds = sampling_bounds
        self.random_state = random_state

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand
        
        self.initweights()

    def _sample_uniform(self, size):
        # sample random matrices of
        sampling_bounds = self.sampling_bounds
        return self.random_state_.uniform(low=-sampling_bounds, 
                                          high=sampling_bounds, 
                                          size=size)
    
    def initweights(self):
        # initialize recurrent weights
        # begin with a reservoir matrix and
        # delete the fraction of connections given by (self.sparsity):
        A = self._sample_uniform(size=(self.n_reservoir, self.n_reservoir))
        A[self.random_state_.rand(*A.shape) < self.sparsity] = 0
        
        # compute the spectral radius of these weights and
        # rescale them to reach the requested spectral radius:
        self.A = A * (self.two_norm/np.linalg.norm(A))

        # random input weights:
        self.C = self._sample_uniform(size=(self.n_reservoir, self.n_inputs))
        
        # random bias weights
        self.Z = self._sample_uniform(size=self.n_reservoir)

    def _get_new_state(self, state, input_pattern):
        """performs one update step.
        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        preactivation = np.dot(self.A, state)\
                        + np.dot(self.C, input_pattern)\
                        + self.Z
        return self.activation(preactivation)
    
    def fit(self, inputs, outputs, inspect=False):
        """
        Collect the network's reaction to training data, train readout weights.
        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states
        Returns:
            the network's output on the training data, using the trained weights
        """
        
        # step the reservoir through the given input,output pairs:
        states = np.zeros((inputs.shape[0],self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._get_new_state(states[n-1,:], inputs[n, :])
        
        new_states=np.zeros_like(states)
        for n in range(states.shape[0]-1):
            new_states[n,:] = states[n,:] - self.gamma*states[n+1,:]

        # Solve for W:
        Y = outputs[:,:]
        X = new_states[:,:]
        
        if self.L2!=0:
            xtransx = np.dot(X.T, X) 
            l2_identity = self.L2 * np.identity(self.n_reservoir)
            inv = np.linalg.inv(xtransx+l2_identity)
            X_inv = np.dot(inv,X.T)
            self.W = np.dot(X_inv,Y)
            
        else:
            self.W = np.dot(np.linalg.pinv(new_states[:, :]),
                                outputs[:, :])

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = outputs[-1, :]

        pred_train=np.zeros_like(outputs)  
        for n in range(inputs.shape[0]):
            pred_train[n, :] = np.dot(self.W.T, states[n, :])
        
        return pred_train.T

    def predict(self, inputs, continuation=False, upd=False):
        """
        Apply the learned weights to the network's reactions to new input.
        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state
        Returns:
            Array of output activations
        """
        n_samples = inputs.shape[0]

        states = np.zeros((n_samples, self.n_reservoir))
        outputs = np.zeros((n_samples, self.n_outputs))

        for n in range(n_samples):
            states[n, :] = self._get_new_state(self.laststate, inputs[n, :])
            outputs[n, :] = np.dot(self.W.T, states[n, :]) 

        if upd:
            self.laststate = states.reshape(-1)
        return outputs[:]
