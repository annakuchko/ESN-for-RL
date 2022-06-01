import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from typing import Tuple

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


        # states[0, :] = self.laststate
        for n in range(n_samples):
            states[n, :] = self._get_new_state(self.laststate, inputs[n, :])
            outputs[n, :] = np.dot(self.W.T, states[n, :]) 

        if upd:
            # print('upd states: ', states.shape)
            self.laststate = states.reshape(-1)
        return outputs[:]


    
if __name__=='__main__':
    print('Testing ESN')
    class LorenzAttractor:
        """
        Parameters
        ----------
        num_points: int
            Number of points for X, Y, Z coordinates.
        init_point: tuple
            Initial point [x0, y0, z0] for chaotic attractor system.
            Do not use zeros or very big values because of unstable behavior of dynamic system.
            Default: x, y, z == 1e-4.
        step: float / int
            Step for the next coordinate of dynamic system. Default: 1.0.
       """
    
        def __init__(
            self,
            num_points: int,
            init_point: Tuple[float, float, float] = (1e-4, 1e-4, 1e-4),
            step: float = 1.0,
            show_log: bool = False,
            **kwargs: dict,
        ):
            if show_log:
                print(f"[INFO]: Initialize chaotic system: {self.__class__.__name__}\n")
            self.num_points = num_points
            self.init_point = init_point
            self.step = step
            self.kwargs = kwargs
    
        def get_coordinates(self):
            return np.array(list(next(self)))
    
        def __len__(self):
            return self.num_points
    
        def __iter__(self):
            return self
    
        def __next__(self):
            points = self.init_point
            for i in range(self.num_points):
                try:
                    yield points
                    next_points = self.attractor(*points, **self.kwargs)
                    points = tuple(prev + curr / self.step for prev, curr in zip(points, next_points))
                except OverflowError:
                    print(f"[FAIL]: Cannot do the next step because of floating point overflow. Step: {i}")
                    break
    
        @abstractmethod
        def attractor(
            self, x: float, y: float, z: float, sigma: float = 10, beta: float = 8 / 3, rho: float = 28,
        ) -> Tuple[float, float, float]:
            r"""Calculate the next coordinate X, Y, Z for 3rd-order Lorenz system
            Parameters
            ----------
            x, y, z : float
                Input coordinates X, Y, Z respectively.
            sigma, beta, rho : float
                Lorenz system parameters. Default:
                    - sigma = 10,
                    - beta = 8/3,
                    - rho = 28
            """
            x_out = sigma * (y - x)
            y_out = rho * x - y - x * z
            z_out = x * y - beta * z
            return x_out, y_out, z_out
    
        def update_attributes(self, **kwargs):
            """Update chaotic system parameters."""
            for key in kwargs:
                if key in self.__dict__ and not key.startswith("_"):
                    self.__dict__[key] = kwargs.get(key)
       
        
    def plot_system(trainlen, future, data, prediction, diff=True):
        part = ['X', 'Y', 'Z']
        for i in range(len(part)):
            plt.figure(figsize=(20,5))
            plt.title(f'Lorenz attractor, {part[i]}')
            plt.plot(range(trainlen,trainlen+future),data[trainlen:trainlen+future][:,i],'g',label="target system")
            plt.plot(range(trainlen,trainlen+future),prediction[:,i],'r', label="ESN prediction")
            if diff:
                plt.plot(
                    range(trainlen,trainlen+future),
                    data[trainlen:trainlen+future][:,i]-prediction[:,i],
                    'grey',label="difference"
                )
            lo,hi = plt.ylim()
            plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
            plt.legend(loc=(0.61,1.1))
            plt.show()
            
            
    def plot_attractor(data, prediction, trainlen, future, rho):
        ax = plt.figure(figsize=(15,15)).add_subplot(projection='3d')
        ax.plot(data[trainlen:,0], data[trainlen:,1], data[trainlen:,2], lw=0.5, color = 'g',label='target system')
        ax.plot(prediction[:,0], prediction[:,1], prediction[:,2], lw=0.5, color='r', label='ESN prediction')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(f"Lorenz Attractor, rho={rho}")
        ax.legend()
    
        plt.show()
           
    trainlen = 2000
    future = 2000
    # rho=28 (chaos)
    chaos = LorenzAttractor(num_points=trainlen+future, 
                                    init_point=(0.1, 0, -0.1), 
                                    step=100, 
                                    rho=28).get_coordinates()
    dim = 3
    esn = ESN(n_inputs = dim,
                  n_outputs = dim,
                  n_reservoir = 300,
                  two_norm = 1,
                  random_state=42,
                   sparsity=0.,
                    L2=0.5,
                    gamma=0
                    
                    
          
                 )
    pred_training = esn.fit((chaos[:trainlen-1000]),chaos[trainlen-1000:trainlen])
        
    prediction = esn.predict((chaos[trainlen:trainlen+future]), upd=True)
    print("test error: \n"+str(np.sqrt(np.mean((prediction - chaos[trainlen:trainlen+future])**2))))
    plt.plot(pred_training[1,:], color='r')
    plt.plot(chaos[trainlen-1000:trainlen,1], color='g')
    plt.show()
    
    plt.plot(prediction[:,1], color='r')
    plt.plot(chaos[trainlen:,1], color='g')
    plt.show()
    