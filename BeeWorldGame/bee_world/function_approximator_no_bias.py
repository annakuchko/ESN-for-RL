import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from typing import Tuple
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

def rec_plot(s):
    steps=1
    eps = 0.25*(s.max()-s.min())
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z


def relu(x):
    x[x < 0] = 0
    return x

def tanh(x):
    x = np.tanh(x.astype(float))
    return x

class ESN():
    def __init__(self, n_inputs:int, n_outputs:int or None, n_reservoir:int or None=300, 
                 activation:str='ReLU', two_norm:float=1, sparsity:float=0.0, 
                 random_state:int=2022, L2:float=10**-9, 
                 gamma:float=0.5, sampling_bounds:float=1, rho=0.05, xi=0.05,
                 estimator='ridge'):
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
        elif activation=='ReLU':
            self.activation = relu
        elif activation=='Tanh':
            self.activation=tanh
        else:
            raise(NotImplementedError)
        self.L2 = L2
        self.gamma = gamma
        self.sampling_bounds = sampling_bounds
        self.random_state = random_state
        
        self.rho = rho
        self.xi = xi
        self.estimator = estimator
        self.pred_states_memory = np.zeros((2000,300))
        self.k = 0

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

    def _sample_uniform(self, size, sb=None):
        # sample random matrices of
        if sb is None:
            sampling_bounds = self.sampling_bounds
        else:
            sampling_bounds = sb
        return self.random_state_.uniform(low=-sampling_bounds, 
                                          high=sampling_bounds, 
                                          size=size)
        # return self.random_state_.uniform(low=0, 
        #                                       high=1, 
        #                                       size=size)
        
    def initweights(self):
        # initialize recurrent weights
        # begin with a reservoir matrix and
        # delete the fraction of connections given by (self.sparsity):
        A = self._sample_uniform(size=(self.n_reservoir, self.n_reservoir))
        A[self.random_state_.rand(*A.shape) < self.sparsity] = 0
        
        # compute the spectral radius of these weights and
        # rescale them to reach the requested spectral radius:
        # print(f'Scaling of A: {(self.two_norm/np.linalg.norm(A))}')
        # self.A = A * (self.two_norm/np.linalg.norm(A))
        self.A = A  / max(abs(np.linalg.eigvals(A)))

        # random input weights:
        # self.C = self._sample_uniform(size=(self.n_reservoir, self.n_inputs+1))
        self.C = self._sample_uniform(size=(self.n_reservoir, self.n_inputs))
        
        # random bias weights
        # self.Z = self._sample_uniform(size=self.n_reservoir, sb=0.1)
        # self.Z = None
        self.collect_states = []
    def _get_new_state(self, state, input_pattern):
        """performs one update step.
        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
       
        preactivation = (self.rho * self.A @ state.copy())\
                        + (self.xi * self.C @ input_pattern.copy())
                        # + self.Z
        self.collect_states.append(preactivation)
        return self.activation(preactivation)
        # return preactivation
        
    
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
        
        self.max = inputs.max(0)
        self.min = inputs.min(0)
        
        sb = self.sampling_bounds
        
        self.scale = (self.max - self.min)/(sb-(-sb))
        
        inputs = (inputs - self.min)*(sb-(-sb))/(self.max - self.min)+(-sb)
        outputs = (outputs - self.min[0])*(sb-(-sb))/(self.max[0] - self.min[0])+(-sb)
        
        # inputs = np.hstack((np.ones(inputs.shape[0]).reshape(-1,1), inputs))
        
        states = np.zeros((inputs.shape[0],self.n_reservoir))

        for n in range(1, inputs.shape[0]):
            states[n, :] = self._get_new_state(states[n-1,:], inputs[n, :])
        
        collected_states = np.array(self.collect_states)
        
        # sns.heatmap(collected_states.T)
        # plt.title('Preactivation states heatmap')
        # plt.show()
        
        # plt.plot(collected_states[:,0])
        # plt.title('Preactivation dynamics')
        # plt.show()
        
        # plt.plot(states[:,0])
        # plt.title('States dynamics')
        # plt.show()
        
        
        # sns.heatmap(states.T)
        # plt.title('States heatmap')
        # plt.show()
        
        # plt.imshow(rec_plot(states[-1,:]), cmap='binary', origin='lower')
        # plt.title('Reccurence plot of states')
        # plt.show()
        
        new_states=np.zeros_like(states)
        for n in range(states.shape[0]-1):
            new_states[n,:] = states[n,:] - self.gamma*states[n+1,:]

        # Solve for W:
        Y = outputs.copy()
        
        X_old = np.hstack((np.ones(states.shape[0]).reshape(-1,1), states.copy()))
        X = np.hstack((np.ones(new_states.shape[0]).reshape(-1,1), new_states.copy()))
        
        if self.L2!=0:
            xtransx = X.T @ X
            identity = np.identity(X.shape[1])
            identity[0] = 0
            l2_identity = self.L2 * identity
            if self.estimator=='liu':
                self.W = np.linalg.inv(X.T@X + identity) @ (X.T@X + l2_identity) @ np.linalg.inv(X.T@X) @ X.T@Y
            elif self.estimator=='ridge':
                self.W = np.linalg.inv(X.T@X + l2_identity) @ X.T@Y
     
                
            
        else:
            self.W = np.linalg.pinv(X[:, :].copy()) @ outputs[:, :].copy()
        
        # plt.scatter(np.arange(self.W.shape[0]), self.W)
        # plt.title('W weights')
        # plt.show()
        # remember the last state for later:
        self.laststate = states[-1, :].copy()
        self.lastinput = inputs[-1, :].copy()
        self.lastoutput = outputs[-1, :].copy()
        self.pred_states_memory = states.copy()
        
            
      
        pred_train=np.zeros_like(outputs)
        # out = np.hstack((np.ones(states.shape[0]).reshape(-1,1), states))
        for n in range(X_old.shape[0]):
            pred_train[n, :] = self.W.T @ X_old[n, :]

        # (outputs - self.min[0])*(sb-(-sb))/(self.max[0] - self.min[0])+(-sb)
        
        # from sklearn.metrics import r2_score
        ident = self.L2*np.identity(X_old.shape[1])
        VIF = np.linalg.inv(X_old.T@X_old + ident)@(X.T@X)@np.linalg.inv(X.T@X+ident)
        self.VIF = np.diagonal(VIF).mean()
        # print(f'Mean VIF: {self.VIF}')
        # print(f'Conditional number: {np.linalg.cond(X)}')
        
        # print(f'train min: {self.min}')
        # print(f'train max: {self.max}')
        
        pred_train = pred_train-(-sb)/(sb-(-sb))*(self.max[0] - self.min[0]) + self.min[0]

        
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
        sb = self.sampling_bounds
        
        inputs = (inputs - self.min)*(sb-(-sb))/(self.max - self.min)+(-sb)
        
        n_samples = inputs.shape[0]
        states = np.zeros((n_samples, self.n_reservoir))
        outputs = np.zeros((n_samples, self.n_outputs))
        
        for n in range(n_samples):
            states[n, :] = self._get_new_state(self.laststate, inputs[n, :])
        
        new_states=np.zeros_like(states)
        for n in range(states.shape[0]-1):
            new_states[n,:] = states[n,:] - self.gamma*states[n+1,:]

        out = np.hstack((np.ones(states.shape[0]).reshape(-1,1), states))
        # out = np.hstack((np.ones(states.shape[0]).reshape(-1,1), states))
        
        for n in range(n_samples):
            outputs[n, :] = self.W.T @ out[n, :]
        if upd:
            self.laststate = states.reshape(-1).copy()
            self.pred_states_memory[self.k,:] = states.reshape(-1).copy()
            self.k+=1
            
        res = outputs.copy()-(-sb)/(sb-(-sb))*(self.max[0] - self.min[0]) + self.min[0]
        
        return res

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
    