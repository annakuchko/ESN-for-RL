import numpy as np
import matplotlib.pyplot as plt
from BeeWorldGame.bee_world.abstract_env import AbstractEnv
    
class BeeWorldEnv(AbstractEnv):
    def __init__(self, steps:int=2000, w:float=0.1, c:float=0.01, 
                 circle_length:int=1):
        """
        Environment for the Bee World game 
        (as in https://www.researchgate.net/publication/352738283_Using_Echo_
         State_Networks_to_Approximate_Value_Functions_for_Control) . 
        The standard Bee World is set on the circle of unit circumference, 
        which is represented as an interval with edges defined. At every point 
        y on the circle, there is a non-negative quantity of nectar which may 
        be enjoyed by th ebee without depletion (meaning that hte bee takes a 
        negligilble amount of nectar from the point y, so the bee occupying 
        point y does not cause the amount of nectar at y to change. The nectar 
        at every point y varies with time t according to the prescribed 
        function:
            n(y,t) = 1 + cos(w*t)*sin(2*pi*y)
        (chosen arbitrary) that is unknown to the bee. The amount of nectar 
        enjoyed by the bee at time t is a value that lies in th eeenterval 
        [0, 2]. Time advances in discrete time steps t = 0, 1, 2, ..., and at 
        any time point t a bee at point y observes the quantity of nectar r at 
        point y and nothing else.

        Parameters
        ----------
        steps : int, optional
            The time over which the environment evolves. The default is 250.
        w : float, optional
            Speed at which the environment evolves over time. The default is 0.1.
        c : float, optional
            Grid interval: the step size the agent can possibly make over the 
            circle. The default is 0.01.
        circle_length : int, optional
            The length of a circle. The default is 1.

        Returns
        -------
        None.

        """
        super().__init__(steps)
        self.c = c
        self.circle_length = circle_length
        self.w = w
        
    def create_environment(self):
        """
        Initialisation of ht eenvironment

        Returns
        -------
        None.

        """
        time = np.arange(self.steps)
        point = np.arange(0,self.circle_length,self.c)
        n = np.zeros((time.shape[0], point.shape[0]))
        for i, t in enumerate(time):
            for j, y in enumerate(point):
                n[i,j] = self.environment_evolution(t=t, y=y, w=self.w)
        # print('n: ', n.shape)
        self.n = n
        
    def environment_evolution(self, t:int,y:int,w:float):
        """ 
        Defines the equating for the environment dynamics.
        Parameters
        ----------
        t : int
            Time step.
        y : int
            Point on the circle.
        w : float
            Speed at which the environment evolves over time.

        Returns
        -------
        float
            The amount of honey at point y, time t.

        """
        return 1+(np.cos(t*w)*np.sin(2*np.pi*y))
    
    def environment_ode_sys(self, t, U, params):
        '''
        Defines the environment control problem in terms of the 
        dynamical system of second order ODEs which can be solved 
        with scipy.integrate.odeint

        Parameters
        ----------
        t: array
            A sequence of time points for which to solve for y. The initial 
            value point should be the first element of this sequence. This 
            sequence must be monotonically increasing or monotonically 
            decreasing; repeated values are allowed.
        U : array
            Initial condition on y (can be a vector).
        params : tuple
            Parameters of the system.

        Returns
        -------
        list
            Returns the lyst of the solutions to the ode system determining 
            the location of the bee in the environment.

        '''
        c, w, eps, gamma = params
        y, v, tau  = U
        a = (2*c*np.cos((np.pi*v)/(2*c))**2)/np.pi
        b = (4*c*np.cos(w*tau)*np.cos(2*np.pi*y))/eps
        c = np.log(gamma)*np.tan((np.pi*v)/(2*c))
        dtaudt = 1
        
        dvdt = a*(b+c)
        return [v, dvdt, dtaudt]
    
    def plot_environment(self, steps=250):
        """
        Plots the environment evolution

        Returns
        -------
        None.

        """
        plt.figure(figsize=(15,20))
        plt.imshow(self.n[:steps,:].T, aspect=self.c*50)
        plt.show()
    
    def get_environment(self):
        """
        Returns the values of the environment evolution.

        Returns
        -------
        numpy.array
            The grid assocoated with the evolution of the amount of honey on 
            the circle over time.

        """
        return self.n
    


if __name__=='__main__':
    BeeWorld = BeeWorldEnv()
    BeeWorld.create_environment()
    print(BeeWorld.n.shape)
    BeeWorld.plot_environment(steps=250) 
    
    
    