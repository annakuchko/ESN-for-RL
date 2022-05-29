import numpy as np
from BeeWorldGame.bee_world.abstract_agent import AbstractAgent

class BeeAgent(AbstractAgent):
    def __init__(self, step_interval:float=0.1,
                 initial_location:float=0.5,
                 random_state:int=2022,
                 n_trials:int=100,
                 c:float=0.01):
        """
        The class for the Bee Agent in the game Bee World

        Parameters
        ----------
        environment : TYPE
            The environment instance.
        agent : TYPE
            The agent instance.
        function : TYPE
            The value function approximator.
        n : int
            The number of time steps in the game.
        c : float, optional
            The grid of the game environment. The default is 0.01.

        Returns
        -------
        None.
        

        Parameters
        ----------
        step_interval : float, optional
            The interval of the agent moves in the environment. The default 
            is 0.1.
        initial_location : float, optional
            The relative starting position of the agent in the environment. 
            The default is 0.5.
        random_state : int, optional
            The random state value. The default is 2022.
        n_trials : int, optional
            The number of simulated steps to evaluate. For a given reservoir
            state we consider n_trials actions uniformely sampled from over
            (-step_interval, step_interval), then for each action we consider 
            the reward-action pairs where the reward for each pair is the 
            current reward; and is therefore the same in every pair. Then we 
            compute the next reservoir state for each pair and estimate the 
            value of executing each action. Then we choose to execute the 
            action with the greatest estimated value - which determines the 
            new policy. The action-reward pair with the greatest reward is then 
            used to update the reservoir state.The default is 100.
        c : float, optional
            The grid of the game environment. The default is 0.01.

        Raises
        ------
        Exception
            Exception for the invalid random seed..

        Returns
        -------
        None.

        """
        
        
        self.step_interval = step_interval
        self.location = initial_location 
        self.reward = None
        self.random_state = random_state
        self.n_trials = n_trials
        self.c = c

        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand
    
    def policy(self, policy_type='uniform',
               approximator=None,
               cur_reward=None,
               stochastic=False):
        
        step_interval = self.step_interval
        n_trials = self.n_trials
        
        if policy_type=='uniform':
            step = self.random_state_.uniform(-step_interval, step_interval)
            self.make_step(step)
        
        elif policy_type=='upd_policy':
            z = self._simulate_value_fun(step_interval=step_interval, 
                                          n_trials=n_trials, 
                                          cur_reward=cur_reward)
            # z = np.array(((cur_reward,-self.step_interval),(cur_reward,self.step_interval)))
            val = approximator.predict(z,upd=False).flatten().reshape(1,-1)[0]
            step = self._get_max_val_step(z=z, val=val, 
                                          cur_reward=cur_reward, 
                                          approximator=approximator)
            approximator.predict(np.array((cur_reward,step), dtype=object).reshape(1,-1),upd=True)
        return step
    
    def _get_max_val_step(self, z, val, cur_reward, approximator):
        self.est_reward = max(val)
        # print('self.est_reward: ', self.est_reward)
        step = z[np.where(val==max(val)),1]
        new_z = np.array((cur_reward, step), dtype=object).reshape(1,-1)
        approximator.predict(inputs=new_z, upd=True).reshape(1,-1)
        self.make_step(step)
        return step 
    
    def _simulate_value_fun(self, step_interval, n_trials, cur_reward):
        z = np.zeros((n_trials,2))
        for n in range(n_trials):
            step = self.random_state_.uniform(-step_interval, step_interval)
            z[n,1] = step
            z[n,0] = cur_reward
        return z
    
    def make_step(self, step):
        new_loc = self.location+step
        if new_loc < 0:
            new_loc = 1-self.c
        elif new_loc > 1-self.c:
            new_loc = 0
        self.location = new_loc
    
    def get_location(self):
        loc = (1/self.c) * self.location
        return int(loc)

