import numpy as np
from MarketMaker.market_maker.abstract_agent import AbstractAgent

class MarketAgent(AbstractAgent):
    def __init__(self, initial_inventory:float=0.0,
                 random_state:int=2022,
                 n_trials:int=100,
                 eta:float=0.05,
                 sigma:float=1,
                 loc:float=0.0):
        """
        The class for the Market Agent in the game Market Maker

        Parameters
        ----------
        initial_inventory : float, optional
            The initial level of inventory held by the agent.
            The default is 0.0.
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
        eta : float, optional
            A constant representing the rate of exponential drift toward 0. 
            The default is 0.05.
        sigma: float, optional
            The volatility parmaeter of the normal distribution from which we 
            sample the policy. The default is 1. 
        loc: float, optional
            The parameter representing the mean of the normal distribution 
            from which we sample the strategy. The default is 0.0.

        Returns
        -------
        None.

        """
        
        
        self.initial_inventory = initial_inventory
        self.inventory = initial_inventory 
        self.reward = None
        self.random_state = random_state
        self.n_trials = n_trials
        self.eta = eta
        self.sigma = sigma
        self.loc = loc
        
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand
    
    def policy(self, policy_type='opt_policy',
               approximator=None,
               cur_inventory=None,
               stochastic=False):
        
        n_trials = self.n_trials

        if policy_type=='init_policy':
            step = self.random_state_.normal(loc=self.loc, scale=self.sigma) #- self.eta*cur_inventory
            self.make_step(step, cur_inventory)
        
        elif policy_type=='upd_policy':
            z = self._simulate_inv_step_pairs(n_trials=n_trials, 
                                          cur_inventory=cur_inventory)
            val = approximator.predict(z,upd=False).reshape(1,-1)[0]
            step = self._get_max_val_step(z=z, val=val, 
                                          cur_inventory=cur_inventory, 
                                          approximator=approximator)
            self.make_step(step, cur_inventory)
            approximator.predict(np.array((cur_inventory,step), dtype=object).reshape(1,-1),upd=True)
        return step
    
    def _get_max_val_step(self, z, val, cur_inventory, approximator):
        min_abs_id = np.where(val==max(val))
        self.est_reward = val[min_abs_id]
        step = z[min_abs_id,1][0][0]
        new_z = np.array((cur_inventory, step), dtype=object).reshape(1,-1)
        approximator.predict(inputs=new_z, upd=True).reshape(1,-1)
        return step 
    
    def _simulate_inv_step_pairs(self, n_trials, cur_inventory):
        z = np.zeros((n_trials,2))
        for n in range(n_trials):
            z[n,0] = cur_inventory
            z[n,1] = self.random_state_.normal(loc=self.loc, scale=self.sigma) #- self.eta*cur_inventory
        return z
    
    def make_step(self, step, cur_inventory):
        self.inventory = step - self.eta*cur_inventory
        
    def get_inventory(self):
        return self.inventory

