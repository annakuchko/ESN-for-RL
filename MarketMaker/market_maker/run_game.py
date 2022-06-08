from MarketMaker.market_maker.abstract_game import AbstractGame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    
class MarketMakerGame(AbstractGame):
    def __init__(self, config):
        """
        Class for creting and simulating the Market Naker game 
        (as in https://www.researchgate.net/publication/352738283_Using_Echo_
         State_Networks_to_Approximate_Value_Functions_for_Control) . 

        Market Maker problem is a stochastic control problem. We consider a 
        market maker who provides liquidity to other market participants by 
        quoting prices at which they are willing to sell (ask) and buy (bid) 
        an asset. By setting the ask price higher than the bid price they can 
        profit from the difference when they receive both a buy and sell order 
        at these prices. However the market faces risk, since if they buy a 
        quantity of the asset the mrket price might move against them before 
        they are able to find a seller.  We consider the market maker's 
        inventory to be a stochastic process (Y_t)_t>=0 with dynamics 
        dY_t = pi_t dt + sigma dW_t
        where (W_t)_t>=0 is a standard Brownian motion.
        The parameter sigma measures the volatility of the incoming order flow, 
        (pi_t)_t>=0 is the control process by which the market maker adds drift 
        into their order flow by moving their bid and ask quotes. There is a 
        cost involved in applying the control, and a further cost to holding 
        the inventory away from zero. We use parameters alpha and beta to 
        quantify these effects. Then the Market maker's profit is a stochastic 
        process modelled as 
        dZ_t = (r-alpha pi_t^2 - beta Y_t^2)dt
        where r is the rate of profit the market maker would achieve from the 
        bid-ask spread if they did not have concerns about the asset price 
        movements. We consider the case where the market maker seeks to 
        maximise their long run discounted profit
        v(y) = max_pi E^y [integral_0^inf e^{-delta*t}dZ_t],
        where E^y is the expectation with the process started at Y_0=y. 
        The market maker's value function and optimal control are
        v(y) = -alpha*h*y^2 + (r-alpha*h*sigma^2)/delta,
        pi*(y) = -h*y,
        where 
        h:= (-alpha*delta + sqrt(alpha^2*delta^2 + 4*beta))/(2*alpha)
        The inventory process (Y_t)_t>=0 when controlled by the optimal policy 
        pi*(y) is given by 
        dY_t = -h*Y_t dt + sigma dW_t 
        whose sationary distribution is a Gaussian N(0, sigma^2/2h).

        Parameters
        ----------
        agent : TYPE, optional
            The agent instance. The default is MarketAgent.
        function : TYPE, optional
            The value function approximator. The default is ESN.
        n : int, optional
            The number of time steps in the game. The default is 10000.
        alpha: float, optional
            The cost of operating the control. The default is 1.
        beta: float, optional
            The cost of straying from the origin. The default is 1.
        eps: float, optional
            The time step parameter. the default is 1.
        sigma: float, optional
            The volatility parameter. The default is 1.
        loc: The mean parameter of the normal distribution. The default is 0.
        r: The baseline priofit parameter. The default is 0.
        eta: The constant rate of the exponential drift toward zero. 
        The default is 0.05.
            
        Returns
        -------
        None.

        """
        
        self.config = config
        if config.agent=='MarketAgent':
            from MarketMaker.market_maker.agent import MarketAgent
            self.agent = MarketAgent(
                initial_inventory=config.initial_inventory,
                random_state=config.random_state,
                n_trials=config.n_trials,
                )
        
        if config.function=='ESN':
            from MarketMaker.market_maker.function_approximator import ESN
            self.function = ESN(
                n_inputs=config.n_inputs, 
                n_outputs=config.n_outputs,
                n_reservoir=config.n_reservoir, 
                activation=config.activation,
                two_norm=config.two_norm, 
                sparsity=config.sparsity, 
                random_state=config.random_state, 
                L2=config.L2, 
                gamma=config.gamma, 
                sampling_bounds=config.sampling_bounds
                )
        
        self.n = config.n
        self.alpha = config.alpha
        self.beta = config.beta
        self.eps = config.eps
        self.sigma = config.sigma
        self.loc = config.loc
        self.r = config.r
        self.eta = config.eta
        self.gamma = config.gamma
        self.initial_inventory = config.initial_inventory

        
    def run_game(self, mode):
        self.init_agent()
        self._analytic_solution()
        
        if mode=='init':
            self._init_z_and_inv()
            self._init_upd()
            
        elif mode=='train':
            self._init_z_and_inv()
            self._train_upd()
        
    def _init_upd(self):
        policy_type='init_policy'
        n = self.n
        z = self.z
        rewards = self.rewards
        y_prev = self.initial_inventory
        for t in range(n):
            step = self.Agent.policy(policy_type=policy_type,
                                          cur_inventory=y_prev)
            inv = self.Agent.get_inventory()
            z[t,:] = np.array((inv,step), dtype=object) 
            rewards[t,:] = self._reward_fun(step, inv)
            y_prev = inv
        self.z = z
        self.rewards = rewards.flatten()
        self.est_rewards = self.function.fit(
            inputs=z, 
            outputs=rewards
            ).flatten()
    
    def _train_upd(self):
        policy_type='upd_policy'
        n = self.n
        z = self.z
        rewards = self.rewards
        y_prev = self.initial_inventory
        for t in range(n):
            step = self.Agent.policy(policy_type=policy_type,
                                          cur_inventory=y_prev,
                                          approximator=self.function)
            inv = self.Agent.get_inventory()
            self.est_rewards[t] = self.Agent.est_reward
            z[t,:] = np.array((inv,step), dtype=object) 
            rewards[t,:] = self._reward_fun(step, inv)
            y_prev = inv

        self.z = z
        self.rewards = rewards.flatten()
        
    def _reward_fun(self,step, new_inv):
        return -(self.alpha*step**2 + self.beta*new_inv)
    
    def _analytic_solution(self):
        self.inventory_range = np.arange(-5,5,0.01)
        values_opt = []
        inventory_opt = []
        r = self.r 
        alpha = self.alpha
        gamma = self.gamma
        beta = self.beta
        sigma = self.sigma
        #  optimal policy
        for y in self.inventory_range:
            a = alpha*(gamma-1)+gamma*beta
            b = (np.sqrt((alpha*(gamma-1)+
                          gamma*beta)**2+
                         4*alpha*beta*gamma))
            c = (2*gamma*alpha)
            p = (a+b)/c
            v = -alpha*p*y**2 + (r-gamma*alpha*p*sigma**2)/(1-gamma)
            values_opt.append(v)
            inventory_opt.append(-p*y)
        self.values_opt = values_opt
        self.inventory_opt = inventory_opt

    def _init_z_and_inv(self):
        self.z = np.zeros((self.n,2))
        self.rewards = np.zeros((self.n,1))
      
    def plot_inventory_reward(self,s=250, plot_solution=True):
        plt.figure(figsize=(10,10))
        plt.plot(self.inventory_range, self.values_opt, color = 'green', label='Optimal reward')
        plt.scatter(self.z[:,0], self.rewards, color='orange', label = 'Observed reward')
        plt.scatter(self.z[:,0], self.est_rewards, color='red', s=0.5, label = f'Estimated reward ({self.config.function})')
        plt.xlabel('Inventory')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()
    
    def plot_game(self, s=250):
        plt.figure(figsize=(15,5))
        plt.plot(self.z[:s,0], color='r', label='Inventory levels')
        plt.xlabel('Time')
        plt.ylabel('Inventory')
        plt.legend()
        plt.show()
            
    def plot_rewards(self, s=250):
        plt.figure(figsize=(15,5))
        plt.plot(self.rewards[:s], color='g', label='Observed reward')
        plt.plot(self.est_rewards[:s], color='r', label='Estimated reward')
        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()
        
    def plot_inventory_action(self):
        plt.figure(figsize=(10,10))
        plt.scatter(self.z[:,0], self.z[:,1])
        plt.xlabel('Inventory')
        plt.ylabel('Action')
        plt.show()
        
    def plot_inventory_distribution(self):
        plt.figure(figsize=(10,10))
        sns.distplot(self.z[:,0], kde=True, hist=True, kde_kws={'color':'r'})
        plt.xlabel('Inventory')
        plt.ylabel('Probability density')
        plt.show()        
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Arguments for the BeeWorld game'
        )

    
    # Agent arguments
    
    parser.add_argument('--random_state', default=2022, help='The random state'
                        ' value')
    parser.add_argument('--n_trials', type=int, default=100, help='The number '
                        'of simulated steps to evaluate')
    parser.add_argument('--initial_inventory', type=float, default=0, 
                        help='The initial amount of inventory held by the agent')

    # Game arguments
    parser.add_argument('--agent', type=str, default='MarketAgent', help='The '
                        'name of the agent')
    parser.add_argument('--function',type=str, default='ESN', help=' name of '
                        'the value function approximator')
    parser.add_argument('--n', type=int, default=10000, help='The number of '
                        'time steps in the game')
    
    parser.add_argument('--alpha', type=float, default=1, help='The cost of '
                        'operating the control')
    parser.add_argument('--beta', type=float, default=1, help='The cost of '
                        'straying from the origin')
    parser.add_argument('--eps', type=float, default=1, help='Timestep')
    parser.add_argument('--sigma', type=float, default=1, help='The volatility'
                        ' parameter')
    parser.add_argument('--loc', type=float, default=0, help='The mean '
                        'parameter of the normal distribution')
    parser.add_argument('--r', type=float, default=0, help='The baseline '
                        'profit parameter')
    parser.add_argument('--eta', type=float, default=0.05, help='The constant'
                        ' representing the rate of exponential drift toward 0')

    # function approximator arguments
    parser.add_argument('--n_inputs', default=2, help='The dimensionality of '
                        'the input')
    parser.add_argument('--n_outputs', default=1, help='The dimensionality of '
                        'the output')
    parser.add_argument('--n_reservoir', default=300, help='The size of the '
                        'reservoir')
    parser.add_argument('--activation', default='ReLU', help='The activation '
                        'function')
    parser.add_argument('--two_norm', type=float, default=1, help='The 2-norm '
                        'of the recurrent weight matrix')
    parser.add_argument('--sparsity', type=float, default=0.0,
                        help='The proportion of recurrent wesights set to zero')
    parser.add_argument('--L2', type=float, default=10**-2, 
                        help='The regularisation parameter of the ridge '
                        'regression')
    parser.add_argument('--gamma', type=float, default=np.exp(-1), help='The discount'
                        ' factor of the value function')
    parser.add_argument('--sampling_bounds', type=float, default=0.05, 
                        help='The uniform bounds of the interval from which to'
                        ' sample the random matrices') 

    config = parser.parse_args()
    
    MMGame = MarketMakerGame(config)
    
    MMGame.run_game(mode='init')
    MMGame.plot_inventory_reward(plot_solution=True)
    MMGame.plot_game()
    MMGame.plot_rewards()
    print(f'Avg inventory: {MMGame.z[:,0].mean()}')
    print(f'Avg reward: {MMGame.rewards.mean()}')
    MMGame.plot_inventory_action()
    MMGame.plot_inventory_distribution()
    
    MMGame.run_game(mode='train')
    MMGame.plot_inventory_reward(plot_solution=True)
    MMGame.plot_game()
    MMGame.plot_rewards()
    print(f'Avg inventory: {MMGame.z[:,0].mean()}')
    print(f'Avg reward: {MMGame.rewards.mean()}')
    MMGame.plot_inventory_action()
    MMGame.plot_inventory_distribution()
        
