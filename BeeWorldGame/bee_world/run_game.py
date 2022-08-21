from BeeWorldGame.bee_world.abstract_game import AbstractGame
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
class BeeWorldGame(AbstractGame):
    def __init__(self, config):
        """
        Class for creting and simulating the Bee World game 
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
        point y and nothing else. Having made this observation, the bee may 
        choose to move anywhere in th einterval (y-c, y+c) for some fixed 
        0<c<1 and arrive at its chosen destination at time t+1. The interval
        of possible moves (-c,c) is called the action space. The goal of the
        game is to devise the policy whereby, given all its previous 
        obsevations, the bee makes a decision as to where to move next, such 
        that the discounted sum over all future nectar is as great as possible. 
        The space of all previous (reward, action) pairs is contained by the 
        space of bi-infinie sequences. The agent playing Bee World either makes 
        no observations beyond the reward (nectar) and actions, or makes its 
        decisions based on a left-infinite sequence of 
        (reward, action, observation) triples.
        The policy adopted by the bee may be realised as a deterministic policy 
        for which the bee axecutes an action a determined by the history of 
        (reward, action) pairs / (reward, action, observation) triples. 
        Alternatively, the bee may adopt a stochastic policy, for which every 
        state history of (reward, action) pairs / (reward, action, observation) 
        triples admits a distribution over actions from which the bee makes a 
        random choice.
        Parameters
        ----------
        environment : TYPE, optional
            The environment instance. The default is BeeWorldEnv.
        agent : TYPE, optional
            The agent instance. The default is BeeAgent.
        function : TYPE, optional
            The value function approximator. The default is ESN.
        n : TYPE, optional
            The number of time steps in the game. The default is 2000.
        Returns
        -------
        None.
        """
        if config.environment=='BeeWorldEnv':
            from BeeWorldGame.bee_world.environment import BeeWorldEnv
            self.environment = BeeWorldEnv(
                steps=config.steps, 
                w=config.w, 
                c=config.c,
                circle_length=config.circle_length
                )
        
        if config.agent=='BeeAgent':
            from BeeWorldGame.bee_world.agent import BeeAgent
            self.agent = BeeAgent(
                step_interval=config.step_interval,
                initial_location=config.initial_location,
                random_state=config.random_state,
                n_trials=config.n_trials,
                c=config.c
                )
        
        if config.function=='ESN':
            # from BeeWorldGame.bee_world.function_approximator import ESN
            from BeeWorldGame.bee_world.function_approximator_no_bias import ESN
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
                sampling_bounds=config.sampling_bounds,
                rho=config.rho,
                xi=config.xi,
                estimator=config.estimator
                )
        
        self.n = config.n
        self.c = config.c
        
        
    def init_agent(self):
        self.Agent = self.agent
    
    def init_environment(self):
        self.Environment = self.environment
        self.Environment.create_environment()
        
    def run_game(self, mode):
        self.init_agent()
        self.init_environment()
        
        if mode=='init':
            self._init_z_end_loc()
            self._init_upd()
            
        elif mode=='train':
            self._init_z_end_loc()
            self._train_upd()
            
        elif mode=='solution':
            self._init_z_end_loc()
            self._analytic_solution()

        
    def _init_upd(self):
        policy_type='uniform'
        n = self.n
        z = self.z
        env = self.Environment.n
        locations = self.locations
        for t in range(n):
            step = self.Agent.policy(policy_type=policy_type)
            loc = self.Agent.get_location()
            reward = env[t,loc].copy()
            z[t,:] = np.array((reward,step)) 
            locations[t,:] = int(loc)
        
        self.z = z
        self.est_rewards = self.function.fit(
            inputs=z[:,:].copy(), 
            outputs=z[:,0].copy().reshape(-1,1)
            ).flatten()
        self.locations = locations.flatten()
        self.train_rmse = mse(self.z[:,0], self.est_rewards, squared=False)
        
    def _train_upd(self):
        policy_type='upd_policy'
        n = self.n
        z=self.z
        env = self.Environment.n
        locations = self.locations
        r0 = z[-1,0].copy()
        est_rewards = []
        for t in range(n):
            step = self.Agent.policy(policy_type=policy_type,
                                          cur_reward=r0,
                                          approximator=self.function)
            loc = self.Agent.get_location()
            est_rewards.append(self.Agent.est_reward)
            reward = env[t, loc].copy()
            # r0 = self.Agent.est_reward.copy()
            r0 = reward.copy()
            z[t,:] = np.array((reward,step), dtype=object) 
            locations[t,:] = int(loc)

        self.est_rewards = np.array(est_rewards).copy()
        self.z = z.copy()
        self.locations = locations.flatten().copy()
        self.test_rmse = mse(self.z[:,0], self.est_rewards, squared=False)
        
        
        
    def _analytic_solution(self, y_init=0, v_init=0, tau_init=0, eps=10**-5):
        fun = self.Environment.environment_ode_sys
        c = self.c
        n = self.n
        env = self.Environment.n
        sol = odeint(func=fun,
                    t = np.arange(0, n, c),
                    y0=[y_init, v_init, tau_init],
                    args = ([self.Agent.step_interval,
                              self.Environment.w,
                              eps,
                              self.function.gamma],),
                    tfirst=True
                    )
        loc = sol[0:int(n/c):int(1/c),0]
        loc = self._correct_analytic_trajectory(loc, c)
        
        for i, el in enumerate(loc):
            self.z[i,:] = np.array((env[i, el], 0), dtype=object) 
        
        self.locations = loc
        
    def _init_z_end_loc(self):
        self.z = np.zeros((self.n,2))
        self.locations = np.zeros((self.n,1))
    
    def _correct_analytic_trajectory(self, loc, c):
        for i, l in enumerate(loc):
            if l<0:
                loc[i] = 1-c
                loc[i+1:] = 1-c+loc[i+1:]
            elif l>1-c:
                loc[i] = 0
                loc[i+1:] = loc[i+1:]-(1-c)
            else:
                loc[i] = l
       
        loc = (loc/c).astype(int)
        return loc
    
    # def plot_rewards(self,s=250, solution=False):
    #     plt.figure(figsize=(25,10))
    #     plt.plot(self.z[-s-1:-1,0], color='g', label='Observed reward',linewidth=2)
    #     if not solution:
    #         states = self.function.pred_states_memory
    #         new_states=np.zeros_like(states)
    #         for n in range(states.shape[0]-1):
    #             new_states[n,:] = states[n,:] - self.function.gamma*states[n+1,:]
    #         W = self.function.W
    #         pred_rew = W.T@np.hstack((np.ones(new_states.shape[0]).reshape(-1,1), 
    #                                   new_states.copy())).T
    #         sb=self.function.sampling_bounds
    #         pred_rew_unsc = pred_rew.copy()-(-sb)/(sb-(-sb))*(self.function.max[0] - self.function.min[0]) + self.function.min[0]
    #         plt.plot(pred_rew_unsc[0,-s:], color='r', label='Estimated reward',linewidth=1.2)
    #     plt.xticks(fontsize=24)
    #     plt.xlabel('t', size=24, rotation=0)
    #     plt.yticks(fontsize=24)
    #     plt.ylabel('n(y,t)', size=24)
    #     plt.legend(fontsize=24)
    #     plt.show()
    
    # def plot_game(self, s=250):
    #     locations = self.locations
    #     env = self.Environment.n
    #     plt.figure(figsize=(25,10))
    #     plt.imshow(env[-s:,:].T, aspect=self.c*60,origin='lower')
    #     plt.scatter(range(s),locations[-s:], 
    #                 color='white',s=80)
    #     plt.xticks(fontsize=24)
    #     plt.xlabel('t', size=24, rotation=0)
    #     plt.yticks(fontsize=24)
    #     plt.ylabel('y', size=24, rotation=0)
    #     # plt.colorbar(shrink=0.5)
    #     cbar = plt.colorbar(shrink=0.5)
    #     for t in cbar.ax.get_yticklabels():
    #         t.set_fontsize(24)
    #     plt.show()
    def plot_rewards(self,s=250, solution=False):
        plt.figure(figsize=(25,10))
        plt.plot(self.z[1:s+1,0], color='g', label='Observed reward',linewidth=2)
        if not solution:
            states = self.function.pred_states_memory
            new_states=np.zeros_like(states)
            for n in range(states.shape[0]-1):
                new_states[n,:] = states[n,:] - self.function.gamma*states[n+1,:]
            W = self.function.W
            pred_rew = W.T@np.hstack((np.ones(new_states.shape[0]).reshape(-1,1), 
                                      new_states.copy())).T
            sb=self.function.sampling_bounds
            pred_rew_unsc = pred_rew.copy()-(-sb)/(sb-(-sb))*(self.function.max[0] - self.function.min[0]) + self.function.min[0]
            plt.plot(pred_rew_unsc[0,:s], color='r', label='Estimated reward',linewidth=1.2)
        plt.xticks(fontsize=24)
        plt.xlabel('t', size=24, rotation=0)
        plt.yticks(fontsize=24)
        plt.ylabel('n(y,t)', size=24)
        plt.legend(fontsize=24)
        plt.show()
    
    def plot_game(self, s=250):
        locations = self.locations
        env = self.Environment.n
        plt.figure(figsize=(25,10))
        plt.imshow(env[:s,:].T, aspect=self.c*60,origin='lower')
        plt.scatter(range(s),locations[:s], 
                    color='white',s=80)
        plt.xticks(fontsize=24)
        plt.xlabel('t', size=24, rotation=0)
        plt.yticks(fontsize=24)
        plt.ylabel('y', size=24, rotation=0)
        # plt.colorbar(shrink=0.5)
        cbar = plt.colorbar(shrink=0.5)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(24)
        plt.show()
        
if __name__=='__main__':
    from sklearn.model_selection import ParameterGrid
    
    param_grid = {
    'estimator': ['ridge'],
    'L2':[0.01,0.1,1,5,10],
    'gamma': [0.5,0.65,0.75,0.85,0.95], 
    'n_reservoir': [300],  
    'sparsity': [0.0,0.25,0.5,0.95], 
    'two_norm': [0.9], 
    'activation': ['Tanh'],
    'rho': [0.99],
    'xi': [0.25,0.5,0.65,0.85]
}
# param_grid = {
#                     'estimator': ['ridge'],
#                     'L2':[0],
#                   'gamma': [.5], 
#                   'n_reservoir': [300],  
#                   'sparsity': [0.0], 
#                   'two_norm': [0.9], 
#                   'activation': ['Tanh'],
#                   'rho': [0.95],
#                   # 'xi': [0.5]
#                    'xi': [0.5]
#                   }
    
    grid = ParameterGrid(param_grid)
    
    
    def argpars_compose(params):
        import argparse
        parser = argparse.ArgumentParser(
            description='Arguments for the BeeWorld game'
            )
    
        # Agent arguments
        parser.add_argument('--step_interval', type=float, default=0.1, 
                            help='The interval of the agent moves in the '
                            'environment')
        parser.add_argument('--initial_location', type=float, default=0.0, 
                            help='The relative starting position of the agent in '
                            'the environment')
        parser.add_argument('--random_state', default=2022, help='The random state'
                            ' value')
        parser.add_argument('--n_trials', type=int, default=100, help='The number '
                            'of simulated steps to evaluate')
        # Environment arguments
        parser.add_argument('--steps', type=int, default=2000, help='The time over'
                            ' which the environment evolves')
        parser.add_argument('--w', type=float, default=0.1, help='Speed at which'
                            ' the environment evolves over time')
        parser.add_argument('--circle_length', type=int, default=1, 
                            help='The length of a circle')
        parser.add_argument('--c', type=float, default=0.01, help='The grid of the'
                            ' game environment')
        # Game arguments
        parser.add_argument('--environment', type=str, default='BeeWorldEnv',
                            help='The name of the environment')
        parser.add_argument('--agent', type=str, default='BeeAgent', help='The '
                            'name of the agent')
        parser.add_argument('--function',type=str, default='ESN', help=' name of '
                            'the value function approximator')
        parser.add_argument('--n', type=int, default=2000, help='The number of '
                            'time steps in the game')
        # function approximator arguments
        parser.add_argument('--n_inputs', default=2, help='The dimensionality of '
                            'the input')
        parser.add_argument('--n_outputs', default=1, help='The dimensionality of '
                            'the output')
        parser.add_argument('--n_reservoir', default=params['n_reservoir'], help='The size of the '
                            'reservoir')
        parser.add_argument('--activation', default=params['activation'], help='The activation '
                            'function')
        parser.add_argument('--two_norm', type=float, default=params['two_norm'], help='The 2-norm '
                            'of the recurrent weight matrix')
        parser.add_argument('--sparsity', type=float, default=params['sparsity'], 
                            help='The proportion of recurrent weights set to zero')
        parser.add_argument('--L2', type=float, default=params['L2'], 
                            help='The regularisation parameter of the ridge '
                            'regression')
        parser.add_argument('--gamma', type=float, default=params['gamma'], help='The discount'
                            ' factor of the value function')
        parser.add_argument('--sampling_bounds', type=float, default=1, 
                            help='The uniform bounds of the interval from which to'
                            ' sample the random matrices') 
        
        parser.add_argument('--rho', type=float, default=params['rho'], help='The scaling '
                            'factor for the input weights matrix')
        
        parser.add_argument('--xi', type=float, default=params['xi'], help='The scaling '
                            'factor for the states weights matrix')
    
        parser.add_argument('--estimator', default=params['estimator'])
        config = parser.parse_args()
        return config
    
    k =  len([i for i in grid])
    results = {}
    vifs = []
    scores = []
    sparsity = []
    trials = 1
    for t in np.arange(trials):
        print(f'Trial {t}/{trials}')
        for i, params in enumerate(grid):
            print(f'{i+1}/{k}')
            
            config = argpars_compose(params)
            BeeGame = BeeWorldGame(config)
            
            BeeGame.run_game(mode='init')
            BeeGame.plot_rewards()
            BeeGame.plot_game()
            print(BeeGame.z[:,0].mean())
            
            BeeGame.run_game(mode='train')
            BeeGame.plot_rewards()
            BeeGame.plot_game()
            # print(BeeGame.z[:,0].mean())
            score = BeeGame.z[:,0].mean()
            print(f'{params}|| score: {score}')
            results[str(params)] = score
            vifs.append(BeeGame.function.VIF)
            sparsity.append(BeeGame.function.sparsity)
            scores.append(score)
        best_score = max([i for i in results.values()])
        best_res = [n for n, v in results.items() if v == best_score]
        # print(f'Best score: {best_score} for params: {best_res}')
        print(f'Best score: {best_score}')
        # mse = np.sqrt(()**2)
    # plt.scatter(np.array(vifs), np.array(scores))
    # plt.xlabel('Mean VIF')
    # plt.ylabel('Mean score')
    # plt.show()
    
    
    # plt.scatter(np.array(sparsity), np.array(scores))
    # plt.xlabel('sparsity')
    # plt.ylabel('Mean score')
    # plt.show()
    # BeeGame.run_game(mode='solution')
    # BeeGame.plot_rewards(solution=True)
    # BeeGame.plot_game()
    # print(BeeGame.z[:,0].mean())

        
        