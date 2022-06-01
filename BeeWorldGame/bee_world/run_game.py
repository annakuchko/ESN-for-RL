from BeeWorldGame.bee_world.agent import BeeAgent
from BeeWorldGame.bee_world.environment import BeeWorldEnv
from BeeWorldGame.bee_world.function_approximator import ESN
from BeeWorldGame.bee_world.abstract_game import AbstractGame
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
    
class BeeWorldGame(AbstractGame):
    def __init__(self, environment=BeeWorldEnv, 
                 agent=BeeAgent, 
                 function=ESN,
                 n=2000,
                 c=0.01):
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

        super().__init__(environment, agent, function, n, c)    
        self.function = function(2,1)
        
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
            reward = env[t,loc]
            z[t,:] = np.array((reward,step)) 
            locations[t,:] = int(loc)
        
        self.z = z
        self.est_rewards = self.function.fit(
            inputs=z[:,:], 
            outputs=z[:,0].reshape(-1,1)
            ).flatten()
        self.locations = locations.flatten()
        
    def _train_upd(self):
        policy_type='upd_policy'
        n = self.n
        z=self.z
        env = self.Environment.n
        locations = self.locations
        r0 = z[-1,0]
        est_rewards = []
        for t in range(n):
            step = self.Agent.policy(policy_type=policy_type,
                                          cur_reward=r0,
                                          approximator=self.function)
            loc = self.Agent.get_location()
            est_rewards.append(self.Agent.est_reward)
            reward = env[t, loc]
            r0 = reward
            z[t,:] = np.array((reward,step), dtype=object) 
            locations[t,:] = int(loc)

        self.est_rewards = np.array(est_rewards)
        self.z = z
        self.locations = locations.flatten()
        
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
    
    def plot_rewards(self,s=250, solution=False):
        plt.figure(figsize=(15,5))
        plt.plot(self.z[:s,0], color='g')
        if not solution:
            plt.plot(self.est_rewards[:s], color='r')
        plt.show()
    
    def plot_game(self, s=250):
        locations = self.locations
        env = self.Environment.n
        plt.figure(figsize=(15,5))
        plt.imshow(env[:s,:].T, aspect=self.c*60,origin='lower')
        plt.scatter(range(s),locations[:s], 
                    color='white',s=50)
        plt.show()
        

        
if __name__=='__main__':
    BeeGame = BeeWorldGame()
    
    BeeGame.run_game(mode='init')
    BeeGame.plot_rewards()
    BeeGame.plot_game()
    print(BeeGame.z[:,0].mean())
    
    BeeGame.run_game(mode='train')
    BeeGame.plot_rewards()
    BeeGame.plot_game()
    print(BeeGame.z[:,0].mean())
    
    BeeGame.run_game(mode='solution')
    BeeGame.plot_rewards(solution=True)
    BeeGame.plot_game()
    print(BeeGame.z[:,0].mean())
    
    # BeeGame.analytic_bee_world()
    # BeeGame.plot_game()
    # print(BeeGame.rewards.mean())
    

        
        
