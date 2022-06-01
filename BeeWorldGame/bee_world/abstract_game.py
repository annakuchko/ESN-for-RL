class AbstractGame:
    def __init__(self, environment, agent, function, n:int, c:float=0.01):
        """
        Basic class for games

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

        """
        self.environment = environment
        self.agent = agent
        self.n = n
        self.function = function
        self.c = c
        
    def init_agent(self):
        self.Agent = self.agent()
    
    def init_environment(self):
        self.Environment = self.environment()
        self.Environment.create_environment()
    
    def run_game(self, mode):
        self.init_agent()
        self.init_environment()
        
    def plot_game(self):
        raise NotImplementedError
        
    def plot_reward(self):
        raise NotImplementedError

