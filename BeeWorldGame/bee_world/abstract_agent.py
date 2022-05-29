class AbstractAgent:
    
    def __init__(self, step_interval:float, initial_location:float):
        """
        Basic class for agents

        Parameters
        ----------
        step_interval : float
            The interval of moves for the agent.
        initial_location : float
            The relative position of the agent in the environment.

        Returns
        -------
        None.

        """
        self.step_interval = step_interval
        self.initial_location = initial_location
    
    def policy(self):
        raise NotImplementedError
        
        