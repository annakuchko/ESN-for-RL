class AbstractEnv:
    def __init__(self, steps:int):
        """
        Basic class for environments

        Parameters
        ----------
        steps : int
            The number of time steps in the environment evolution..

        Returns
        -------
        None.

        """
        self.steps = steps
    
    def create_environment(self):
        raise NotImplementedError
    
    def environment_evolution(self):
        raise NotImplementedError

    def get_environment(self):
        raise NotImplementedError
        
    def plot_environment(self):
        raise NotImplementedError