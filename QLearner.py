""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Joshua Lutkemuller (replace with your name)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT User ID: jlutkemuller3 (replace with your User ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 904051695(replace with your GT ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import random as rand  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np



class QLearner(object):  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This is a Q learner object.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param num_states: The number of states to consider.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type num_states: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param num_actions: The number of actions available..  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type num_actions: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type alpha: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type gamma: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type rar: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type radr: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type dyna: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    def __init__(  		  	   		 	 	 			  		 			 	 	 		 		 	
        self,  		  	   		 	 	 			  		 			 	 	 		 		 	
        num_states=100,  		  	   		 	 	 			  		 			 	 	 		 		 	
        num_actions=4,  		  	   		 	 	 			  		 			 	 	 		 		 	
        alpha=0.2,  		  	   		 	 	 			  		 			 	 	 		 		 	
        gamma=0.9,  		  	   		 	 	 			  		 			 	 	 		 		 	
        rar=0.5,  		  	   		 	 	 			  		 			 	 	 		 		 	
        radr=0.99,  		  	   		 	 	 			  		 			 	 	 		 		 	
        dyna=0,  		  	   		 	 	 			  		 			 	 	 		 		 	
        verbose=False,  		  	   		 	 	 			  		 			 	 	 		 		 	
    ):  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        Class `QLearner` Constructor method  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        self.verbose = verbose  		  	   		 	 	 			  		 			 	 	 		 		 	
        self.number_of_actions = num_actions
        self.number_of_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.Q = np.zeros((num_states, num_actions))
        self.s = 0  #initialize to 0
        self.a = 0  #initialize to 0

        self.dynaQ_storage = []
  		  	   		 	 	 			  		 			 	 	 		 		 	
    def querysetstate(self, s):  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        Update the state without updating the Q-table  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
        :param s: The new state  		  	   		 	 	 			  		 			 	 	 		 		 	
        :type s: int  		  	   		 	 	 			  		 			 	 	 		 		 	
        :return: The selected action  		  	   		 	 	 			  		 			 	 	 		 		 	
        :rtype: int  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        self.s = s

        if rand.random() < self.alpha:
            action = rand.randint(0, self.number_of_actions-1)
        else:
            action = np.argmax(self.Q[s, :])
        self.a = action

        if self.verbose:  		  	   		 	 	 			  		 			 	 	 		 		 	
            print(f"s = {s}, a = {action}")  		  	   		 	 	 			  		 			 	 	 		 		 	
        return action  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    def query(self, s_prime, r):  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        Update the Q table and return the last action  		  	   		 	 	 			  		 			 	 	 		 		 	
        :param s_prime: The new state  		  	   		 	 	 			  		 			 	 	 		 		 	
        :type s_prime: int  		  	   		 	 	 			  		 			 	 	 		 		 	
        :param r: The immediate reward  		  	   		 	 	 			  		 			 	 	 		 		 	
        :type r: float  		  	   		 	 	 			  		 			 	 	 		 		 	
        :return: The selected action  		  	   		 	 	 			  		 			 	 	 		 		 	
        :rtype: int     		 	 	 			  		 			 	 	 		 		 			  	   		 	 	 			  		 			 	 	 		 		 	
        """
        # update Q Table with the `actual` experience/action
        old_q_value = self.Q[self.s,self.a]
        max_q_value = np.max(self.Q[s_prime, :])
        self.Q[self.s,self.a] = old_q_value + self.alpha * (r + self.gamma * max_q_value - old_q_value)

        # Save experience for Dyna
        if self.dyna > 0:
            self.dynaQ_storage.append((self.s, self.a, s_prime, r))

        # Dyna hallucinations
        for _ in range(self.dyna):
            s_dyna, a_dyna, s_prime_dyna, r_dyna = rand.choice(self.dynaQ_storage)
            old_q_dyna = self.Q[s_dyna, a_dyna]
            max_q_dyna = np.max(self.Q[s_prime_dyna, :])
            self.Q[s_dyna, a_dyna] = old_q_dyna + self.alpha * (r_dyna + self.gamma * max_q_dyna - old_q_dyna)

        # Decay exploration rate with radr
        self.rar *= self.radr

        # Choose next action
        if rand.random() < self.rar:
            action = rand.randint(0, self.number_of_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime, :])

        if self.verbose:
            print(f"query: s={self.s}, a={self.a}, r={r}, s'={s_prime}, a'={action}")

        self.s = s_prime
        self.a = action
        return action  # return the last action

    def plot_q_heatmap(self, action=0, grid_size=(10, 10), title_prefix="Q-Value for Action"):
        """
        Heatmap of Q-values for a specific action (0=N, 1=E, 2=S, 3=W).
        """
        import matplotlib.pyplot as plt
        q_values = np.zeros(grid_size)
        for state in range(self.number_of_states):
            row, col = divmod(state, grid_size[1])
            q_values[row, col] = self.Q[state, action]

        plt.imshow(q_values, cmap='viridis', origin='upper')
        plt.colorbar(label='Q-Value')
        plt.title(f'{title_prefix} {action}')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show()
    def plot_policy(self, grid_size=(10, 10), title="Greedy Policy"):
        """
        Visualize the greedy policy using directional arrows for 8 actions.
        """
        import matplotlib.pyplot as plt

        arrows = {
            0: '↑',  # N
            1: '↗',  # NE
            2: '→',  # E
            3: '↘',  # SE
            4: '↓',  # S
            5: '↙',  # SW
            6: '←',  # W
            7: '↖'   # NW
        }
        policy_grid = np.full(grid_size, ' ')
        for state in range(self.number_of_states):
            row, col = divmod(state, grid_size[1])
            best_action = np.argmax(self.Q[state])
            policy_grid[row, col] = arrows.get(best_action, '?')

        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.axis('off')
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                ax.text(j, i, policy_grid[i, j], ha='center', va='center', fontsize=14)
        plt.gca().invert_yaxis()
        plt.show()


    def study_group(self):
        """
        Returns
            A comma separated string of GT_Name of each member of your study group
            # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone

        Return type
            str
        """
        return "jlutkemuller3"

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jlutkemuller3"  # Change this to your user ID
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	

    ql = QLearner(num_actions = 8,verbose=False)
    if ql.verbose:
        print("Remember Q from Star Trek? Well, this isn't him")
        ql.plot_policy()
        ql.plot_q_heatmap()