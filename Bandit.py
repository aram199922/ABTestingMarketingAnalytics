from abc import ABC, abstractmethod
from logs import *
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    #@abstractmethod
    def __init__(self, p):
        """
        Initialize the EpsilonGreedy arm.

        Parameters:
        p (float): The win rate of the arm.
        """
        self.p = p
        self.p_estimate = 0 
        self.N = 0
        self.r_estimate = 0 
        # Putting 0-s as we don't have prior knowledge

    #@abstractmethod
    def __repr__(self):
        
        """
        For debugging purposes
        """
        return f'An Arm with Win Rate {self.p}'

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    #@abstractmethod
    def report(self, N, results, algorithm = "Epsilon Greedy"):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        
        """
        Save csv-s.

        Parameters:
        N (int): The number of trials in the experiment.
        results (tuple): A tuple of experiment results.
        algorithm (string): Name of the algorithm used.
        """
        if algorithm == 'EpsilonGreedy':
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal = results 
        else:
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward = results 
        
        # Save experiment data to a CSV file
        experiment = pd.DataFrame({
            'Bandit': [bandit for bandit in chosen_bandit],
            'Reward': [re for re in reward],
            'Algorithm': algorithm
        })

        experiment.to_csv(f'{algorithm}_Exp.csv', index=False)

        # Save Final Results to a CSV file
        final = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })


        final.to_csv(f'{algorithm}.csv', index=False)

        for b in range(len(bandits)):
            print(f'Bandit with True Win Rate {bandits[b].p} - Pulled {bandits[b].N} times - Estimated average reward - {round(bandits[b].p_estimate, 4)} - Estimated average regret - {round(bandits[b].r_estimate, 4)}')
        
        
        print(f"Cum. Reward : {sum(reward)}")
        
        print(" ")
        
        print(f"Cum. Regret : {cumulative_regret[-1]}")
              
        print(" ")
        
        if algorithm == 'EpsilonGreedy':                            
            print(f"suboptimal %: {round((float(count_suboptimal) / N), 4)}")


#--------------------------------------#

class Visualization:
    def plot1(self, N, results, algorithm='EpsilonGreedy'):
        """
        Linear and log scale plots of cumulative average reward and optimal reward.
        """
        
        #Retrieving the bandits and Cumulative Average Reward
        
        cumulative_reward_average = results[0]
        bandits = results[3]
        
        ## LINEAR SCALE
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Linear Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.show()

        ## LOG SCALE
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.xscale("log")
        plt.show()

    def plot2(self, results_eg, results_ts):
        """
        Comparison plot
        Comparing cumulative rewards and cumulative regrets for Epsilon-Greedy and Thompson Sampling.
        """
        # Retrieving Cumulative reward and regret
        cumulative_rewards_eps = results_eg[1]
        cumulative_rewards_th = results_ts[1]
        cumulative_regret_eps = results_eg[2]
        cumulative_regret_th = results_ts[2]

        ## Cumulative Reward
        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        ## Cumulative Regret
        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()

class EpsilonGreedy(Bandit):
    ## Epsilon Greedy multibandit

    def __init__(self, p):
        super().__init__(p)

    def pull(self):
        return np.random.randn() + self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = (1 - 1.0/self.N) * self.p_estimate + 1.0/ self.N * x
        self.r_estimate = self.p - self.p_estimate


    def experiment(self, BANDIT_REWARDS, N, t = 1):
        #Initializing Bandits
        bandits = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
        means = np.array(BANDIT_REWARDS)
        true_best = np.argmax(means)  
        count_suboptimal = 0
        EPS = 1/t

        #Keep Track of Which Bandit was chosen and the resulting reward 
        reward = np.empty(N)
        chosen_bandit = np.empty(N)


        for i in range(N):
            p = np.random.random()
            
            if p < EPS:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])

            x = bandits[j].pull()
            
            bandits[j].update(x)
    

            if j != true_best:
                count_suboptimal += 1
            
            reward[i] = x
            chosen_bandit[i] = j
            
            t+=1
            EPS = 1/t

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        
        cumulative_regret = np.empty(N)
        for i in range(len(reward)):
            cumulative_regret[i] = N*max(means) - cumulative_reward[i]

        return cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal



class ThompsonSampling(Bandit):
    """
    A class for implementing the Thompson Sampling algorithm for multi-armed bandit problems.
    """
    
    def __init__(self, p):
        super().__init__(p)
        self.lambda_ = 1
        self.tau = 1


    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.p
    
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate
    
    def update(self, x):
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate
        
    def plot(self, bandits, trial):
        x = np.linspace(-3, 6, 200)
        for b in bandits:
            y = norm.pdf(x, b.p_estimate, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"real mean: {b.p:.4f}, num plays: {b.N}")
            plt.title("Bandit distributions after {} trials".format(trial))
        plt.legend()
        plt.show()

    def experiment(self, BANDIT_REWARDS, N):
        bandits = [ThompsonSampling(m) for m in BANDIT_REWARDS]

        sample_points = [5, 20, 50,100,200,500,1000,1999, 5000,10000, 19999]
        reward = np.empty(N)
        chosen_bandit = np.empty(N)
        
        for i in range(N):
            j = np.argmax([b.sample() for b in bandits])

            if i in sample_points:
                self.plot(bandits, i)

            x = bandits[j].pull()

            bandits[j].update(x)

            reward[i] = x
            chosen_bandit[i] = j

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        
        cumulative_regret = np.empty(N)
        
        for i in range(len(reward)):
            cumulative_regret[i] = N*max([b.p for b in bandits]) - cumulative_reward[i]


        return cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward 
 



def comparison(N, results_eg, results_ts):
    # think of a way to compare the performances of the two algorithms VISUALLY 
    cumulative_reward_average_eg = results_eg[0]
    cumulative_reward_average_ts = results_ts[0]
    bandits_eg = results_eg[3]
    reward_eg = results_eg[5]
    reward_ts = results_ts[5]
    regret_eg = results_eg[2][-1]
    regret_ts = results_ts[2][-1]

    
    print(f"Total Reward Epsilon Greedy : {sum(reward_eg)}")
    print(f"Total Reward Thomspon Sampling : {sum(reward_ts)}")
        
    print(" ")
        
    print(f"Total Regret Epsilon Greedy : {regret_eg}")
    print(f"Total Regret Thomspon Sampling : {regret_ts}")
        

    plt.figure(figsize=(12, 5))

    ## Linear
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_reward_average_eg, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_average_ts, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits_eg]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence  - Linear Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")


    ## Log
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward_average_eg, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_average_ts, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits_eg]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence  - Log Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")
    plt.xscale("log")
    
    
    plt.tight_layout()
    plt.show()
    

    