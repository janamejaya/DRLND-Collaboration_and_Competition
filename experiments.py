import numpy as np
import random
import time
from collections import deque
import torch
from unityagents import UnityEnvironment

class Continuous_Action_Experiments():
    """Interacts with and learns from the environment using discrete actions."""
    
    def __init__(self, nagents, nruns, nepisodes, maxt,
                 current_agent, env_file_name, current_env, target_score, num_episodes_score_avg):
        """Initialize a Discrete Experiment object.
        Params
        ======
            nagents (int):         Number of non-interacting agents
            nruns (int):           Number of times the experiment will be run
            nepisodes (int):       Number of episodes in each run
            maxt (int):            Maximum number of steps per episode
            current_agent (Class): Selected agent
            current_env (Class):   Selected_environment
            target_score (float):  Target score to be achieved for successful run
            num_episodes_score_avg(int): number of scores over which a running average will be monitored
        """
        self.num_agents = nagents
        self.nruns = nruns
        self.nepisodes = nepisodes
        self.maxt = maxt
        self.agent = current_agent
        self.env_file = env_file_name
        self.env = current_env,
        self.brain_name = self.env[0].brain_names[0]
        #print('self.brain_name = ', self.brain_name)
        self.target_score = target_score
        self.num_episodes_score_avg = num_episodes_score_avg
        #print('brain_name = ', self.env[0].brain_names[0])
        
        print('Continuous Action Experiment initialized')

    def execute_one_episode(self):
        env_info = self.env[0].reset(train_mode=True)[self.brain_name]
        
        # Get the next state. This is an array of size number_of_agents X number_of_environment_variables per agent
        # Concatenate the state observation for each agent and prepare a batch of size 1
        states = np.reshape(env_info.vector_observations, (1,-1))
                    
        # Initialize the score for each agent
        scores = np.zeros(self.num_agents)
        
        # Reset the agents noise
        self.agent.reset_noise()
        
        # While the episode has not terminated
        while True:
            # For each agent, predict the action based on the full states vector
            # then combine them into a 1-D action vector
            actions = self.agent.act(states)
            
            # Take the action and collect the next state, reward and terminal state status
            env_info = self.env[0].step(actions)[self.brain_name]
            
            # Get the next state, reward, and episode termination info
            next_states = np.reshape(env_info.vector_observations, (1,-1))
            rewards = env_info.rewards                        # get the reward
            dones = env_info.local_done                       # see if episode has finished
            
            # Update the agents policy and action-value networks
            self.agent.step(states, actions, rewards, next_states, dones)
            
            # Update state information for each agent
            states = next_states
            
            # Add reward to the score
            scores += rewards
            
            # If any of the 20 agents terminates, the episode ends
            if np.any(dones):
                break
        return scores
    
    def execute_one_run(self, runid):
        """
        Execute one run with several episodes until the average score over the last num_episodes_score_avg
        exceeds the target value of target_score
        """
        # Initialize scores
        scores = []
        scores_window = deque(maxlen=self.num_episodes_score_avg)
         
        # Start the timer
        done_flag = 0
        start_time=time.time()
        print('\n')
        for episode_num in range(self.nepisodes): 
            # Run one episode and return the score for that episode
            # score corresponds to the total reward
            current_scores = self.execute_one_episode()
            #print('episode_num = ', episode_num, ' current score = ',current_score)
            
            # Find the maximum score from any agent
            max_score_per_agent = np.max(current_scores)

            # Append current_score to the scores list
            scores.append(max_score_per_agent)
            scores_window.append(max_score_per_agent)
            
            # Get the average score from scores_window
            avg_score = np.mean(scores_window)
            #print('episode : ',episode_num, ' max_score = ',max_score_per_agent, ' avg score = ', avg_score)
            
            # Show the average score every num_episodes_score_avg time steps
            if episode_num%self.num_episodes_score_avg==0:
                print('Run {:d} \tEpisode_Num {:d} \tAverage Score: {:.3f}'.format(runid, episode_num, avg_score))

            # If avg_score exceeds the target score
            if avg_score>=self.target_score:
                end_time = time.time()
                
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f} \tTotal time/seconds: {:.3f}'.format(episode_num, avg_score, end_time-start_time))
                
                self.agent.save_model(runid)
                #torch.save(self.agents.actor_local.state_dict(), 'checkpoint_actor_local'+str(runid)+'.pth')
                #torch.save(self.agents.critic_local.state_dict(), 'checkpoint_critic_local'+str(runid)+'.pth')
                
                done_flag = 1
                break
        print('end of run : ',runid)
        end_time=time.time()
        
        # Return scores for this run, the number of episodes required and the total time taken
        return scores, avg_score, episode_num, end_time-start_time, done_flag
    
    def run_experiment(self):
        print('Running experiment')
        # Initial list to store list of scores from each run
        all_scores = []
        all_num_episodes = []
        all_total_times = []
        all_avg_scores = []
        all_done_flags = []
        
        # For each run
        nsuccess = 0
        for current_runid in range(self.nruns):
            print('current experiment number = ', current_runid)
            
            # Perform the experiment for one run and return the scores, number of episodes and total time required
            scores, avg_score, num_episodes, total_time, done_flag = self.execute_one_run(current_runid)
            
            # store the scores, num_episodes and total_time
            all_scores.append(scores)
            all_num_episodes.append(num_episodes)
            all_total_times.append(total_time)
            all_avg_scores.append(avg_score)
            all_done_flags.append(done_flag)
            if done_flag==1:
                nsuccess += 1
                
            # Reset the Actor and Critic network weights, the noise, start time, clear memory
            self.agent.reset()
            
        # Find the average number of episodes required to reach the target score
        if nsuccess>0:
            avg_number_of_episodes = np.mean(np.ma.masked_array(all_num_episodes,1-np.array(all_done_flags)))
            std_number_of_episodes = np.std(np.ma.masked_array(all_num_episodes,1-np.array(all_done_flags)))
            print(' Number of experiments that were successful = ',nsuccess,'/',self.nruns)
            print('\nAverage number of episodes required to reach target score : {:2f} +/- {:2f}'.format(avg_number_of_episodes, std_number_of_episodes))
         
            # Find the average number time required to reach the target score
            #print('list of time taken = ', all_total_times)
            avg_time = np.mean(np.ma.masked_array(all_total_times, 1-np.array(all_done_flags)))
            std_time = np.std(np.ma.masked_array(all_total_times, 1-np.array(all_done_flags)))
            print('Average time/seconds per run required to reach target score : {:2f} +/- {:2f}'.format(avg_time, std_time))
        else:
            print(' Target score was not reached for any of the experiments performed')

        # Return all scores
        return all_scores, all_avg_scores, nsuccess
    

