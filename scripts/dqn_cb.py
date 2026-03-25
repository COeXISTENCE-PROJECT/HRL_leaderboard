from __future__ import annotations

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


import argparse
import ast
import json
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from collections     import deque
from routerl         import TrafficEnvironment
from tqdm            import tqdm

from baseline_models import BaseLearningModel
from utils           import clear_SUMO_files
from utils           import print_agent_counts

from routerl import Keychain as kc





####################################   OBSERVATION AND Q-NETWORK IMPLEMENTATION   #####################################################


###############################
# Global observation
################################
class GlobalObservation:
    """
    Store and manage global observation state from the AV fleet perspective.

    This class stores and manages global observation table (as a pd.DataFrame),
    that tracks the state of all AV agents in the environment.
    Table fields include: agent start time, origin, destination, chosen route, travel time, completion status.


    Key responsibilities:
        - Maintain agents’ data in a pd.DataFrame of shape (num_agents x num_features).
        - Keep track on the currently departing agent.
        - Update the global observation table every environment timestep based on
          information from the TrafficEnvironment.
        - Provide each agent with a view of the global state from the perspective
          of their start time.
    """


    ##############################
    ### Initialization & reset ###
    def __init__(self, agents: list[BaseAgent])->None:

        # Columns in state table & default values for representing empty data
        self.features = {
            'start_time': -1,
            'origin': -1,
            'destination': -1,
            'route': -1,
            'travel_time': -1.0,
            'has_finished': 0,
            'is_known': 0,
            #'is_acting': 0, #<- kept in self.acting_agent_id and appended only when generating agent's observation
        }

        self.state_table = self._initialize_state_table(agents)
        self.acting_agent_id = None

        self.episode_finished = False

    def reset(self)->None:
        """
        Reset the global observation:
            - Fill state table (pd.DataFrame of shape=(n_machine_agents,n_features)) with default values for each feature column.
            - Set self.episode_finished flag to False.
            - Set currently acting agent to None.

        Args:
            None
        Returns:
            None
        """
        # Fill columns with default empty values for columns
        self.state_table[:] = pd.DataFrame(self.features, index=self.state_table.index) # note: add permuting agents with the same start times here or somwhere else (e.g. compare _initialize_state_table)
        self.acting_agent_id = None
        self.episode_finished = False 
        return

    def _initialize_state_table(self, agents: list[BaseAgent] )->None:
        """
        Initialize state table.
        Set row indices to agent IDs (integers) sorted by start times. Set column names as in self.features.
        Fill with default values for each column.

        Args:
            agents (list[BaseAgent]): list of agents to be included in the global state table.

        Returns:
            pd.DataFrame
        """

        idx_time_sorted = [agent.id for agent in sorted(agents, key=lambda x: x.start_time)] # note: some permutations on agents with equal start times can be added

        empty_columns = {
            key: [value] * len(idx_time_sorted)
            for key, value in self.features.items()
        }

        return pd.DataFrame(empty_columns, index=idx_time_sorted)


    ##################################
    ### Registering starting agent ###

    def register_starting_agent(self, agent: BaseAgent)->None:
        """
        Add start time, origin and destination to agent row.
        Move acting agent indicator to this agent.

        Args:
            agent (BaseAgent): An agent whose info is to be added.
        Returns:
            None
        """

        assert agent.id in self.state_table.index
        assert self.state_table.at[agent.id, 'is_known'] == 0, "Trying to overwrite information for known (already registered) agent"

        # Check that agents are sorted by travel time and added in this order (-> start time in prev row is defined and not greater than current)
        idx_iloc = self.state_table.index.get_loc(agent.id)
        col = 'start_time'
        assert (idx_iloc == 0) or (self.state_table.iloc[idx_iloc-1][col] != self.features[col] and self.state_table.iloc[idx_iloc-1][col] <= agent.start_time)
        

        # Register agent
        self.state_table.at[agent.id, 'is_known'] = 1
        self.state_table.loc[agent.id, ['start_time', 'origin', 'destination']] = [agent.start_time, agent.origin, agent.destination]
        self.acting_agent_id = agent.id
        return

    def register_starting_agent_action(self, agent_id: int, action: int)->None:
        """
        Add action value (route identifier) to 'route' column for currently acting agent.
        """
        assert agent_id == self.acting_agent_id
        self.state_table.at[agent_id, 'route'] = action
        return


    ################################
    ### Updates from environment ###

    def update_state_with_recently_finished_machines(self, env: TrafficEnvironment)->None:
        """
        Update the global observation table with changes since the last environment snapshot.

        Specifically:
            - register travel times for agents that have finished their trips,
            - set the 'has_finished' indicators for those agents.
        """ 

        # Get travel times for recently finished agents
        active_agents = self.get_active_agents() # <- IDs of agents that were active in the last snapshot of the observation
        finished_agents_times = { # <- Agents that finished after last snapshot - according to update from TrafficEnvironment
            info[kc.AGENT_ID] : info[kc.TRAVEL_TIME]
            for info in env.travel_times_list
            if
                info[kc.AGENT_ID] in active_agents and
                kc.TRAVEL_TIME in info and 
                info[kc.TRAVEL_TIME] != self.features['travel_time'] # check if assigned travel time is not 'empty value'
        } 

        # Update state table
        self.state_table['travel_time'].update(pd.Series(finished_agents_times))
        self.state_table['has_finished'].update(pd.Series({agent: 1 for agent in finished_agents_times}))

        if self.state_table['has_finished'].all():
            self.episode_finished = True
        return 


    #######################################
    ### Agent view of global obsevation ###

    def generate_agent_observation(self, agent_id: int)->np.ndarray:
        """
        Return the agent's view of the global observation table.

        Constructs a view of the global observation table for the specified agent.
        Adds a one-hot column indicating agent as currently acting.

        Args:
            agent_id (int): identifier of the agent for whom the observation is generated (agent.id).
        Returns:
            np.ndarray: a NumPy array representing the agent’s view of the global observation, preserving original column order
            with an additional one-hot column containing a 1 in the row corresponding to the acting agent.
        """

 
        assert agent_id in self.state_table.index
        assert agent_id == self.acting_agent_id

        #######################################################################################################################################################################
        # Potential more sanity checks to perform (when ensuring corectness after changes; swithed off for efficiency)
        #   - check if agent travel_times are sorted: assert self._is_column_nondescending(colname='start_time')
        #   - optionally: check if all rows below current agent are filled with empty vals (assumed in current version; scheduled future agents may be added in next versions)
        #######################################################################################################################################################################


        # Get agent's view of state table
        obs = self.state_table.copy()
        obs['is_acting'] = 0
        obs.at[agent_id, 'is_acting'] = 1

        return obs.to_numpy()

    def get_flattened_agent_observation(self, agent_id: int)->np.ndarray:
        return self.generate_agent_observation(agent_id).flatten()


    #################################
    ####### Auxiliary methods #######

    ### Accessing state table info ###
    @property
    def num_table_columns(self)->int:
        return len(self.features) + 1 # features + 'is_acting' column
    
    def get_active_agents(self):
        """
        Get IDs of agents that are marked as 'started' but not 'finished' in the observation table.
        """
        df = self.state_table
        return df.index[df['is_known'] & ~df['has_finished']]

    def get_agent_feature(self, agent_id: int, feature: str)->Any:
        return self.state_table.at[agent_id, feature]


    ### Checking correctness ###
    def _is_column_nondescending(self, colname: str)->bool:
        """
        Check if state table column is sorted in non-descending order.
        Ignore suffix filled with default empty values.
        """

        # Get column and default value; raises error if column not present
        col = self.state_table[colname]
        default_val = self.features[colname]

        if len(col) <= 1:
            return True


        # Verify if empty and nonempty values are not mixed; get non-empty prefix
        isempty_mask = col.eq(default_val)
        if isempty_mask.any():

            first_empty = isempty_mask.idxmax()  # index of first True ( True is argmax in T/F boolean series) occurence

            # Check that all values after first empty are also empty
            clean_suffix = isempty_mask.loc[first_empty:].all()
            if not clean_suffix:
                return False
            
            # Get prefix
            prefix = col.iloc[:first_empty]

        else:
            prefix = col

        is_nondesc = (prefix.diff().iloc[1:] >= 0).all()  # drop NaN for first row
        return is_nondesc


        



###############################
# DQN Network
################################

### Simplified single-DQN implementation for single-step decision-making
class DQN(BaseLearningModel):
    """
    DQN structure:
        - predicting network
        - replay buffer
        - 
    """
    def __init__(self,
                state_size,
                action_space_size,
                device="cpu",
                eps_init=0.99,
                eps_decay=0.998,
                buffer_size=256, 
                batch_size=16, 
                lr=0.003, 
                num_epochs=1, 
                num_hidden=2, 
                widths=[32, 64, 32]):

        super().__init__()
        self.device = device

        # Q-network
        self.q_network = Network(state_size, action_space_size, num_hidden, widths).to(self.device)
        self.action_space_size = action_space_size

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Behavior policy
        self.epsilon = eps_init
        self.epsilon_decay = eps_decay

        # Training 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.loss = list()

    def act(self, state: np.ndarray)->int:
        """
        Act epsilon-greedy.
        """
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        return action
    
    def push(self, state, action, reward):
        """
        Add (s,a,r) tuple to the buffer.
        """
        self.memory.append((state, action, reward)) # All interactions are single-step, so we only store the last state, action, and reward
        return

    def learn(self):
        """
        Update network parameters.
        """

        # Skip learning if not enough data in the buffer to form a batch
        if len(self.memory) < self.batch_size:
            return

        step_loss = list()
        for _ in range(self.num_epochs):

            # Get batch of states, actions and rewards
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards = zip(*batch)
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device) ## TODO: check how actions are encoded in buffer (int or one-hot)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

            # Predict Q-values (travel times) for actions, compare with recorded travel times
            current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
            target_q_values = rewards_tensor

            # Backpropagate & optimize
            loss = self.loss_fn(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            step_loss.append(loss.item())

        self.loss.append(sum(step_loss)/len(step_loss))
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay


class Network(nn.Module):
    def __init__(self, in_size, out_size, num_hidden, widths):
        super(Network, self).__init__()
        assert len(widths) == (num_hidden + 1), "DQN widths and number of layers mismatch!"
        
        self.input_layer = nn.Linear(in_size, widths[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(widths[x], widths[x+1]) for x in range(num_hidden)])
        self.out_layer = nn.Linear(widths[-1], out_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.out_layer(x)
        return x

##################################################################################################################################






# Main script to run the centralized DQN experiment
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    args = parser.parse_args()

    ALGORITHM = "dqn_cb"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Device is: ", device)
        
    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], env_params, task_params

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    
    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    
    # Copy agents.csv from custom_network_folder to records_folder
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["script"] = os.path.abspath(__file__)
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    
    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters" : {
                "model" : human_model
            },
            "machine_parameters" : {
                "behavior" : av_behavior,
                "observation_type" : "previous_agents_plus_start_time"
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo"
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "visualize_paths" : False
        } 
    )

    env.start()
    env.reset()
    print_agent_counts(env)


    ### Human learning phase ###
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()


    # Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    print_agent_counts(env)
    
    global_observation = GlobalObservation(env.machine_agents)

    # Define Q-network
    q_net = DQN(
            state_size = len(env.machine_agents) * global_observation.num_table_columns, ##
            action_space_size = env.environment_params[kc.ACTION_SPACE_SIZE], ##
            device=device,
            eps_init=eps_init,
            eps_decay=eps_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            lr=lr, 
            num_epochs=num_epochs,
            num_hidden=num_hidden,
            widths=widths)
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}
    
    
    ### Learning phase ###
    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)


    train_every_counter = 0 # Counter for training Q-Net every k agents 

    for episode in range(training_eps):
        env.reset()
        global_observation.reset()
        temp_memory = {agent.id: dict() for agent in env.machine_agents} # Keep buffer data (observation, action) before reward is known (termination/truncation iteration)


        # Simulate trafic day by day, collect data and keep training the network
        for agent_id in env.agent_iter():
            
            _, reward, termination, truncation, info = env.last() # observation, reward, termination, truncation, info

            assert isinstance(agent_id, str) and agent_id.isnumeric()
            agent_id_int = int(agent_id)
            agent_obj = agent_lookup[agent_id]



            if termination or truncation: # Episode finished: add (s,a,r) tuples to replay buffer
                train_every_counter += 1

                # Add agent (s,a,r) to the replay buffer
                assert isinstance(reward, (np.floating, float)) and reward<0, f"Reward: {reward} ({type(reward)}); agent: {agent_id}"
                state, action = temp_memory[agent_id_int]['observation'], temp_memory[agent_id_int]['action']
                q_net.push(state, action, reward)

                action = None

                # Learn (optionally may be changed from each-k-agents to each-k-episodes)
                if train_every_counter % update_every_k_agents == 0:
                    train_every_counter = 0 # reset counter
                    q_net.learn()


            
            else: # Episode in progress - manage starting & finishing agents

                # Update global observation - with env state change
                global_observation.update_state_with_recently_finished_machines(env)
                global_observation.register_starting_agent(agent_obj)

                # Get agent observation from global observation
                agent_observation_vect = global_observation.get_flattened_agent_observation(agent_id_int)

                # Select agent action
                action = q_net.act(agent_observation_vect)
                temp_memory[agent_id_int].update({'observation': agent_observation_vect, 'action': action}) # save to add to the buffer when reward is known
                global_observation.register_starting_agent_action(agent_id_int, action)


                

            env.step(action)
            """Note: travel times for agents finished after last agent departure will not be included in global observation - 
            because: TrafficEnvironment.step(action) first adds them to TrafficEnvironment.travel_times_list, then resets this list to [] in one call,
            so it is impossible to get these last part of travel times in this arrangement.
            But, the other fact is that we do not necessarly need them in global observation - for the last agent departing,
            he does know these times at the moment his start timepoint anyway. So they can be left as unknown in global obs when the episode ends."""

        # Sanity check
        global_observation._is_column_nondescending(colname='start_time')


        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
    
    
    ### Testing phase ###
    q_net.epsilon = 0.0
    q_net.q_network.eval()

    pbar.set_description("Testing")
    global_observation = GlobalObservation(env.machine_agents)

    for episode in range(test_eps):
        env.reset()
        global_observation.reset()

        for agent_id in env.agent_iter():
            agent_id_int, agent_obj = int(agent_id), agent_lookup[agent_id]

            _, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None

            else:
                # Update global observation
                global_observation.update_state_with_recently_finished_machines(env)
                global_observation.register_starting_agent(agent_obj)

                # Get agent observation from global observation
                agent_observation_vect = global_observation.get_flattened_agent_observation(agent_id_int)

                action = q_net.act(agent_observation_vect)
                global_observation.register_starting_agent_action(agent_id_int, action)

            env.step(action)
        pbar.update()

    
    # Finalize the experiment
    pbar.close()
    env.plot_results()
    losses_df = pd.DataFrame({"losses": q_net.loss})
    losses_df.to_csv(os.path.join(records_folder, "losses.csv"))
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)