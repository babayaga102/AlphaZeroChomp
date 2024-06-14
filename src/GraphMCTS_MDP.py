"""
This code is the standard way of building a classical MCTS on a Graph Tree. It's not
used for AlphaZeroChomp directly. It just helped me to understand and develop the project on the top of it.
"""



from collections import deque
import random
import numpy as np
from math import sqrt, log
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from Alpha_Chomp_Env import AlphaChompEnv
from Resnet import ResNet
import time
from functools import lru_cache, cache



class NodeData:
    def __init__(self, action, action_prob):
        self.action = action
        self.action_prob = action_prob


class Node:
    def __init__(self, game, model, args, state, valid_moves, parent_node=None, action_taken=None):
        self.game = game
        self.model = model
        self.state = state
        self.args = args
        self.valid_moves = valid_moves
        self.expandable_moves = valid_moves.copy()
        self.parents = {}  # Parent_node : NodeData(parent_action, parent_action_probability)
        self.children = {}  # Child_node : NodeData(action, action_prob)
        self.visit_count = 1 if self.args['MCTS_add_dirich_noise_par'] > 0 else 0
        self.score = 0.0
        self.action_taken = action_taken
        self.prior = 0.0
        self.best_child = None
        self.best_ucb = -np.inf
        torch.manual_seed(self.args['seed'])
        np.random.seed(self.args['seed'])
        random.seed(self.args['seed'])

        if parent_node is not None:
            self.add_parent(parent_node, action_taken, 0.0)

    def add_parent(self, parent_node, action, action_prob):
        '''Add parent to a node'''
        if parent_node not in self.parents:
            self.parents[parent_node] = NodeData(action, action_prob)

    def is_ending_node(self, node_state=None):
        if node_state is not None:
            return torch.equal(torch.zeros(self.args['max_size'], self.args['max_size']), node_state)
        else:
            return torch.equal(torch.zeros(self.args['max_size'], self.args['max_size']), self.state)

    def add_child(self, child_node, action, action_prob):
        '''Add parent and children to a node'''
        if child_node not in self.children:
            self.children[child_node] = NodeData(action, action_prob)
        child_node.add_parent(self, action, action_prob)

    def update(self, reward):
        self.visit_count += 1
        self.score += reward

    def is_fully_expanded(self):
        return torch.equal(torch.zeros(self.args['max_size'], self.args['max_size']), self.state) or self is None or len(self.expandable_moves) == 0

    def select(self, node_storage):
        #self.best_child_decay()

        if self.children:
            for child_node, data in self.children.items():
                ucb = self.get_ucb(child_node)
                if ucb > self.best_ucb:
                    self.best_child = child_node
                    self.best_ucb = ucb
                    self.action_taken = data.action

            return self.best_child, self.action_taken
        elif not self.children and self.is_fully_expanded():
            self.action_taken = None
            return self, self.action_taken
        elif not self.children and not self.is_fully_expanded():
            self.action_taken = None
            return self, self.action_taken
        else:
            raise ValueError("Select function received unexpected crash")

    def get_ucb(self, child):
        if child.visit_count > 0:
            q_value = 1 - (((child.score / child.visit_count) + 1) / 2)
        else:
            q_value = 0.
        ucb = q_value + self.args['C'] * np.sqrt((np.log(self.visit_count)*child.score) / (child.visit_count+ 1))
        return ucb

    def expand(self, node_storage):
        node_key = self.get_state_key(self.state)
        if node_key not in node_storage:
            node_storage[node_key] = self

        if not self.is_fully_expanded():
            action = random.choice(self.expandable_moves)
            self.expandable_moves.remove(action)

            child_state, child_valid_moves, looser, _ = self.game.get_next_state(self.state.clone(), action)
            child_key = self.get_state_key(child_state)

            if child_key in node_storage:
                child = node_storage[child_key]
                self.add_child(child, action, 0.0)
            else:
                child = Node(self.game, self.model, self.args, child_state, child_valid_moves, self, action)
                self.add_child(child, action, 0.0)
                node_storage[child_key] = child

            return child, node_storage
        else:
            return self, node_storage

    def simulate(self):
        looser, reward = self.game.get_reward_and_looser(self.state, self.action_taken)
        self.score += reward
        if looser is True:
            return reward

        rollout_state = self.state.clone()
        rollout_valid_moves = self.game.get_valid_moves(rollout_state)
        rollout_player = 1
        turn = 0

        while True:
            turn += 1
            action = random.choice(rollout_valid_moves)
            rollout_state, rollout_valid_moves, looser, reward = self.game.get_next_state(rollout_state, action)
            if looser is True:
                if turn % 2 != 0:
                    reward = -reward
                return reward

    def backpropagate(self, reward, parent_path):
        only_path = self.args['MCTS_only_path_backpropagation']
        self.update(reward)
        reward = -reward
        prov_parent_path = parent_path.copy()
        prov_parent_path = prov_parent_path[:-1]    #exclude the last element to avoid backpropagate 2 times
        #print(f"\nparent_path:{prov_parent_path}")

        if only_path:
            if len(prov_parent_path) > 0:
                for parent, action in reversed(prov_parent_path):
                    parent.update(reward)
                    reward = -reward
            else:
                pass
        else:
            for parent in self.parents:
                parent.backpropagate(reward,prov_parent_path)


    def __str__(self):
        return f"\nState: {self.state}, \nVisits: {self.visit_count}, Value: {self.score}, how many children: {len(self.children)}" \
               f"\nHow many Parents: {len(self.parents)}, How many Exp_moves left: {len(self.expandable_moves)}"

    def get_state_key(self, state):
        return tuple(map(tuple, state.numpy()))

    def best_child_decay(self):
        if self.visit_count % 10 == 0:
            self.best_child = None
            self.best_ucb = - np.inf





class GraphMCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.node_storage = {}  # State : Node
        self.parent_path_mcts = []  #Nodes we have walked trought
        self.batch_norm = nn.BatchNorm2d(1)  # BatchNorm2d for a single channel

    @torch.no_grad()
    def search(self, state, playable_cells, start_node=None):
        start_time = time.time()

        if start_node is None:
            state_key = self.get_state_key(state)
            if state_key in self.node_storage:
                starting_node = self.node_storage[state_key]
            else:
                starting_node = Node(self.game, self.model, self.args, state, playable_cells)
                self.node_storage[state_key] = starting_node
        else:
            starting_node = start_node

        node = starting_node

        for _ in tqdm(range(self.args['MCTS_num_searches']), desc="searches"):
            self.append_parent_path((node, 0.))
            while (not node.is_ending_node() and  len(node.children) > 0 ):
                node, action_taken = node.select(self.node_storage)
                self.append_parent_path((node, action_taken))

            looser, reward = self.game.get_reward_and_looser(node.state, node.action_taken)

            if looser is False:
                while not node.is_fully_expanded():
                    exp_node, exp_node_storage_updated = node.expand(self.node_storage)
                    self.node_storage.update(exp_node_storage_updated)

                reward = node.simulate()
            node.backpropagate(reward, self.parent_path_mcts)  # Change only_path to False if you want to backpropagate through all parents
            if looser is True:
                node = starting_node
            # Reset the parent_path after backpropagation
            #print(f"\nParent_Path: {self.parent_path_mcts}")
            self.empty_parent_path()

            #if _ == 0:
            #   break

        # Assuming action_probs is a numpy array
        action_probs = torch.zeros((self.args['max_size'], self.args['max_size']))

        for child, data in starting_node.children.items():
            action = list(data.action)
            if self.args['MCTS_only_path_backpropagation'] is True:
                action_probs[action[0], action[1]] = ((child.score) / (child.visit_count + 1))
            else:
                action_probs[action[0], action[1]] = (self.args['MCTS_num_searches']**2)*((child.score) / (child.visit_count + 1))

        action_probs_value = action_probs.clone()
        sum_abs_action_probs_value = torch.sum(torch.abs(action_probs_value))
        action_probs_value = (action_probs_value/sum_abs_action_probs_value)

         # Apply BatchNorm2d to action_probs
        action_probs = action_probs.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        action_probs = self.batch_norm(action_probs)
        action_probs = action_probs.squeeze(0).squeeze(0)  # Remove batch and channel dimensions


        # Flatten, apply softmax, and reshape back to original dimensions
        action_probs = self.masking_action_probs(action_probs, starting_node)
        action_probs = action_probs.view(-1)
        action_probs = F.softmax(action_probs, dim=0)
        action_probs = action_probs.view((self.args['max_size'], self.args['max_size']))

        end_time = time.time()
        search_time = end_time - start_time

        if self.args['verbose_mcts']:
            print(f"\nSearch time: {search_time:.2f} seconds for num_searches: {self.args['MCTS_num_searches']} of max_size: {(self.args['max_size'], self.args['max_size'])} \
            with OnlyPathBackpropagation: {self.args['MCTS_only_path_backpropagation']}")
            print(f"\nAction_Probs:\n{action_probs}\n\nAction_Probs_Value:\n{action_probs_value}")
            print(f"\nMost probable move:{self.num_to_move(torch.argmax(action_probs))}, of probability:{torch.max(action_probs):.5f}")
            print(f"\nLeast probable move:{self.num_to_move(torch.argmin(action_probs))}, of probability:{torch.min(action_probs):.5f}")
            print(f"\nMost Valuable move:{self.num_to_move(torch.argmax(action_probs_value))}, of Value:{torch.max(action_probs_value):.5f}")
            print(f"\nLeast Valuable move:{self.num_to_move(torch.argmin(action_probs_value))}, of Value:{torch.min(action_probs_value):.5f}")
            print(f"\nBest_UCB: {starting_node.best_ucb:.5f} Best_child_action from starting Node: {starting_node.children[starting_node.best_child].action}")
            print(f"\nBest_child state of Starting_Node:\n{starting_node.best_child.state}")
            print(f"\nStarting_Node:\n{starting_node}")
            print(f"\nLenght of Node Storage: {len(self.node_storage)}")

        return action_probs

    def masking_action_probs(self, action_probs, starting_node):
        action_probs[starting_node.state == 0.] = float('-inf')
        return action_probs

    def append_parent_path(self, node):
        if node not in self.parent_path_mcts:
            self.parent_path_mcts.append(node)

    def empty_parent_path(self):
        self.parent_path_mcts = []

    @staticmethod
    def get_state_key(state):
        return tuple(map(tuple, state.numpy()))

    @staticmethod
    def build_graph(node, graph=None):
        if graph is None:
            graph = nx.DiGraph()
        if node not in graph:
            graph.add_node(node, label=f"{node.state}\nVisits: {node.visit_count}\nValue: {node.score:.2f}\n N_Children: {len(node.children)}")
        for child, data in node.children.items():
            if child not in graph:
                graph.add_node(child, label=f"{child.state}\nVisits: {child.visit_count}\nValue: {child.score:.2f} N_Children: {len(child.children)}")
            graph.add_edge(node, child, action=data.action, action_prob = data.action_prob)
            GraphMCTS.build_graph(child, graph)
        return graph

    @staticmethod
    def visualize_mcts(root):
        G = GraphMCTS.build_graph(root)
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        edge_labels = nx.get_edge_attributes(G, 'action', 'action_prob')
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, labels=labels, with_labels=True, node_color='skyblue', node_size=3000, font_size=8, font_weight='bold', edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("MCTS Visualization")
        plt.show()


    def find_node_by_state(self, target_state):
        state_key = self.get_state_key(target_state)
        return self.node_storage.get(state_key, None)


    def num_to_move(self,action_number):
        action_number = np.array(action_number)
        row = action_number // self.args['max_size']
        column = action_number % self.args['max_size']
        return (row,column)

# Example usage
if __name__ == "__main__":
    # Define your game class and args
    args = {
        'min_size': 2,
        'max_size': 3,      #The maximum size of the game grid or game board
        'save_data_env': False,
        'model_device': 'cpu',  #Device requested for the model
        'device': 'cpu',    #Device for other operations, tipically 'cpu'
        'verbose_A_game': False,    #Bool: if AlphaChompEnv is verbose
        'verbose_mcts': True,     #Bool: if GraphMCTS is verbose
        'verbose_resnet': True,     #Bool: if Resnet is verbose
        'verbose_Alphazero' : False,    #Bool: if Alphazero is verbose
        'mix_up': False,
        'C': 2,     #UCB coefficient
        'MCTS_num_searches': 1000,    #How many GraphMCTS searchs to choose 1 move to be played in SelfPlay
        'learn_iterations': 100,     #How many Selplay and Training
        'selfPlay_iterations' : 256,     #Number of games that if will play against itself
        'epochs': 8,       #Training epochs
        'shuffle_replaybuffer': True,   # Bool: to shuffle or not the training Data befor training
        'num_hidden' : 16,      #Number of hidden convolution layers
        'num_resBlocks': 16,        #Number of resonants blocks in Resnet
        'weight_decay' : 0.0001,    #Weight_decay for loss normalization
        'temperature' : 1.5,    #from 0 to +inf. Increase exploration in selfPlay
        'MCTS_add_dirich_noise_par': 1, #if 0 no dirch_noise, if 1 dirch_noise only to root_node = (torch.ones(max_size, max_size)), if >1  dirch_noise to every node
        'dirichlet_epsilon': 0.1,   #epsilon for dirchlet noise
        'dirichlet_alpha': 0.1,     #alpha for dirchlet noise
        'seed' : 42,    #Random seed for reproducibility
        'MCTS_only_path_backpropagation' : True,     #If is desidered to backpropagate only trough path or trough eveery node
        'MCTS_best_child_decay': True,   #If you want to reset the best_child and best_ucb every treshold visits of a node
        'MCTS_updating_children_prior': False,    #If True updates the action_prob of children of the starting_node which than updates the UCB formula
        'MCTS_batch_norm': False,    #If you want or not Batch_Normalization in GraphMCTS to predicts action_probs
        'MCTS_progress_disabled': True,
        'dis_Splay_progress': True,
        'dis_Train_progress': True,
        'batch_size': 64,   #Batch_size dimension. It's also implemented dynamic batching
        'lr' : 0.001,    #Learning rate of the optimizer
        'normal_move_reward': 0.01    #The reward of a legal move apart from winning\loosing move
    }

    # Instantiate the game and MCTS
    resnet = ResNet(args)
    game = AlphaChompEnv(args)
    mcts = GraphMCTS(game, model=resnet, args=args)
    initial_state, valid_moves, looser = game.get_given_state((args['max_size'],args['max_size']))

    # Run MCTS search from the initial state
    print("Running MCTS search from the initial state...")
    action_probs = mcts.search(initial_state, valid_moves)

    root_node = mcts.find_node_by_state(initial_state)
    mcts.visualize_mcts(root_node)
    action_probs = mcts.search(initial_state, valid_moves)

    #mcts.visualize_mcts(root_node)
    action_probs = mcts.search(initial_state, valid_moves)

    #mcts.visualize_mcts(root_node)
    action_probs = mcts.search(initial_state, valid_moves)

    #mcts.visualize_mcts(root_node)
    action_probs = mcts.search(initial_state, valid_moves)

    #mcts.visualize_mcts(root_node)
    print("MCTS search completed.")


    # Starting MCTS search from a smaller state
    initial_state, valid_moves, looser = game.get_given_state((args['max_size'],args['max_size']))
    small_state, valid_moves_small, _, __ = game.get_next_state(initial_state, (1,1))
    small_node = mcts.find_node_by_state(small_state)
    action_prob2 = mcts.search(small_state,valid_moves_small)
    action_probs = mcts.search(initial_state, valid_moves, root_node)
    action_probs = mcts.search(initial_state, valid_moves)
    action_prob2 = mcts.search(small_state,valid_moves_small)
    action_probs = mcts.search(initial_state, valid_moves)
    
    mcts.visualize_mcts(root_node)
    action_prob2 = mcts.search(small_state,valid_moves_small)
    mcts.visualize_mcts(root_node)
    action_prob2 = mcts.search(small_state,valid_moves_small)
    mcts.visualize_mcts(root_node)
    action_prob2 = mcts.search(small_state,valid_moves_small)
    mcts.visualize_mcts(root_node)
    exit()

    for child_node, action in root_node.children.items():
        mcts.search(child_node.state, child_node.valid_moves, child_node)
    mcts.visualize_mcts(root_node)

   # Retrieve the key for the zero state
    zero_state_key = mcts.get_state_key(torch.zeros((args['max_size'], args['max_size'])))
    node_storage_out = mcts.node_storage.copy()
    # Remove the entry from the node storage dictionary
    if zero_state_key in mcts.node_storage:
        del node_storage_out[zero_state_key]


    # Iterate over node storage items

    for state, node in tqdm(node_storage_out.items()):
        if not node.is_fully_expanded():
            mcts.search(node.state, node.valid_moves, node)
            #mcts.visualize_mcts(root_node)
            print(f"lenght of last state:{len(node_storage_out)}")

    for state, node in tqdm(node_storage_out.items()):
        if len(node.children) == 0:
            mcts.search(node.state, node.valid_moves, node)
            #mcts.visualize_mcts(root_node)
    f_state = torch.zeros((args['max_size'], args['max_size']))
    f_state[0,0] = 1.
    f_state[1,0] = 1.
    f_state[0,1] = 1.
    f_node = mcts.find_node_by_state(f_state)
    mcts.search(f_node.state, f_node.valid_moves, f_node)
    #mcts.visualize_mcts(root_node)


    print(f"\n################            HELLO           ###########################\n")
    #print(f"\nSmall node: {mcts.node_storage[small_key]}\n")
    mcts.visualize_mcts(root_node)


'''
    if small_node:
        print("Small state node found, running MCTS search from the smaller state...")
        action_probs_from_small = mcts.search(small_state, [(0, 1), (0, 2), (1, 0), (1, 1), (2, 0)], start_node=small_node)
        print(action_probs_from_small)
    else:
        print("Small state node not found in the tree.")
'''
