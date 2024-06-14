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
import hashlib

# The NodeData class represents a node with an action and its corresponding probability.
class NodeData:
    def __init__(self, action, action_prob):
        self.action = action
        self.action_prob = action_prob


class Node:
    def __init__(self, A_game, args, state, valid_moves, parent_node=None, action_taken=None, action_prob = None):
        self.A_game = A_game
        self.state = state
        self.args = args
        self.valid_moves = valid_moves
        self.expandable_moves = valid_moves.copy()
        self.parents = {}  # Parent_node : NodeData(parent_action, parent_prior_probability)
        self.children = {}  # Child_node : NodeData(action, action_prob), It gets updated using update_starting_node_children_prior
        self.visit_count = 1 if self.args['MCTS_add_dirich_noise_par'] > 0 else 0
        self.score = 0.0
        self.action_taken = action_taken
        self.best_child = None
        self.best_ucb = -np.inf
        torch.manual_seed(self.args['seed'])
        np.random.seed(self.args['seed'])
        random.seed(self.args['seed'])

        if parent_node is not None:
            self.add_parent(parent_node, action_taken, action_prob)

    def add_parent(self, parent_node, action, action_prob):
        '''Add parent to a node'''
        if parent_node not in self.parents:
            self.parents[parent_node] = NodeData(action, action_prob)


    def is_ending_node(self, node_state=None):
        """
        The function `is_ending_node` checks if a given node state or the current state is equal to a tensor
        of zeros.

        :param node_state: The `node_state` parameter represents the state of a node in your code. It is a
        tensor with dimensions `max_size x max_size`. The function `is_ending_node` checks if the given
        `node_state` (or the default `self.state` if `node_state` is not
        :return: The `is_ending_node` function returns a boolean value indicating whether the input
        `node_state` (or the default `self.state` if `node_state` is not provided) is equal to a tensor of
        zeros with dimensions `self.args['max_size']` by `self.args['max_size']`.
        """
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
        '''Update a node increasing the visit_count and the score'''
        self.visit_count += 1
        self.score += reward

    def is_fully_expanded(self):
        """
        The function `is_fully_expanded` checks if a state is fully expanded based on certain conditions.
        :return: The function `is_fully_expanded` returns a boolean value. It checks if the state is fully
        expanded by comparing it with a tensor of zeros of size `max_size x max_size`, or if the state is
        `None`, or if there are no more expandable moves left.
        """
        return torch.equal(torch.zeros(self.args['max_size'], self.args['max_size']), self.state.clone()) or self is None or len(self.expandable_moves) == 0

    def select(self, node_storage):
        """
        The `select` function iterates through children nodes to find the one with the highest UCB value and
        returns it along with the corresponding action taken.

        :param node_storage: It looks like the `select` method is part of a larger class or function related
        to a Monte Carlo Tree Search (MCTS) algorithm. The method is responsible for selecting the best
        child node based on some criteria
        :return: The `select` function returns either the best child node along with the action taken, or
        the current node along with a `None` action if there are no children or if the node is not fully
        expanded.
        """
        self.best_child_decay() if self.args['MCTS_best_child_decay'] else None

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
        """
        The function `get_ucb` calculates the Upper Confidence Bound (UCB) value for a given child node in
        a Monte Carlo Tree Search algorithm, taking into account the child's visit count, score, and action
        probability.

        :param child: It looks like the code you provided is a method for calculating the Upper Confidence
        Bound (UCB) value for a child node in a Monte Carlo Tree Search algorithm. The UCB value is used to
        balance exploration and exploitation in the search process
        :return: The `get_ucb` method is returning the Upper Confidence Bound (UCB) value for a given child
        node in the context of a Monte Carlo Tree Search (MCTS) algorithm. The UCB value is calculated
        based on the child's visit count, score, and other parameters such as the exploration constant `C`.
        The formula used in the method combines these factors to determine the UCB value
        """
        if child.visit_count > 0:
            # 0 < q < 1 and switch viewpoint to the opponent
            q_value = 1 - (((child.score / child.visit_count) + 1) / 2)
        else:
            q_value = 0.

        '''Handcrafted UCB formula. I added child.score at the numerator so the MCTS will take in consideration
        the score of aech child. Otherwise the last child, (torch.zeros(max_size, max_size)) which is the leaf node
         of the GraphTree has the highest visit count cause every times you end up there and this mechanism breaks the effective search'''
        ucb = q_value+self.args['C']*((np.sqrt(self.visit_count)*child.score)/(child.visit_count+ 1)) \
            * self.children[child].action_prob
        return ucb

    def expand(self, node_storage, masked_policy):
        """
        The `expand` function in Python is used to expand a node in a tree structure based on a given
        policy, creating child nodes if necessary and updating the node storage accordingly.

        :param node_storage: The `node_storage` parameter in the `expand` method is a dictionary that
        stores nodes based on their unique keys. This dictionary is used to keep track of nodes that have
        already been visited or expanded during the search process. If a node is not already in the
        `node_storage`, it is added
        :param masked_policy: The `masked_policy` parameter in the `expand` method seems to represent a
        list of action probabilities that have been masked in some way. The method then uses these
        probabilities to determine which actions to take during the expansion phase of the Monte Carlo
        Tree Search (MCTS) algorithm
        :return: In this `expand` method, a `Node` object and the `node_storage` dictionary are being
        returned. The `Node` object represents a child node that has been expanded from the current node,
        and the `node_storage` dictionary contains information about all nodes encountered during the
        expansion process.
        """
        #Check if it's already in node_storage
        node_key = self.get_state_key(self.state.clone())
        if node_key not in node_storage:
            node_storage[node_key] = self

        if not self.is_fully_expanded():
            if self.args['MCTS_set_equal_prior'] is False:
                for action_num , action_prob in enumerate(masked_policy):
                    action = self.num_to_move(action_num)   #Number to coordinates
                    if action_prob > 0. and action in self.expandable_moves:
                        self.expandable_moves.remove(action)    #Remove if the action has already been taken
                        child_state, child_valid_moves, looser, _ = self.A_game.get_next_state(self.state.clone(), action)

                        child_key = self.get_state_key(child_state)

                        if child_key in node_storage:
                            child = node_storage[child_key]
                            self.add_child(child, action, action_prob)
                        else:
                            child = Node(self.A_game, self.args, child_state, child_valid_moves, self, action, action_prob)
                            self.add_child(child, action, action_prob)  #Child_node: (action, action_prob_prior)
                            node_storage[child_key] = child

                        return child, node_storage
            else:
                equal_prior = 1. / len(self.valid_moves)
                for action_num , action_prob in enumerate(masked_policy):
                    action = self.num_to_move(action_num)   #Number to coordinates
                    if action in self.expandable_moves:
                        self.expandable_moves.remove(action)    #Remove if the action has already been taken
                        child_state, child_valid_moves, looser, _ = self.A_game.get_next_state(self.state.clone(), action)

                        child_key = self.get_state_key(child_state)

                        if child_key in node_storage:
                            child = node_storage[child_key]
                            self.add_child(child, action, equal_prior)
                        else:
                            child = Node(self.A_game, self.args, child_state, child_valid_moves, self, action, equal_prior)
                            self.add_child(child, action, equal_prior)  #Child_node: (action, action_prob_prior)
                            node_storage[child_key] = child

                        return child, node_storage
                return

        else:
            #we are at terminal Node
            return self, node_storage

    def simulate(self):
        """
        This function simulates a game by making random moves until a player loses, updating the score
        accordingly.
        :return: The `simulate` method returns the reward value when a looser is detected during the
        simulation. If a looser is not detected, the method continues with a rollout simulation until a
        looser is found, and then returns the reward value.
        """
        looser, reward = self.A_game.get_reward_and_looser(self.state, self.action_taken)
        self.score += reward
        if looser is True:
            return reward

        rollout_state = self.state.clone()
        rollout_valid_moves = self.A_game.get_valid_moves(rollout_state)
        rollout_player = 1
        turn = 0

        while True:
            turn += 1
            action = random.choice(rollout_valid_moves)
            rollout_state, rollout_valid_moves, looser, reward = self.A_game.get_next_state(rollout_state, action)
            if looser is True:
                if turn % 2 != 0:
                    reward = -reward
                return reward

    def backpropagate(self, reward, parent_path):
        """
        The `backpropagate` function updates the rewards of nodes in a tree structure based on a given
        reward and parent path, with an option for only propagating along a specific path.

        :param reward: The `reward` parameter in the `backpropagate` method represents the reward signal
        that is being propagated back through the tree during the reinforcement learning process. It is
        used to update the values associated with each node in the tree based on the outcome of the
        simulation or game that was played
        :param parent_path: The `backpropagate` method is used to update the rewards of the nodes in the
        tree based on the received reward and the path taken to reach the current node
        """
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
        """
        The `__str__` function returns a string representation of an object with information about its
        state, visit count, score, number of children, parents, and expandable moves.
        :return: The `__str__` method is returning a formatted string that includes the state, visit
        count, score, number of children, number of parents, and number of expandable moves of the
        object.
        """
        return f"\nState: {self.state}, \nVisits: {self.visit_count}, Value: {self.score}, how many children: {len(self.children)}" \
               f"\nHow many Parents: {len(self.parents)}, How many Exp_moves left: {len(self.expandable_moves)}"

    def get_state_key(self, state):
        """
        The function `get_state_key` takes a state, converts it to a byte array, calculates its SHA-1
        hash, and returns the hexadecimal representation of the hash.

        :param state: The `state` parameter in the `get_state_key` function seems to be a numpy array
        that is being converted to bytes and then hashed using the SHA-1 algorithm to generate a
        hexadecimal digest. This function is likely used to create a unique key based on the state of the
        numpy array for some
        :return: The function `get_state_key` takes a `state` as input, converts it to a byte array using
        NumPy, calculates the SHA-1 hash of the byte array, and returns the hexadecimal representation of
        the hash.
        """
        c = np.array(state).tobytes()
        return hashlib.sha1(c).hexdigest()

    def num_to_move(self,action_number):
        action_number = np.array(action_number)
        row = action_number // self.args['max_size']
        column = action_number % self.args['max_size']
        return (row,column)


    def best_child_decay(self):
        """
        The function `best_child_decay` resets the `best_child` and `best_ucb` variables after a certain
        number of visits.
        """
        treshold = 10
        #every treshold visits reset best_child and best_ucb
        if self.visit_count % treshold == 0:
            self.best_child = None
            self.best_ucb = - np.inf



class GraphMCTS:
    def __init__(self, A_game, model, args):
        self.A_game = A_game
        self.model = model
        self.args = args
        self.node_storage = {}  # hashed_State : Node
        self.parent_path_mcts = []  #Nodes we have walked trought
        self.batch_norm = nn.BatchNorm2d(1)  # BatchNorm2d for a single channel
        self.current_actions = []
        self.opponent_actions = []
        self.turn = 0

    @torch.no_grad()
    def search(self, state, playable_cells, start_node=None):
        """
        The `search` function in Python uses Monte Carlo Tree Search (MCTS) to find the best move in a game
        state, updating node values and probabilities based on the search results.

        :param state: The `search` method you provided seems to be implementing the Monte Carlo Tree Search
        (MCTS) algorithm for decision-making in a game. The method takes several parameters and performs a
        search to find the best action to take based on the current state of the game
        :param playable_cells: The `playable_cells` parameter in the `search` method represents the cells on
        the game board where a player can make a move. The method uses this information to determine the
        possible actions or moves that can
        :param start_node: The `start_node` parameter in the `search` method is used to specify a starting
        node for the Monte Carlo Tree Search (MCTS) algorithm. If a `start_node` is provided, the search
        will begin from that node instead of starting from scratch. This can be useful in certain scenarios
        :return: The `search` method returns two tensors: `action_probs` and `action_probs_value`. These
        tensors represent the action probabilities calculated during the Monte Carlo Tree Search (MCTS)
        algorithm. The `action_probs` tensor contains the probabilities of each possible action, while the
        `action_probs_value` tensor contains the values associated with those probabilities.
        """
        start_time = time.time()

        if start_node is None:
            state_key = self.get_state_key(state)
            if state_key in self.node_storage:
                starting_node = self.find_node_by_state(state)
            else:
                starting_node = Node(self.A_game, self.args, state, playable_cells)
                self.node_storage[state_key] = starting_node
        else:
            starting_node = start_node
        node = starting_node
        #print(f"\nSearch Node: {node}")    #Debug
        #Add dirichlet noise at the strarting node
        self.add_dirichlet_noise(node)

        for search in tqdm(range(self.args['MCTS_num_searches']), desc="searches", disable=self.args['MCTS_progress_disabled']):
            self.turn = 0   #set the turn counter to zero
            self.append_parent_path((node, None))
            while (not node.is_ending_node() and len(node.children) > 0):
                #Selection
                node, action_taken = node.select(self.node_storage)
                self.append_parent_path((node, action_taken))

            looser, reward = self.A_game.get_reward_and_looser(node.state, node.action_taken)

            if looser is False:
                self.collect_actions()  #Collect players actions
                while not node.is_fully_expanded():
                    masked_policy, policy, value = self.model(node.state.clone(), self.current_actions.copy(), self.opponent_actions.copy())
                    #Expansion
                    exp_node, exp_node_storage_updated = node.expand(self.node_storage, masked_policy)
                    self.node_storage.update(exp_node_storage_updated)

            node.backpropagate(reward, self.parent_path_mcts)  # Change only_path to False if you want to backpropagate through all parents
            if looser is True:
                node = starting_node
            self.empty_parent_path()

        action_probs = torch.zeros((self.args['max_size'], self.args['max_size']))

        for child, data in starting_node.children.items():
            action = list(data.action)
            if self.args['MCTS_only_path_backpropagation'] is True:
                #Handcrafted action_probs formula = ((child.score) / (child.visit_count + 1))
                action_probs[action[0], action[1]] =  starting_node.get_ucb(child) #((child.score) / (child.visit_count + 1))
            else:
                action_probs[action[0], action[1]] = (self.args['MCTS_num_searches']**2)*((child.score) / (child.visit_count + 1))

        action_probs_value = action_probs.clone()
        sum_abs_action_probs_value = torch.sum(torch.abs(action_probs_value))
        action_probs_value = (action_probs_value/sum_abs_action_probs_value)

        # Flatten, apply softmax, and reshape back to original dimensions
        action_probs = self.masking_action_probs(action_probs_value, starting_node)
        action_probs = 10 * action_probs.flatten()
        action_probs = F.softmax(action_probs, dim=0)
        action_probs = action_probs.view((self.args['max_size'], self.args['max_size']))
        action_probs_sum = torch.sum(action_probs)

        self.update_starting_node_children_prior(starting_node, action_probs.clone()) if self.args['MCTS_updating_children_prior'] else None

        end_time = time.time()
        search_time = end_time - start_time

        if self.args['verbose_mcts'] is True:
            print(f"\nSearch time: {search_time:.2f} seconds for num_searches: {self.args['MCTS_num_searches']} of max_size: {(self.args['max_size'], self.args['max_size'])} \
                        with OnlyPathBackpropagation: {self.args['MCTS_only_path_backpropagation']}")
            print(f"\nAction_Probs:\n{action_probs}\n\nAction_Probs_Value:\n{action_probs_value}")
            print(f"\nMost probable move:{self.num_to_move(torch.argmax(action_probs))}, of probability:{torch.max(action_probs):.5f}")
            print(f"\nLeast probable move:{self.num_to_move(torch.argmin(action_probs))}, of probability:{torch.min(action_probs):.5f}")
            print(f"\nProbabilities sum: {action_probs_sum}")
            print(f"\nMost Valuable move:{self.num_to_move(torch.argmax(action_probs_value))}, of Value:{torch.max(action_probs_value):.5f}")
            print(f"\nLeast Valuable move:{self.num_to_move(torch.argmin(action_probs_value))}, of Value:{torch.min(action_probs_value):.5f}")
            print(f"\nBest_UCB: {starting_node.best_ucb:.5f} Best_child_action from starting Node: {starting_node.children[starting_node.best_child].action}")
            print(f"\nBest_child state of Starting_Node:\n{starting_node.best_child.state}")
            print(f"\nStarting_Node:\n{starting_node}")
            print(f"\nLenght of Node Storage: {len(self.node_storage)}")

        return action_probs, action_probs_value

    def masking_action_probs(self, action_probs, starting_node):
        '''Masks the action probabilities using the state of the starting_node to avoid
            that illegal action will be taken. The renormalization will be done later in the code.
            This snippet of code masks square action_prob using square_node_state'''
        action_probs[starting_node.state.clone() == 0.] = float('-inf')
        return action_probs

    def append_parent_path(self, node):
        '''Append a new node to the parent path for later backpropagation'''
        if node not in self.parent_path_mcts:
            self.parent_path_mcts.append(node)

    def empty_parent_path(self):
        '''It just empty the parent_path at the end of every backpropagation'''
        self.parent_path_mcts = []

    @staticmethod
    def get_state_key(state):
        '''Get the key of the hash table of a state'''
        c = np.array(state).tobytes()
        return hashlib.sha1(c).hexdigest()

    @staticmethod
    def build_graph(node, graph=None):
        '''Build the graph of the GraphMCTS'''
        if graph is None:
            graph = nx.DiGraph()
        if node not in graph:
            graph.add_node(node, label=f"{node.state}\nVisits: {node.visit_count}\nValue: {node.score:.2f}\n N_Children: {len(node.children)}")
        for child, data in node.children.items():
            if child not in graph:
                graph.add_node(child, label=f"{child.state}\nVisits: {child.visit_count}\nValue: {child.score:.2f} N_Children: {len(child.children)}")
            graph.add_edge(node, child, action=data.action, action_prob=data.action_prob)
            GraphMCTS.build_graph(child, graph)
        return graph

    @staticmethod
    def visualize_mcts(root):
        '''Build and visualize the graph of the GraphMCTS'''
        G = GraphMCTS.build_graph(root)
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        # Update edge_labels to include both action and action_prob
        edge_labels = {
            (u, v): f"Action: {d['action']}\nProb: {d['action_prob']:.3f}"
            for u, v, d in G.edges(data=True)}
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, labels=labels, with_labels=True, node_color='skyblue', node_size=3000, font_size=8, font_weight='bold', edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("MCTS Visualization")
        plt.show()


    def collect_actions(self):
        """Collect the actions of player 1 and player 2 from the parent_path_mcts. These action will be fed to the NN ResNet
        model as a list of tuple and the model will transcribe that into matrices and stacking togheter in a sandwitch
        with the state of the game"""
        prov_parent_path = self.parent_path_mcts.copy()
        prov_parent_path = prov_parent_path[:-1]    #The last element is in parent path 2 times, so i remove it
        self.current_actions = []
        self.opponent_actions = []
        for i, (node, action) in enumerate(prov_parent_path):
            if action is not None:
                if i % 2 == 0:
                    self.current_actions.append(action)
                else:
                    self.opponent_actions.append(action)
            elif action is None:
                pass
            else:
                raise ValueError(f"\nThe action: {action}, of Node_State: {node.state} encountered an issue!")

    def find_node_by_state(self, target_state):
        '''Find a node in node_storage the state of the node'''
        state_key = self.get_state_key(target_state)
        return self.node_storage.get(state_key, None)

    def num_to_move(self,action_number):
        '''Convert a numerical action into a coordinates action'''
        action_number = np.array(action_number)
        row = action_number // self.args['max_size']
        column = action_number % self.args['max_size']
        return (row,column)

    def add_dirichlet_noise(self, node):
        """
        The `add_dirichlet_noise` function adds Dirichlet noise to nodes based on specified conditions
        before starting a search process.

        :param node: The `add_dirichlet_noise` function is used to add Dirichlet noise to a given node
        before starting a search. The function takes into account the value of
        `self.args['MCTS_add_dirich_noise_par']` to determine how the noise should be added:
        Add the Dirichlert noise to the node before starting serach. If self.args['MCTS_add_dirich_noise_par'] > 1: it add Dirichlet Noise
        to every node. If self.args['MCTS_add_dirich_noise_par'] == 1 it will add Dirichlet Noise ONLY to the root node (biggest matrix of ones).
        If self.args['MCTS_add_dirich_noise_par'] == 0 it will add no noise to any node. We first calculate the policy, then we add the Noise
        and then we expand the node setting the policy with dirichlet noise as action_prob of the children dictionary
        """
        if self.args['MCTS_add_dirich_noise_par'] > 1:
            #We add dirichlet noise to every node

            masked_policy, _, value = self.model(node.state.clone())

            dirch_policy = (1 - self.args['dirichlet_epsilon']) * masked_policy + self.args['dirichlet_epsilon'] \
                * np.random.dirichlet([self.args['dirichlet_alpha']] * (self.args['max_size']**2))

            dirch_policy[node.state.clone().flatten() == 0.] = float('-inf')   #Masking
            dirch_policy = F.softmax(dirch_policy, dim=0)     #Renormalization
            while not node.is_fully_expanded():
                exp_node, exp_node_storage_updated = node.expand(self.node_storage.copy(), dirch_policy)
                self.node_storage.update(exp_node_storage_updated)

        elif torch.equal(node.state.clone(), torch.ones(self.args['max_size'],self.args['max_size'])) and 0 < self.args['MCTS_add_dirich_noise_par'] <= 1:
            #So that's root node and we add dirchlet noise only to root node
            masked_policy, _, value = self.model(node.state.clone())

            dirch_policy = (1 - self.args['dirichlet_epsilon']) * masked_policy + self.args['dirichlet_epsilon'] \
                * np.random.dirichlet([self.args['dirichlet_alpha']] * (self.args['max_size']**2))

            dirch_policy[node.state.clone().flatten() == 0.] = float('-inf')   #Masking
            dirch_policy = F.softmax(dirch_policy, dim=0)     #Renormalization
            while not node.is_fully_expanded():
                exp_node, exp_node_storage_updated = node.expand(self.node_storage.copy(), dirch_policy)
                self.node_storage.update(exp_node_storage_updated)


        elif self.args['MCTS_add_dirich_noise_par'] <= 0:
            #No Dirichlet Noise
            pass


    def update_starting_node_children_prior(self, starting_node, mcts_action_probs):
        '''Updates the prior probaility of the children nodes of starting_node. In this way the GraphMCTS updates the UCB
            and the GraphMCTS selection phase will select different nodes without getting stuck with the same best_child forever'''

        for child_node, data in starting_node.children.items():
            action = list(data.action)
            starting_node.children[child_node].action_prob = mcts_action_probs[action[0], action[1]].item()



'''
# testing
if __name__ == "__main__":

    args = {
        'min_size': 2,
        'max_size': 3,
        'save_data_env': False,
        'device': 'cpu',
        'model_device': 'cpu',
        'verbose_A_game': False,
        'verbose_mcts': True,
        'verbose_resnet': True,
        'mix_up': False,
        'C': 2,
        'MCTS_num_searches': 1000,
        'learn_iterations': 100,
        'selfPlay_iterations' : 100,
        'epochs': 100,
        'num_hidden' : 64,
        'num_resBlocks': 16,
        'weight_decay' : 0.0001,
        'temperature' : 1.25,    #from 0 to +inf
        'MCTS_add_dirich_noise_par': 1, #if 0 no dirch_noise, if 1 dirch_noise only to root_node, if >1  dirch_noise to every node
        'dirichlet_epsilon': 0.1,   #epsilon for dirchlet noise
        'dirichlet_alpha': 0.1,     #alpha for dirchlet noise
        'seed' : 42,
        'MCTS_updating_children_prior':True,
        'MCTS_only_path_backpropagation' : True,
        'MCTS_best_child_decay': True,
        'MCTS_progress_disabled': False,
        'MCTS_set_equal_prior':False,
        'batch_size': 32,
        'lr' : 0.001
    }

    # Instantiate the A_game and MCTS
    resnet = ResNet(args)
    resnet.eval()
    Alpha_game = AlphaChompEnv(args)
    mcts = GraphMCTS(Alpha_game, model=resnet, args=args)
    initial_state, valid_moves, looser = Alpha_game.get_given_state((args['max_size'],args['max_size']))


    # Run MCTS search from the initial state
    print("Running MCTS search from the initial state...")
    action_probs = mcts.search(initial_state, valid_moves)

    root_node = mcts.find_node_by_state(initial_state)
    mcts.visualize_mcts(root_node)
    action_probs = mcts.search(initial_state, valid_moves)
    mcts_state_key = mcts.get_state_key(initial_state)
    node_state_key = root_node.get_state_key(initial_state)
    print(f"\nmcts_state_key: {mcts_state_key}, node_state_key: {node_state_key}")
    exit()

    #mcts.visualize_mcts(root_node)
    action_probs = mcts.search(initial_state, valid_moves)

    #mcts.visualize_mcts(root_node)
    action_probs = mcts.search(initial_state, valid_moves)

    #mcts.visualize_mcts(root_node)
    action_probs = mcts.search(initial_state, valid_moves)

    #mcts.visualize_mcts(root_node)
    print("MCTS search completed.")


    # Starting MCTS search from a smaller state
    initial_state, valid_moves, looser = A_game.get_given_state((args['max_size'],args['max_size']))
    small_state, valid_moves_small, _, __ = A_game.get_next_state(initial_state, (1,1))
    small_node = mcts.find_node_by_state(small_state)
    action_prob2 = mcts.search(small_state,valid_moves_small)
    wait(5)
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



    if small_node:
        print("Small state node found, running MCTS search from the smaller state...")
        action_probs_from_small = mcts.search(small_state, [(0, 1), (0, 2), (1, 0), (1, 1), (2, 0)], start_node=small_node)
        print(action_probs_from_small)
    else:
        print("Small state node not found in the tree.")

'''
