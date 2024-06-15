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
        Check if a given node state or the current state is equal to a tensor of zeros.

        Parameters
        ----------
        node_state : torch.Tensor, optional
            The state of a node. It is a tensor with dimensions `max_size x max_size`.
            If not provided, the default is `self.state`.

        Returns
        -------
        bool
            True if the input `node_state` (or the default `self.state` if `node_state` is not provided)
            is equal to a tensor of zeros with dimensions `self.args['max_size']` by `self.args['max_size']`.
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
        Check if a state is fully expanded based on certain conditions.

        Returns
        -------
        bool
            True if the state is fully expanded, which is determined by:
            - The state being equal to a tensor of zeros of size `max_size x max_size`.
            - The state being `None`.
            - No more expandable moves left.
        """
        return torch.equal(torch.zeros(self.args['max_size'], self.args['max_size']), self.state.clone()) or self is None or len(self.expandable_moves) == 0

    def select(self, node_storage):
        """
        Iterate through children nodes to find the one with the highest UCB value and return it along with the
        corresponding action taken.

        Parameters
        ----------
        node_storage : dict
            A dictionary used to keep track of nodes that have already been visited or expanded during the
            search process.

        Returns
        -------
        tuple
            The best child node along with the action taken, or the current node along with a `None` action
            if there are no children or if the node is not fully expanded.

        Raises
        ------
        ValueError
            If the function encounters an unexpected condition.
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
        Calculate the Upper Confidence Bound (UCB) value for a given child node in a Monte Carlo Tree Search
        algorithm.

        Parameters
        ----------
        child : Node
            The child node for which the UCB value is being calculated. The node must have attributes
            `visit_count`, `score`, and `action_prob`.

        Returns
        -------
        float
            The Upper Confidence Bound (UCB) value for the given child node. The UCB value is calculated based
            on the child's visit count, score, and the exploration constant `C`. The formula used combines these
            factors to determine the UCB value.

        Notes
        -----
        The UCB value is used to balance exploration and exploitation in the search process. A handcrafted UCB
        formula is used in this method, which takes into account the child's score to avoid favoring leaf nodes
        with high visit counts.

        If the child's visit count is greater than 0, the Q value is calculated as `1 - (((child.score /
        child.visit_count) + 1) / 2)`, which switches the viewpoint to the opponent. If the visit count is 0,
        the Q value is set to 0.

        The final UCB value is computed as:

        .. math::
            \text{ucb} = q\_value + C \times \left( \frac{\sqrt{\text{self.visit_count}} \times \text{child.score}}{\text{child.visit_count} + 1} \right) \times \text{self.children[child].action_prob}
        """
        if child.visit_count > 0:
            # 0 < q < 1 and switch viewpoint to the opponent
            q_value = 1 - (((child.score / child.visit_count) + 1) / 2)
        else:
            q_value = 0.

        '''Handcrafted UCB formula. I added child.score at the numerator so the MCTS will take in consideration
        the score of aech child. Otherwise the last child, (torch.zeros(max_size, max_size)) which is the leaf node
         of the GraphTree has the highest visit count cause every times you end up there and this mechanism breaks the effective search'''
        ucb = q_value+self.args['C']*((np.sqrt(self.visit_count)*child.score/(child.visit_count+ 1))) \
            * self.children[child].action_prob
        return ucb

    def expand(self, node_storage, masked_policy):
        """
        Expand a node in a tree structure based on a given policy, creating child nodes if necessary and
        updating the node storage accordingly.

        Parameters
        ----------
        node_storage : dict
            A dictionary used to keep track of nodes that have already been visited or expanded during the
            search process. If a node is not already in the `node_storage`, it is added.
        masked_policy : list of float
            List of action probabilities that have been masked in some way. The method uses these probabilities
            to determine which actions to take during the expansion phase of the Monte Carlo Tree Search (MCTS)
            algorithm.

        Returns
        -------
        Node
            The `Node` object represents a child node that has been expanded from the current node.
        dict
            The `node_storage` dictionary contains information about all nodes encountered during the expansion
            process.

        Notes
        -----
        The function checks if the current node is fully expanded. If not, it uses the masked policy to
        determine which actions to take. If the node is fully expanded, it returns the current node and
        `node_storage`.

        If `MCTS_set_equal_prior` is `False`, the function iterates over the `masked_policy`, converting action
        numbers to coordinates, and expanding child nodes based on the action probabilities. If the action
        probability is greater than 0 and the action is in `expandable_moves`, it removes the action from
        `expandable_moves`, gets the next state, and either retrieves or creates a child node, adding it to the
        node storage.

        If `MCTS_set_equal_prior` is `True`, the function assigns equal probability to each valid move and
        performs similar steps to expand child nodes.

        If the node is fully expanded, it returns the current node and `node_storage`.
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
        Simulate a game by making random moves until a player loses, updating the score accordingly.

        Returns
        -------
        int
            The reward value when a looser is detected during the simulation. If a looser is not detected,
            the method continues with a rollout simulation until a looser is found, and then returns the
            reward value.

        Notes
        -----
        The function first checks for a looser using `self.A_game.get_reward_and_looser` with the current state
        and action. If a looser is found, the function updates the score and returns the reward.

        If no looser is found, the function enters a rollout simulation loop where it makes random moves
        until a looser is detected. The state and valid moves are updated at each step, and the turn counter
        is incremented. If a looser is found during the rollout, the reward is adjusted based on the turn
        number and returned.
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
        Update the rewards of nodes in a tree structure based on a given reward and parent path, with an option
        for only propagating along a specific path.

        Parameters
        ----------
        reward : float
            The reward signal that is being propagated back through the tree during the reinforcement learning process.
            It is used to update the values associated with each node in the tree based on the outcome of the
            simulation or game that was played.
        parent_path : list
            The path taken to reach the current node, represented as a list of parent nodes and the corresponding
            actions taken to traverse from parent to child.

        Notes
        -----
        The `backpropagate` method updates the rewards of the nodes in the tree based on the received reward and
        the path taken to reach the current node. If the `MCTS_only_path_backpropagation` option is enabled, the
        method only propagates the reward along the specific path taken to reach the current node. Otherwise, it
        propagates the reward to all parent nodes. The reward is negated at each step to alternate between
        maximizing and minimizing nodes.
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
        Return a string representation of an object with information about its state, visit count, score,
        number of children, parents, and expandable moves.

        Returns
        -------
        str
            A formatted string that includes the state, visit count, score, number of children, number of parents,
            and number of expandable moves of the object.

        Notes
        -----
        The `__str__` method provides a human-readable representation of the object's key attributes, making it
        easier to inspect the state and properties of the object during debugging or logging.
        """
        return f"\nState: {self.state}, \nVisits: {self.visit_count}, Value: {self.score}, how many children: {len(self.children)}" \
               f"\nHow many Parents: {len(self.parents)}, How many Exp_moves left: {len(self.expandable_moves)}"

    def get_state_key(self, state):
        """
        Convert a state to a unique key by hashing its byte representation.

        Parameters
        ----------
        state : numpy.ndarray
            The state to be converted to a unique key. This state is expected to be a NumPy array.

        Returns
        -------
        str
            The hexadecimal representation of the SHA-1 hash of the byte array corresponding to the input state.

        Notes
        -----
        This function is useful for creating a unique identifier for a given state, which can be used in
        various applications such as caching, lookup tables, or ensuring uniqueness of states in a search
        algorithm.

        Examples
        --------
        >>> state = np.array([[0, 1], [1, 0]])
        >>> key = obj.get_state_key(state)
        >>> print(key)
        'f7c3bc1d808e04732adf679965ccc34ca7ae3441'
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
        Reset the `best_child` and `best_ucb` variables periodically based on the visit count.

        Notes
        -----
        This function resets the `best_child` and `best_ucb` attributes after a certain number of visits,
        determined by the threshold value. This is useful in algorithms like Monte Carlo Tree Search (MCTS)
        to ensure that the search does not overly favor previously selected children.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> obj.visit_count = 10
        >>> obj.best_child_decay()
        >>> print(obj.best_child)
        None
        >>> print(obj.best_ucb)
        -inf
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
        self.current_actions = []
        self.opponent_actions = []
        self.turn = 0

    @torch.no_grad()
    def search(self, state, playable_cells, start_node=None):
        """
        Perform Monte Carlo Tree Search (MCTS) to find the best move in a given game state.

        Parameters
        ----------
        state : torch.tensor
            The current state of the game represented as a tensor.
        playable_cells : list of tuples
            A tensor indicating the cells on the game board where a player can make a move.
        start_node : Node, optional
            The starting node for the search. If provided, the search will begin from this node instead of
            starting from scratch (default is None).

        Returns
        -------
        action_probs : torch.Tensor
            A tensor containing the probabilities of each possible action.
        action_probs_value : torch.Tensor
            A tensor containing the values associated with the action probabilities.

        Notes
        -----
        This method performs MCTS to evaluate possible actions in the game and update node values and
        probabilities based on the search results. The search process includes selection, expansion,
        simulation, and backpropagation phases.

        Examples
        --------
        >>> state = torch.tensor([...])
        >>> playable_cells = torch.tensor([...])
        >>> action_probs, action_probs_value = search(state, playable_cells)
        >>> print(action_probs)
        >>> print(action_probs_value)
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
                    masked_policy, policy, value = self.model(node.state.clone().to(self.model.device), self.current_actions.copy(), self.opponent_actions.copy())
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
        """
        Masks illegal action probabilities based on the state of the starting node.

        Parameters
        ----------
        action_probs : torch.Tensor
            Action probabilities for each possible action.
        starting_node : Node
            Starting node containing state information.

        Returns
        -------
        torch.Tensor
            Masked action probabilities.
        """
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
        c = np.array(state.cpu()).tobytes()
        return hashlib.sha1(c).hexdigest()

    @staticmethod
    def build_graph(node, graph=None):
        """
        Builds the graph of the GraphMCTS.

        Parameters
        ----------
        node : Node
            The current node from which to build the graph.
        graph : networkx.DiGraph, optional
            The graph to which nodes and edges are added. If None, a new directed graph is created.

        Returns
        -------
        networkx.DiGraph
            The graph with all nodes and edges added.
        """
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
        """
        Build and visualize the graph of the GraphMCTS.

        Parameters
        ----------
        root : Node
            The root node of the GraphMCTS from which the graph is built.

        Returns
        -------
        None
            This function does not return anything. It displays a visualization of the MCTS graph.
        """
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
        """
        Collect the actions of player 1 and player 2 from the parent_path_mcts.

        These actions will be fed to the ResNet model as a list of tuples. The model will transcribe
        them into matrices and stack them together with the state of the game.

        Raises
        ------
        ValueError
            If an unexpected issue with the action or node state is encountered.
        """
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
        Add Dirichlet noise to nodes before starting a search process based on specified conditions.

        Parameters
        ----------
        node : Node
            The node to which Dirichlet noise will be added before starting a search.

        Notes
        -----
        - If `self.args['MCTS_add_dirich_noise_par'] > 1`, Dirichlet noise is added to every node.
        - If `self.args['MCTS_add_dirich_noise_par'] == 1`, Dirichlet noise is added only to the root node.
        - If `self.args['MCTS_add_dirich_noise_par'] <= 0`, no Dirichlet noise is added.

        The function first calculates the policy, then adds the noise, and then expands the node, setting the policy
        with Dirichlet noise as `action_prob` of the children dictionary. It works with 'mps' using 'MCTS_set_equal_prior' = True.
        """


        if self.args['MCTS_add_dirich_noise_par'] > 1:      #We add dirichlet noise to every node
            masked_policy, _, value = self.model(node.state.clone().to(self.model.device))
            dirichlet_noise = torch.tensor(np.random.dirichlet([self.args['dirichlet_alpha']] * (self.args['max_size']**2)), dtype=torch.float32).to(self.model.device)
            dirch_policy = (1 - self.args['dirichlet_epsilon']) * masked_policy + self.args['dirichlet_epsilon'] * dirichlet_noise

            #dirch_policy[node.state.clone().to(self.model.device).flatten() == 0.] = float('-inf')
            #dirch_policy = F.softmax(dirch_policy, dim=0)
            dirch_policy *= node.state.clone().to(self.model.device).flatten()
            dirch_policy /= torch.sum(dirch_policy)
            while not node.is_fully_expanded():
                exp_node, exp_node_storage_updated = node.expand(self.node_storage.copy(), dirch_policy)
                self.node_storage.update(exp_node_storage_updated)
        elif torch.equal(node.state.clone(), torch.ones(self.args['max_size'], self.args['max_size'])) and 0 < self.args['MCTS_add_dirich_noise_par'] <= 1:
            #So that's root node and we add dirchlet noise only to root node
            masked_policy, _, value = self.model(node.state.clone().to(self.model.device))
            dirichlet_noise = torch.tensor(np.random.dirichlet([self.args['dirichlet_alpha']] * (self.args['max_size']**2)), dtype=torch.float32).to(self.model.device)
            dirch_policy = (1 - self.args['dirichlet_epsilon']) * masked_policy + self.args['dirichlet_epsilon'] * dirichlet_noise

            #dirch_policy[node.state.clone().to(self.model.device).flatten() == 0.] = float('-inf')
            #dirch_policy = F.softmax(dirch_policy, dim=0)
            dirch_policy *= node.state.clone().to(self.model.device).flatten()
            dirch_policy /= torch.sum(dirch_policy)
            while not node.is_fully_expanded():
                exp_node, exp_node_storage_updated = node.expand(self.node_storage.copy(), dirch_policy)
                self.node_storage.update(exp_node_storage_updated)
        elif self.args['MCTS_add_dirich_noise_par'] <= 0:
            #No Dirichlet Noise
            pass




    def update_starting_node_children_prior(self, starting_node, mcts_action_probs):
        """
        Update the prior probabilities of the children nodes of the starting node.

        This method updates the prior probabilities of the children nodes of `starting_node` to ensure that the
        GraphMCTS updates the UCB values. This helps in selecting different nodes during the selection phase,
        preventing the algorithm from getting stuck with the same best child forever.

        Parameters
        ----------
        starting_node : Node
            The node whose children nodes' prior probabilities need to be updated.
        mcts_action_probs : torch.Tensor
            A tensor containing the updated action probabilities from the MCTS search.
        """
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
