import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from itertools import product
from random import shuffle
import torch
from itertools import product, repeat
from random import shuffle
import csv
import os
import matplotlib.pyplot as plt
from functools import lru_cache, cache
from loguru import logger

class AlphaChompEnv():
    def __init__(self, args, current_player = None):
        self.args = args

    @staticmethod
    def other_player(current_player):
        return 2 if current_player == 1 else 1


    def get_given_state(self, size = None):
        """
        Generate a state grid with specified dimensions and valid moves for a game.

        Parameters
        ----------
        size : tuple of int, optional
            The dimensions of the state matrix to be created. If not provided, the function defaults to using
            the maximum size specified in `self.args`.

        Returns
        -------
        tuple
            A tuple containing the following:
            - state : torch.Tensor
                The state matrix with dimensions `max_size x max_size`, where the specified area is filled with ones.
            - valid_moves : list of tuple
                A list of valid moves (i.e., coordinates) in the state matrix where the value is 1.
            - looser : bool
                A boolean value indicating whether the current state results in a loss.
        """
        if size is None:
            h = self.args['max_size']
            w = self.args['max_size']
        else:
            h = size[0]
            w = size[1]

        state = torch.zeros(self.args['max_size'],self.args['max_size'])
        state[:w, :h] = 1.
        valid_moves = [(i, j) for i in range(w) for j in range(h) if state[i, j] == 1]
        looser = False

        return state, valid_moves, looser

    def check_looser(self, state, action):
        """
        Determine if a player is the loser based on the game state and action taken.

        Parameters
        ----------
        state : torch.Tensor
            A tensor of size `max_size x max_size` representing the current state of the game.
        action : tuple of int
            A tuple specifying the row and column where a move is being made on the game board.

        Returns
        -------
        bool
            A boolean value indicating whether the player is considered a "looser" based on the given state and action.
        """

        if torch.equal(torch.zeros(self.args['max_size'],self.args['max_size']), state):
            looser = True
        elif action == (0,0):
            looser = True
        else:
            looser = False
        return looser

    def get_valid_moves(self, state):
        """
        Return a list of valid moves based on the input state where the value is equal to 1.

        Parameters
        ----------
        state : torch.Tensor
            A 2D tensor representing the game board or grid where the value 1 indicates a valid move.

        Returns
        -------
        list of tuple of int
            A list of tuples, where each tuple represents the coordinates of a valid move in the given `state` tensor.
        """
        h = state.shape[1]
        w = state.shape[0]
        valid_moves = [(i, j) for i in range(w) for j in range(h) if state[i, j] == 1]
        return valid_moves


    def get_next_state(self, state, action):
        """
        Update the game state based on the given action and return the updated state, valid moves,
        information about the loser, and the reward.

        Parameters
        ----------
        state : torch.Tensor
            A 2D tensor representing the current state of the game board.
        action : tuple of int
            A tuple containing the coordinates (row, column) where the action will be performed.

        Returns
        -------
        state : torch.Tensor
            The updated game state.
        valid_moves : list of tuple of int
            A list of valid moves after the action is performed.
        looser : bool
            The result of checking for a loser.
        reward : float
            The reward associated with the action.
        """
        #action = list(action)
        valid_moves = self.get_valid_moves(state)
        if state[action[0], action[1]] == 1:
            state[action[0]:, action[1]:] = 0
            looser = self.check_looser(state, action)
            valid_moves = self.get_valid_moves(state)
            reward = self.get_reward(action)
            self.print_state(state, valid_moves,  looser, action ) if self.args['verbose_A_game'] else None
        elif state not in valid_moves :
            raise ValueError(f"\nThe action: {action} is not a valid move: {valid_moves} on state: {state}")
        else:
            raise ValueError(f"\nOther error in updating state")

        return state, valid_moves, looser, reward



    def get_reward(self, action):
        """
        Assign rewards based on the action taken by the player.

        Parameters
        ----------
        action : tuple of int
            A tuple containing the coordinates (row, column) where the action is performed.

        Returns
        -------
        reward_curr_player : int
            The reward for the current player based on the action taken.
        """
        looser_reward = -1
        winning_reward = 1
        normal_move_reward = self.args['normal_move_reward']

        if action == (0,0):
            reward_curr_player = looser_reward
        else:
            reward_curr_player = normal_move_reward
        return reward_curr_player


    def print_state(self, state,  valid_moves, looser, action = None, reward = None):
        """
        Print the current game state, valid moves, loser status, action taken, and reward received.

        Parameters
        ----------
        state : torch.Tensor
            The current state of the game board.
        valid_moves : list of tuple of int
            The possible moves that a player can make in the current state.
        looser : bool
            Indicates if the current player is the loser.
        action : tuple of int, optional
            The action taken by the player (default is None).
        reward : int, optional
            The reward received by the current player (default is None).

        Returns
        -------
        None
        """
        print(f"\nState: \n{state}\n")
        print(f"\nValid_moves: {valid_moves}")
        if action is not None:
            print(f"\n Action: {action}")
        if looser is  True:
            print(f"\nLooser is the current player")
        else:
            print("\nNo winner yet.")

        if reward is not None:
            print(f"\nReward current player: {reward}")
        return

    def get_reward_and_looser(self, state, action):
        looser = self.check_looser(state, action)
        reward = self.get_reward(action)
        return looser, reward

    def get_all_rectangular_sizes(self):
        """Generate all possible rectangular sizes from max_size x max_size to min_size x min_size."""
        sizes = []
        actual_min_size = 1
        for h in range(self.args['max_size'], actual_min_size - 1, -1):
            for w in range(self.args['max_size'], actual_min_size - 1, -1):
                sizes.append((h, w))
        return sizes

    def create_list_of_tuples(self, n):
        '''Creates a list of tuples with all grid sizes'''
        max_size_tuple = (self.args['max_size'], self.args['max_size'])
        all_sizes = self.get_all_rectangular_sizes()
        stacked_list = list(repeat(max_size_tuple, n)) + all_sizes
        return stacked_list

    def is_square(self, grid_size):
        '''
        Check if a generated grid is a square
        '''
        if grid_size[0] == grid_size[1] :
            return True
        else:
            return False


    def display_game(self, state, valid_moves, looser):
        """
        Generate a visual representation of the Alpha Chomp game state.

        Parameters
        ----------
        state : torch.Tensor
            The current state of the game board.
        valid_moves : list of tuple of int
            The list of valid moves as (row, column) coordinates.
        looser : bool
            Indicates if the current player is the loser.

        Returns
        -------
        None
        """
        state_np = state.cpu().numpy()
        h, w = state_np.shape

        # Create an image array
        img = np.zeros((h, w, 3), dtype=np.uint8)  # RGB image

        for i in range(h):
            for j in range(w):
                if (i, j) == (0, 0):  # Velenom cell
                    if state_np[i, j] == 0:
                        img[i, j] = [255, 255, 255]  # X
                    else:
                        img[i, j] = [255, 0, 0]  # Red
                elif state_np[i, j] == 1:
                    img[i, j] = [0, 0, 255]  # Blue
                else:
                    img[i, j] = [128, 128, 128]  # Gray

        plt.imshow(img)
        plt.title("Alpha Chomp Game State")

        # Display grid
        plt.grid(which='both', color='black', linestyle='-', linewidth=2)
        plt.xticks(np.arange(0.5, w, 1), np.arange(0, w, 1))
        plt.yticks(np.arange(0.5, h, 1), np.arange(0, h, 1))
        plt.gca().set_xticks(np.arange(-0.5, w, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, h, 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=2)

        plt.gca().xaxis.tick_top()

        # Mark cell (0, 0) with 'X' if it equals 0
        if state_np[0, 0] == 0:
            plt.text(0, 0, 'X', fontsize=15, ha='center', va='center', color='black')

        # Awritedown valid moves at the bottom
        valid_moves_str = ', '.join([f'({i},{j})' for i, j in valid_moves])
        plt.figtext(0.5, 0.01, f'Valid Moves: {valid_moves_str}', wrap=True, horizontalalignment='center', fontsize=12)

        # show looser information
        if looser:
            plt.figtext(0.5, 0.06, 'Looser: The current player', wrap=True, horizontalalignment='center', fontsize=12, color='red')
        else:
            plt.figtext(0.5, 0.06, 'No winner yet!', wrap=True, horizontalalignment='center', fontsize=12, color='green')

        plt.show()

'''
#testing
args = {
        'min_size': 2,
        'max_size': 5,      #The maximum size of the game grid or game board
        'save_data_env': False,
        'model_device': 'cpu',  #Device requested for the model
        'device': 'cpu',    #Device for other operations, tipically 'cpu'
        'verbose_A_game': False,    #Bool: if AlphaChompEnv is verbose
        'verbose_mcts': False,     #Bool: if GraphMCTS is verbose
        'verbose_resnet': True,     #Bool: if Resnet is verbose
        'verbose_Alphazero' : False,    #Bool: if Alphazero is verbose
        'mix_up': False,
        'C': 2,     #UCB coefficient
        'MCTS_num_searches': 250,    #How many GraphMCTS searchs to choose 1 move to be played in SelfPlay
        'learn_iterations': 10,     #How many Selplay and Training
        'selfPlay_iterations' : 64,     #Number of games that if will play against itself
        'epochs': 5,       #Training epochs
        'shuffle_replaybuffer': True,   # Bool: to shuffle or not the training Data befor training
        'num_hidden' : 16,      #Number of hidden convolution layers
        'num_resBlocks': 16,        #Number of resonants blocks in Resnet
        'weight_decay' : 0.0001,    #Weight_decay for loss normalization
        'temperature' : 1.5,    #from 0 to +inf. Increase exploration in selfPlay
        'MCTS_add_dirich_noise_par': 2, #if 0 no dirch_noise, if 1 dirch_noise only to root_node = (torch.ones(max_size, max_size)), if >1  dirch_noise to every node
        'dirichlet_epsilon': 0.1,   #epsilon for dirchlet noise
        'dirichlet_alpha': 0.1,     #alpha for dirchlet noise
        'seed' : 42,    #Random seed for reproducibility
        'MCTS_only_path_backpropagation' : True,     #If is desidered to backpropagate only trough path or trough eveery node
        'MCTS_best_child_decay': True,   #If you want to reset the best_child and best_ucb every treshold visits of a node
        'MCTS_updating_children_prior': False,    #If True updates the action_prob of children of the starting_node which than updates in the UCB formula
        'MCTS_progress_disabled': True,
        'Debug' : False,
        'dis_Splay_progress': True,
        'dis_Train_progress': True,
        'batch_size': 64,   #Batch_size dimension. It's also implemented dynamic batching
        'lr' : 0.001,    #Learning rate of the optimizer
        'normal_move_reward': 0.01    #The reward of a legal move apart from winning\loosing move
    }



chomp_env = AlphaChompEnv(args)

state, valid_moves, looser = chomp_env.get_given_state()
chomp_env.print_state(state, valid_moves, looser)
action = (1,1)
state, valid_moves, looser, reward = chomp_env.get_next_state(state, action)
chomp_env.print_state(state, valid_moves, looser, reward = reward)
action1 = (0,0)
state, valid_moves, looser, reward = chomp_env.get_next_state(state, action1)
chomp_env.print_state(state, valid_moves, looser, action1, reward)
rect_states = chomp_env.get_all_rectangular_sizes()

print(f"\nrect_states: {rect_states}\nLen(rect_states): {len(rect_states)}")
'''
