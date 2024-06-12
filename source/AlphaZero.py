import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from Alpha_Chomp_Env import AlphaChompEnv
from Resnet import ResNet
from Alpha_GraphMCTS import GraphMCTS



class AlphaZeroChomp():
    def __init__(self, args, model, Alpha_game, Alpha_MCTS):
        self.loss_hist = []
        self.args = args
        self.A_game = Alpha_game
        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr= self.args['lr'], weight_decay=self.args['weight_decay'])
        self.mcts = Alpha_MCTS
        self.action_size = self.args['max_size'] ** 2
        torch.manual_seed(self.args['seed'])
        np.random.seed(self.args['seed'])
        random.seed(self.args['seed'])
        self.all_rectangular_sizes = self.A_game.get_all_rectangular_sizes()    # max_size ** 2
        self.optim_stategy_counter_sqr = 0
        self.hist_sqr_optim_str = []
        self.P1_winning_rate = 0
        self.P1_winning_rate_history = []
        self.hist_avg_eval_turn = []

    def learn(self):
        '''Calls SelfPlay and Train to train the Resnet for a number of lear_iterations'''
        retrain = input('Do you want to retrain a model? (y/n): ').strip().lower()
        self.load_model() if retrain == "y" else None
        alfa_start_time = time.time()
        max_size_iterations = min(self.args['selfPlay_iterations'], self.args['selfPlay_iterations'] - len(self.A_game.get_all_rectangular_sizes()))
        self.list_grids_size = self.A_game.create_list_of_tuples(max_size_iterations)


        for iteration in tqdm(range(self.args['learn_iterations']), desc='learn_iter'):
            TrainMemory = []

            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args['selfPlay_iterations']), desc='Selfplay_iter',disable=self.args['dis_Splay_progress']):
                TrainMemory += self.selfPlay(selfPlay_iteration)  #Extend the training memory

            #self.model.train()
            for epoch in tqdm(range(self.args['epochs']),desc='Training_epoch',disable=self.args['dis_Train_progress']):
                self.train(TrainMemory)


            self.SelfPlay_evaluation()    #Evaluate the improvement of the model strategies

            self.save_checkpoints(iteration)
        alfa_end_time = time.time()
        #if self.args['verbose_Alphazero']:
        print(f"\nLearn Time Alphazero: {alfa_end_time - alfa_start_time} seconds for ...")
        root_node = self.mcts.find_node_by_state(torch.ones(self.args['max_size'], self.args['max_size']))
        #self.mcts.visualize_mcts(root_node)    #To show the GraphMCTS at the end of the learning. It may be extremely slow. I mean hours...
        self.plot_analitics()
        with torch.no_grad():
            masked_policy, policy, value = self.model(torch.ones(self.args['max_size'], self.args['max_size']))
            self.plot_policy(policy)

    def train(self, TrainMemory):
        self.model.train()
        random.shuffle(TrainMemory) if self.args['shuffle_replaybuffer'] else None
        # Calculate the number of batches, for Dybamic Batching
        num_batches = (len(TrainMemory) + self.args['batch_size'] - 1) // self.args['batch_size']

        for batch_index in range(num_batches):
            start_index = batch_index * self.args['batch_size']
            end_index = min(start_index + self.args['batch_size'], len(TrainMemory))
            sample = TrainMemory[start_index:end_index]

            state, policy_target, curr_actions, opp_actions, value_target = zip(*sample)
            state, policy_target, curr_actions, opp_actions, value_target = (
                np.array(state), np.array(policy_target),
                np.array(curr_actions), np.array(opp_actions), np.array(value_target)
            )

            state, policy_target, curr_actions, opp_actions, value_target = (
                torch.tensor(state, dtype=torch.float32, device = self.model.device),
                torch.tensor(policy_target, dtype=torch.float32, device = self.model.device),
                torch.tensor(curr_actions, dtype=torch.float32, device = self.model.device),
                torch.tensor(opp_actions, dtype=torch.float32, device = self.model.device),
                torch.tensor(value_target, dtype=torch.float32, device = self.model.device).view(-1, 1)
            )

            # Print debug information to ensure tensors are correct
            print("State tensor :", state) if self.args['verbose_Alphazero'] else None
            #print("State tensor size:", state.shape)     #Debug
            #print(f"\ncurr_actions shape: {curr_actions.shape}")     #Debug
            #print("Policy target tensor shape:", policy_target.shape)     #Debug
            #print("Value target tensor shape:", value_target.shape)     #Debug

            masked_policy, policy_logit, value = self.model(state, curr_actions, opp_actions)

            # shapes for dynamic batching
            current_batch_size = state.size(0)  # Dynamic batching
            masked_policy = masked_policy.view(current_batch_size, -1)  # Flatten the masked_policy
            policy_logit = policy_logit.view(current_batch_size, -1)
            policy_target = policy_target.view(current_batch_size, -1)  # Flatten the policy_target

            #print(f"\nMasked policy shape: {masked_policy.shape}")     #Debug
            #print(f"\nPolicy target shape: {policy_target.shape}")     #Debug
            #print(f"\n Sum of a masked policy: {torch.sum(masked_policy[0])}")     #Debug
            #print(f"\nValue output model shape: {value.shape}")     #Debug
            #print(f"\nValue output model: {value}") if self.args['verbose_Alphazero'] else None
            #print(f"\nValue target: {value_target}") if self.args['verbose_Alphazero'] else None

            #masked_policy_log = torch.log(masked_policy + 1e-40)  # Convert to logit
            policy_logit_log = torch.log(policy_logit + 1e-40 )

            policy_loss = F.cross_entropy(policy_logit_log, policy_target)
            value_loss = F.mse_loss(value, value_target)
            loss = policy_loss + value_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.loss_hist.append(loss.item())
        return

    def selfPlay(self, selfPlay_iteration):
        '''Collect Data for Training'''
        #print(f"\nStarting selfplay!")
        BufferMemory = []
        current_player = 1
        self.turn = 0
        if self.args['seflPlay_subgrids']:
            grid_size = self.list_grids_size[selfPlay_iteration]
            state, valid_moves, looser = self.A_game.get_given_state(grid_size)

        else:
            state, valid_moves, looser = self.A_game.get_given_state((self.args['max_size'],self.args['max_size']))

        while not looser:
            print(f"\nNew round") if self.args['verbose_Alphazero'] else None
            print(f"\nCurrent state:\n{state}") if self.args['verbose_Alphazero'] else None

            self.turn += 1
            action_prob_mcts, action_value_mcts = self.mcts.search(state.clone(), valid_moves.copy())
            action_prob_mcts = action_prob_mcts.flatten()
            print(f"\nAlphazero Action Probs:{action_prob_mcts}") if self.args['verbose_Alphazero'] else None
            #Apply tempetarure fro exploration/exploitation increase
            temp_action_prob_mcts = action_prob_mcts.clone() ** (1/self.args['temperature'])

            action = torch.multinomial(temp_action_prob_mcts, 1).item()
            action = self.mcts.num_to_move(action)
            print(f"\nAlphazer state: {state}") if self.args['verbose_Alphazero'] else None
            current_actions, opponent_actions = self.collect_actions(action)

            # Append a copy of the state tensor to BufferMemory
            BufferMemory.append((state.clone(), action_prob_mcts.clone(), current_actions.clone(), opponent_actions.clone(), current_player))
            print(f"\nAlphazero action: {action}\n") if self.args['verbose_Alphazero'] else None

            state, valid_moves, looser, reward = self.A_game.get_next_state(state.clone(), action)
            print(f"\nAlphazero looser: {looser}\n") if self.args['verbose_Alphazero'] else None

            if looser is True:
                ReturnMemory = []
                normal_move_reward = self.args['normal_move_reward']  # Define your normal move reward value here
                for i, (hist_state, hist_action_probs_mcts, hist_curr_actions, hist_opp_actions, hist_player) in enumerate(BufferMemory):
                    hist_outcome = reward if hist_player == current_player else -reward
                    if i < len(BufferMemory) - 2:
                        hist_outcome = normal_move_reward
                    hist_action_probs_mcts = hist_action_probs_mcts.view(self.args['max_size'], self.args['max_size'])
                    ReturnMemory.append((hist_state, hist_action_probs_mcts, hist_curr_actions, hist_opp_actions, hist_outcome))
                #print(f"\nReturnMemory:\n{ReturnMemory}")
                #print(f"\nlenght_of_ReturnMemory: {len(ReturnMemory)}\nReturnMemory:\n{ReturnMemory}")
                #exit()
                return ReturnMemory

            current_player = self.A_game.other_player(current_player)




    def play_game(self, grid_size= None):
        self.load_model()
        self.model.eval()
        current_player_actions = []
        opponent_player_actions = []
        h = int(input(f"Enter height of the grid to be played (2 ≤ h ≤ {self.args['max_size']}): "))
        w = int(input(f"Enter width of the grid to be played (2 ≤ w ≤ {self.args['max_size']}): "))
        grid_size = (h,w)

        if grid_size is not None:
            if int(grid_size[0]) > self.args['max_size'] or int(grid_size[1]) > self.args['max_size']:
                raise ValueError(f"The grid_size is bigger than the argument max_size: {(self.args['max_size'], self.args['max_size'])}")

            elif  int(grid_size[0]) < self.args['min_size'] or int(grid_size[1]) < self.args['min_size'] :
                raise ValueError(f"The grid_size is smaller than the argument min_size: {(self.args['min_size'], self.args['min_size'])}")
            else:
                #initialize special state
                state, valid_moves, looser = self.A_game.get_given_state(grid_size)
        else:
            # Initialize the normal game state
            state, valid_moves, looser = self.A_game.get_given_state((self.args['max_size'], self.args['max_size']))
        # Ask if the human wants to play first or second
        AI_first_player = int(input(f"\nEnter 1 if you want to play first, else the AI will begin!\n"))
        if AI_first_player == 1:
            print(f"Great, you'll be the first player!")
        else:
            print(f"Ok, AI goes first!")


        self.A_game.print_state(state, valid_moves, looser)
        self.A_game.display_game(state, valid_moves, looser)
        while not looser:
            if AI_first_player == 1:
                # Human plays

                move = int(input("Enter your move (0 ≤ int ≤ {self.args['max_size']**2 - 1}): "))
                move = self.mcts.num_to_move(move)
                current_player_actions.append(move)
                current_player_actions, opponent_player_actions = opponent_player_actions.copy(), current_player_actions.copy()
                if move not in valid_moves:
                    print("Invalid move. Try again.")
                    continue
                state, valid_moves, looser, reward = self.A_game.get_next_state(state.clone(), move)
                self.A_game.print_state(state, valid_moves, looser)
                self.A_game.display_game(state, valid_moves, looser)
                AI_first_player = 2  # Switch to AI turn

            else:
                # AI plays
                with torch.inference_mode():
                    masked_policy, policy, value = self.model(torch.tensor(state.clone().detach().cpu(), dtype=torch.float32),current_player_actions.copy(), opponent_player_actions.copy())
                    action_prob_mcts, _ = self.mcts.search(state.clone(), valid_moves.copy())
                    playing_node = self.mcts.find_node_by_state(state.clone()) if self.args['verbose_Play'] else None
                    print(f"\nMasked_policy:\n{masked_policy.view(self.args['max_size'], self.args['max_size'])}") if self.args['verbose_Play'] else None
                    print(f"\nprobabiliy sum:{torch.sum(masked_policy)}")  if self.args['verbose_Play'] else None
                    print(f"\nValue fo the state: {value}") if self.args['verbose_Play'] else None
                    print(f"\nMCTS Action Prob: {action_prob_mcts}") if self.args['verbose_Play'] else None
                    print(f"\nBest_UCB: {playing_node.best_ucb:.5f} Best_child_action from starting Node: {playing_node.children[playing_node.best_child].action}")   if self.args['verbose_Play'] else None
                    self.plot_policy(masked_policy) if self.args['verbose_Play'] else None

                    move = torch.argmax(masked_policy).item()
                move = self.mcts.num_to_move(move)
                current_player_actions.append(move)
                current_player_actions, opponent_player_actions = opponent_player_actions.copy(), current_player_actions.copy()
                print(f"AI chooses move: {move}")
                state, valid_moves, looser, reward = self.A_game.get_next_state(state, move)
                self.A_game.print_state(state, valid_moves, looser)
                self.A_game.display_game(state, valid_moves, looser)
                AI_first_player = 1  # Switch to human turn


        #New game
        new_game = int(input(f"Enter 1 if you want to play a new game, else quit: "))
        if new_game == 1:
            self.play_game()
        else:
            exit()


    def switch_num(self, tup):
        '''
        Switches the numbers in a tuple
        '''
        return (tup[1], tup[0])

    def collect_actions(self, action):
        '''Collects the actions during the game and creates 2 matrix which displays the actions of
        current player and opponent. Then, these 2 matrix will be saved in ReturnMemory and will be
        feed into the NN Resnet model '''
        if self.turn == 1:
            #Create blank tensors at first turn
            self.current_actions_mat = torch.zeros(self.args['max_size'],self.args['max_size'])
            self.opponent_actions_mat = torch.zeros(self.args['max_size'],self.args['max_size'])
        #multiply by -1 to switch side
        self.current_actions_mat, self.opponent_actions_mat = (self.opponent_actions_mat * (-1), self.current_actions_mat * (-1))
        #print(f"\n self.current_actions_mat: {self.current_actions_mat}")   #Debug

        if self.current_actions_mat[action[0], action[1]] == 0:
            self.current_actions_mat[action[0], action[1]] = 1
        else:
            raise ValueError(f"The current action:{action} has already been taken!")
        return self.current_actions_mat.clone() , self.opponent_actions_mat.clone()

    def save_checkpoints(self, iteration):
        directory = 'checkpoints'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = directory + '/'

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, f"{path}model_{iteration}.pt")

        # Save optimizer state
        torch.save({
            'optimizer_state_dict': self.optim.state_dict()
        }, f"{path}optimizer_{iteration}.pt")

    def plot_analitics(self):
        '''Plots the loss and other analitics to estimate the goodness of the game'''
        loss_array = np.array(self.loss_hist)
        plt.figure('Training Loss Over Batches', figsize=(10, 6))
        plt.plot(loss_array, label='Training Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Batches')
        plt.grid(True)
        plt.text(0.98, 0.98,
                f"MCTS_updating_children_prior: {self.args['MCTS_updating_children_prior']}\nMCTS_only_path_backpropagation: {self.args['MCTS_only_path_backpropagation']}\n \
                MCTS_set_equal_prior: {self.args['MCTS_set_equal_prior']}\nmax_size: {self.args['max_size']}\nlearn_iterations: {self.args['learn_iterations']}\nbatch_size: {self.args['batch_size']}\n \
                selfPlay_iterations: {self.args['selfPlay_iterations']}\nMCTS_num_searches: {self.args['MCTS_num_searches']}\nepochs: {self.args['epochs']}\n \
                weight_decay: {self.args['weight_decay']}\nhidden_layers: {self.args['num_hidden']}\nnum_resBlocks: {self.args['num_resBlocks']}\n \
                temperature: {self.args['temperature']}\ndirichlet_epsilon: {self.args['dirichlet_epsilon']}\ndirichlet_alpha: {self.args['dirichlet_alpha']}\n \
                learning_rate: {self.args['lr']}\nseed: {self.args['seed']}\nC: {self.args['C']}\n \
                ",
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.6)
                )

        if len(self.P1_winning_rate_history) > 0 and len(self.hist_sqr_optim_str) > 0:
            fig, axs = plt.subplots(3, 1, figsize=(8, 8))

            # plotting P1 winning rate
            axs[0].plot(np.array(self.P1_winning_rate_history), label='P1 Winning Rate')

            axs[0].set_ylabel('P1 winning rate')
            axs[0].set_title('P1 winning rate')
            axs[0].axhline(y=0.5, color='r', linestyle='--')
            axs[0].grid(True)
            axs[0].legend()
            axs[0].text(0.3, 0.9, f"max_size: {self.args['max_size']}", transform=axs[0].transAxes,
                        fontsize=12, verticalalignment='top', horizontalalignment='right')

            # plotting Optim strategy histo on square state
            axs[1].plot(np.array(self.hist_sqr_optim_str), label='Optim strategy square')
            axs[1].set_ylabel('Optim strategy square')
            axs[1].set_title('Optimal strategy history on square state')
            axs[1].grid(True)
            axs[1].legend()

            axs[2].plot(np.array(self.hist_avg_eval_turn), label='Average turn in testing')
            axs[2].set_title('Average turn in testing')
            axs[2].set_ylabel('Average testing turn')
            axs[2].set_xlabel('Learning iteration')
            axs[2].grid(True)

            plt.tight_layout()

        plt.show()

    def load_model(self):
        # List available model checkpoints
        checkpoints = [f for f in os.listdir('checkpoints') if f.startswith('model_') and f.endswith('.pt')]
        checkpoints.sort(reverse=True)

        print("\nAvailable checkpoints:")
        for i, checkpoint in enumerate(checkpoints):
            print(f"{i + 1}. {checkpoint}")

        # Ask user to choose a checkpoint or load the latest
        choice = input("\nEnter the number of the checkpoint to load (or press Enter to load the latest): ")
        if choice == "":
            checkpoint_path = os.path.join('checkpoints', checkpoints[0])
        else:
            checkpoint_path = os.path.join('checkpoints', checkpoints[int(choice) - 1])

        # Load the model state
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load the corresponding optimizer state
        iteration = checkpoint_path.split('_')[-1].split('.')[0]
        optimizer_checkpoint_path = os.path.join('checkpoints', f'optimizer_{iteration}.pt')
        optimizer_checkpoint = torch.load(optimizer_checkpoint_path)
        self.optim.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])

        print(f"Loaded model checkpoint: {checkpoint_path}")
        print(f"Loaded optimizer checkpoint: {optimizer_checkpoint_path}")

    def plot_policy(self, action_probs):

        action_probs = action_probs.cpu().numpy()
        if action_probs.ndim == 2:
            action_probs = np.squeeze(action_probs)

        if action_probs.ndim != 1:
            raise ValueError("action_probs should be a 1D array after squeezing")
        plt.figure(figsize=(6, 5))
        plt.bar(range(len(action_probs)), action_probs, width=0.8)
        plt.ylim(0, 1)
        plt.xlabel('Actions')
        plt.ylabel('Probability')
        plt.title('AI Policy')
        plt.show()

    def test_optim_strategies(self, grid_size, P1_actions, P2_actions, winner):
        '''Test if the action are optimal strategy or not on a square root. If the grid_size
        is not square it collects and update the winning rate'''
        if self.A_game.is_square(grid_size) is True and winner == 1:
            self.P1_winning_rate += 1
            if P1_actions[0] == (1,1) and len(P2_actions) > 1:

                adjusted_P1_moves = P1_actions[1:]  # skip the first element
                adjusted_P2_moves = P2_actions[:-1]  # skip the last element
                switched_P2_moves = [self.switch_num(tup) for tup in adjusted_P2_moves]
                #Check that P1 replies simmetrically to the moves of P2
                self.sqr_optimal_strategy = all(p1 == p2 for p1, p2 in zip(adjusted_P1_moves, switched_P2_moves))
                self.optim_stategy_counter_sqr += 1 if self.sqr_optimal_strategy  else 0
                print(f"\nOptimal Strategy on grid_size: {grid_size}!") if self.args['verbose_Alphazero'] else None

        elif self.A_game.is_square(grid_size) is True and winner == 2:
            pass
        else:
            #if the grid is not square
            if winner == 1:
                self.P1_winning_rate += 1
            else:
                pass

    def SelfPlay_evaluation(self):
        '''The AI plays againt itself at is best and determines a winning rate and collects
         the best optimal strategies on square roots.'''
        self.model.eval()
        self.turn_counter = 0

        for grid_size in self.all_rectangular_sizes:
            state, valid_moves, looser = self.A_game.get_given_state(grid_size)
            self.eval_turn = 0
            current_player_actions = []
            opponent_player_actions = []

            while not looser:
                self.eval_turn += 1
                with torch.inference_mode():
                    masked_policy, policy, value = self.model(state.clone(),current_player_actions.copy(), opponent_player_actions.copy())

                move = torch.argmax(masked_policy).item()
                move = self.mcts.num_to_move(move)
                current_player_actions.append(move)
                state, valid_moves, looser, reward = self.A_game.get_next_state(state, move)
                if looser is True:
                    if self.eval_turn % 2 == 0:     #P2 is the Looser and the current player
                        P2_actions, P1_actions = current_player_actions.copy(), opponent_player_actions.copy()
                        winner = 1
                    else:       #P1 is the Looser and the current player
                        P1_actions, P2_actions = current_player_actions.copy(), opponent_player_actions.copy()
                        winner = 2
                    break
                current_player_actions, opponent_player_actions = opponent_player_actions.copy(), current_player_actions.copy()
            self.test_optim_strategies(grid_size, P1_actions, P2_actions, winner)
            self.turn_counter += self.eval_turn

        self.hist_sqr_optim_str.append(self.optim_stategy_counter_sqr)
        self.P1_winning_rate /= len(self.all_rectangular_sizes)
        self.P1_winning_rate_history.append(self.P1_winning_rate)
        self.turn_counter /= len(self.all_rectangular_sizes)
        self.hist_avg_eval_turn.append(self.turn_counter)
        self.optim_stategy_counter_sqr = 0
        self.P1_winning_rate = 0

'''
# beta testing
if __name__ == "__main__":
    # Define your A_game class and args
    args = {
        'min_size': 2,      #It's highly suggested to set min_size = 2
        'max_size': 5,      #The maximum size of the game grid or game board
        'save_data_env': False,
        'model_device': 'cpu',  #Device requested for the model
        'device': 'cpu',    #Device for other operations, tipically 'cpu'
        'verbose_A_game': False,    #Bool: if AlphaChompEnv is verbose
        'verbose_mcts': False,     #Bool: if GraphMCTS is verbose
        'verbose_resnet': True,     #Bool: if Resnet is verbose
        'verbose_Alphazero' : False,    #Bool: if Alphazero is verbose
        'verbose_Play': False,      #Bool: prints more when playing
        'C': 2,     #UCB coefficient to balance exploration/exploitation
        'MCTS_num_searches': 300,    #How many GraphMCTS searchs to choose 1 move to be played in SelfPlay
        'learn_iterations': 15,     #How many Selplay and Training
        'selfPlay_iterations' : 128,     #Number of games that if will play against itself for every SelfPlay
        'seflPlay_subgrids': True,      #If True, allow SelfPlay over all subgrids of the biggest state Board
        'epochs': 5,       #Training epochs
        'shuffle_replaybuffer': True,   # Bool: to shuffle or not the training Data befor training
        'num_hidden' : 10,      #Number of hidden convolution layers
        'num_resBlocks': 16,        #Number of resonants blocks in Resnet Model
        'weight_decay' : 0.0001,    #Weight_decay for loss normalization
        'temperature' : 1.5,    #from 0 to +inf. Increase exploration in SelfPlay
        'MCTS_add_dirich_noise_par': 1, #if 0 No dirch_noise, if 1 dirch_noise only to root_node = (torch.ones(max_size, max_size)), if >1  dirch_noise to every node
        'dirichlet_epsilon': 0.1,   #epsilon for dirchlet noise
        'dirichlet_alpha': 0.1,     #alpha for dirchlet noise
        'seed' : 42,    #Random seed for reproducibility
        'MCTS_only_path_backpropagation' : True,     #If is desidered to backpropagate only trough path or trough eveery node
        'MCTS_best_child_decay': True,   #If you want to reset the best_child and best_ucb every treshold visits of a node
        'MCTS_updating_children_prior': True,    #If True updates the action_prob of children of the starting_node which than updates in the UCB formula
        'MCTS_progress_disabled': True,
        'MCTS_set_equal_prior':False,    #BOOL: Set equal prior for all child node when expanded. Uses Resnet.mask_and_renorm_NoSoftmax if set to True
        'dis_Splay_progress': True,     #Bool: Display progrss bar for SelfPlay
        'dis_Train_progress': True,     #Bool: Display progrss bar for Training
        'batch_size': 64,   #Batch_size dimension. It's also implemented dynamic batching
        'lr' : 0.001,    #Learning rate of the optimizer
        'normal_move_reward': 0.003,    #The reward of a legal move apart from winning\loosing move

    }

    # Instantiate the A_game and MCTS
    resnet = ResNet(args)
    Alpha_game = AlphaChompEnv(args)
    Alpha_MCTS = GraphMCTS(Alpha_game, model=resnet, args=args)
    alphazero = AlphaZeroChomp(args, resnet, Alpha_game, Alpha_MCTS)
    alphazero.learn()
    alphazero.play_game()
'''
