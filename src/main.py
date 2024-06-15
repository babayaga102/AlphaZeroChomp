import argparse
import sys
import time
from Alpha_Chomp_Env import AlphaChompEnv
from Resnet import ResNet
from Alpha_GraphMCTS import GraphMCTS
from AlphaZero import AlphaZeroChomp



class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        print(f"\nError: {message}")
        print("Invalid value for --mode. Allowed values are 'play' or 'train_and_play'.")
        sys.exit(2)



def main():

    def parse_arguments():
        """
        The function `parse_arguments` uses the `argparse` module in Python to define and parse
        command-line arguments for a program related to AlphaZero Chomp.
        :return: The `parse_arguments` function is returning the parsed arguments from the command line
        using the `argparse` module in Python.
        """
        parser = argparse.ArgumentParser(description='AlphaZero Chomp')
        parser.add_argument('--mode', choices=['play', 'train_and_play'], default='play',
                            help='Choose whether to play a game or train the neural network and then play a game')
        return parser.parse_args()



    args = {
        'min_size': 2,      #It's highly suggested to set min_size = 2
        'max_size': 4,      #The maximum size of the game grid or game board
        'save_data_env': False,
        'model_device': 'cpu',  #Device requested for the model. Supports 'cpu' or 'mps'. To train on 'mps' set 'MCTS_set_equal_prior' = True
        'device': 'cpu',    #Device for other operations Graph-MCTS search, tipically 'cpu'
        'verbose_A_game': False,    #Bool: if AlphaChompEnv is verbose
        'verbose_mcts': False,     #Bool: if GraphMCTS is verbose
        'verbose_resnet': True,     #Bool: if Resnet is verbose
        'verbose_Alphazero' : False,    #Bool: if Alphazero is verbose
        'verbose_Play': True,      #Bool: prints more when playing
        'C': 2,     #UCB coefficient to balance exploration/exploitation
        'MCTS_num_searches': 300,    #How many GraphMCTS searchs to choose 1 move to be played in SelfPlay
        'learn_iterations': 2,     #How many Selplay and Training
        'selfPlay_iterations' : 128,     #Number of games that if will play against itself for every SelfPlay. It's recomended to set selfPlay_iterations ≈ 50 + max_size**2
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
    # Instantiating stuff...
    resnet = ResNet(args)
    Alpha_game = AlphaChompEnv(args)
    Alpha_MCTS = GraphMCTS(Alpha_game, model=resnet, args=args)
    alphazero = AlphaZeroChomp(args, resnet, Alpha_game, Alpha_MCTS)

    cli_args = parse_arguments()

    if cli_args.mode == 'train_and_play':
        print("Training the neural network and then playing the game...")
        alphazero.learn()
        print("Playing the game...")
        time.sleep(2)
        alphazero.play_game()
    elif cli_args.mode == 'play':
        print("Playing the game...")
        alphazero.play_game()





if __name__ == "__main__":
    main()






#
