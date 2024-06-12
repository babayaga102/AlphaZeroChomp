import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = self.args['seed']
        num_hidden = self.args['num_hidden']
        num_resBlocks = self.args['num_resBlocks']
        self.action_size = self.args['max_size']**2
        self.coordinates_matrix = self.create_coordinate_matrix()   #Adds coordinates to exploit simmetries

        self.startBlock = nn.Sequential(
            nn.Conv2d(4, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
 
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.action_size, self.action_size),
            nn.Softmax(dim=1)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.action_size , 1),  # Adjust the input size here
            nn.Tanh()
        )
        self.set_seed()
        self.set_device()
        self.apply(self._init_weights)


        if self.args.get('verbose_resnet', False):
            print(f"\nResNet Device: {next(self.parameters()).device}")
            print(f"ResNet:\n{self}")

    
    def set_device(self):
        """
        This function sets the device for a PyTorch model based on the specified model device and
        availability of different devices.
        """
        if self.args['model_device'] == 'mps' and torch.backends.mps.is_available():
             self.to('mps')
             self.device = 'mps'
        elif self.args['model_device'] == 'cuda' and torch.cuda.is_available():
             self.to('cuda')
             self.device = 'cuda'
        else:
            self.to('cpu')
            self.device = 'cpu'

    def create_coordinate_matrix(self):
        '''Creates a symmetric matrix (max_size, max_size) which has only positive numbers always different
        on the diagonal (from 0 to max_size - 1) but symmetric to encode the positional arguments symmetries
        [[0, 3, 4],
         [3, 1, 5],
         [4, 5, 2]]'''
        max_size = self.args['max_size']
        matrix = torch.zeros((max_size, max_size), dtype=torch.float32)

        for i in range(max_size):
            matrix[i, i] = i

        positive_value = max_size
        for i in range(max_size):
            for j in range(i + 1, max_size):
                matrix[i, j] = positive_value
                matrix[j, i] = positive_value
                positive_value += 1

        return matrix

    def set_seed(self):
        '''Set all possible seeds to make the experiment reproducibol'''
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # if you are using multi-GPU but im not now
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _init_weights(self, module):
        '''Set the weights and biases to make the experiment reproducibol'''
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def mask_and_renormalize(self, x_input, unrenorm_policy):
        '''Masks and renormalize the policy using Softmax function. This method works pretty well,
        however has some drawbacks. When the probability of bad move, which is close to 0, pass trought
        Softmax function gets smooth out and became much higher. This allow the probability of a bad move to became
        much higher than should be. Using this masking method the network will converge slower. Conversely, it
        adds kind of noise in the network when choicing which can be positive.
        It can work with args['MCTS_set_equal_prior'] = False.
        This function can distinguish between trayning and playing input using DynamicBatching. '''
        if x_input.dim() == 2:  # Unbatched input
            masked_policy = unrenorm_policy.clone().squeeze()
            x_input_flat = torch.flatten(x_input)
            masked_policy[x_input_flat == 0.] = float('-inf')
            masked_policy = F.softmax(masked_policy, dim=0)

        elif x_input.dim() == 3:  # Batched input
            batch_size = x_input.size(0)
            masked_policy = unrenorm_policy.clone()
            for i in range(batch_size):
                x_input_flat = torch.flatten(x_input[i])

                masked_policy[i][x_input_flat == 0] = float('-inf')
            masked_policy = F.softmax(masked_policy, dim=1) #The dim=1 because dim=0 is batch_dim

        return masked_policy


    def mask_and_renorm_NoSoftmax(self, x_input, unrenorm_policy):
        '''Masks and renormalize the policy using No-Softmax function. This method works the Best,but
        it must be used in conjuction with args['MCTS_set_equal_prior'] = True. It do not smooth out
        The probability of bad moves and allow to a faster convergenge of the Resnet network.
        This function can distinguish between training and playing input using DynamicBatching. '''

        x_input = x_input.clone()
        if x_input.dim() == 2:  # Unbatched input
            masked_policy = unrenorm_policy.clone().squeeze()
            x_input_flat = torch.flatten(x_input)
            masked_policy = masked_policy * x_input_flat  #  element-wise multiplication
            sum_probs = torch.sum(masked_policy)
            #print(f"Unbatched sum_probs: {sum_probs}")
            if sum_probs > 0:
                masked_policy = masked_policy / sum_probs
                masked_policy = masked_policy
            else:
                masked_policy = torch.zeros_like(masked_policy)

        elif x_input.dim() == 3:  # Batched input
            batch_size = x_input.size(0)
            masked_policy = unrenorm_policy.clone()

            for i in range(batch_size):
                x_input_flat = torch.flatten(x_input[i])
                masked_policy[i] = masked_policy[i] * x_input_flat
                sum_probs = torch.sum(masked_policy[i])

                if sum_probs > 0:
                    masked_policy[i] = masked_policy[i] / sum_probs
                else:
                    masked_policy[i] = torch.zeros_like(masked_policy[i])

        return masked_policy


    def prepare_data(self, x_input, current_actions=None, opponent_actions=None):
        """
        The `prepare_data` function in Python processes input data for a neural network model by
        incorporating current and opponent actions.
        
        :param x_input: It seems like the description of `x_input` is missing in your message. If you
        provide me with the details of `x_input`, I can assist you further with the `prepare_data` function
        :param current_actions: The `current_actions` parameter is a list of actions taken by the current
        player. Each action is represented as a tuple containing the coordinates of the action in the game
        grid. For example, `current_actions = [(0, 1), (2, 2), (1, 0)]
        :param opponent_actions: The `opponent_actions` parameter in the `prepare_data` method represents
        the actions taken by the opponent in a game or scenario. These actions are used to update the
        opponent's state in the input data tensor. In the provided code snippet, the opponent's actions are
        used to populate the `op
        :return: The `prepare_data` method returns the modified input data `x_input_mod` after processing
        based on the input dimensions and provided actions for both the current player and opponent. The
        returned `x_input_mod` is a tensor that includes the current player's actions, opponent's actions,
        the original input `x_input`, and a coordinates matrix stacked along the batch dimension if the
        input is batched.
        """
        if x_input.dim() == 2:  # Unbatched input
            current = torch.zeros(self.args['max_size'], self.args['max_size'])
            opponent = torch.zeros(self.args['max_size'], self.args['max_size'])
            if current_actions is not None and opponent_actions is not None:
                for action in current_actions:
                    current[action[0], action[1]] = 1.
                for action in opponent_actions:
                    opponent[action[0], action[1]] = -1.
            x_input_mod = torch.stack((current, x_input, opponent, self.coordinates_matrix), dim=0)
            x_input_mod = x_input_mod.unsqueeze(0)  # Add batch dimension

        elif x_input.dim() == 3:  # Batched input
            batch_size = x_input.size(0)    #dynamic batching
            x_input_mod = []
            for i in range(batch_size):
                current = current_actions[i]
                opponent = opponent_actions[i]
                x_input_m = torch.stack((current, x_input[i], opponent, self.coordinates_matrix), dim=0)
                x_input_mod.append(x_input_m)
            x_input_mod = torch.stack(x_input_mod, dim=0)  # Stack along the batch dimension

        else:
            raise ValueError(f"The input shape: {x_input.dim()} do not correspond to the possibilities!")

        return x_input_mod


    def forward(self, x_input, current_actions=None, opponent_actions=None):
        x_prep = self.prepare_data(x_input, current_actions, opponent_actions)
        x = self.startBlock(x_prep)
        for resBlock in self.backBone:
            x = resBlock(x)
            policy = self.policyHead(x)
            value = self.valueHead(x)
            masked_policy = self.mask_and_renorm_NoSoftmax(x_input, policy) if self.args['MCTS_set_equal_prior'] else self.mask_and_renormalize(x_input, policy)
        return masked_policy, policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()

        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, stride = 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual   # Resonance
        x = F.relu(x)
        return x


'''
args = {
    'min_size': 2,
    'max_size': 6,
    'save_data_env': False,
    'model_device': 'cpu',
    'verbose_game': False,
    'verbose_mcts': True,
    'verbose_resnet': True,
    'mix_up': False,
    'C': 1.4,
    'num_searches': 300,
    'num_hidden' : 64,
    'num_resBlocks': 16,
    'seed' : 42,
    'only_path_backpropagation' : True,
    'MCTS_set_equal_prior':True    #Set equal prior for all child node when expanded
}
resnt = ResNet(args)
print(f"{resnt.coordinates_matrix}")'''
