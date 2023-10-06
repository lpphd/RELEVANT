from torch import nn
from utils import *
from torch.distributions import Bernoulli
import torch
import torch.nn.functional as F


class RLPolicies(nn.Module):
    def __init__(self, model_config, data_config):
        super(RLPolicies, self).__init__()

        self.n_epochs = model_config["n_epochs"]
        self.timesteps = data_config["timesteps"]
        ## Dimensions of hidden state is determined by number of channels, filters per channel and features per filter
        ## In addition, as described in the paper, a float indicates the point in time of processing and the history of actions
        self.hidden_state_dim = 1 * (data_config['channels'] * model_config['num_filters_per_channel'] * model_config[
            'num_feats_per_filter']) + 1 + (model_config['num_checkpoints'] + 1) * (model_config['num_channel_slices'])
        self.n_channel_slices = model_config['num_channel_slices']
        self.num_checkpoints = model_config['num_checkpoints']
        self.discount = model_config['discount_rewards']
        self.n_classes = data_config["n_classes"]
        self.stop_action_thresh = model_config['stop_action_threshold']

        filternetOut = self.n_channel_slices

        self.filterPolicyNet = createNet(self.hidden_state_dim, filternetOut,
                                         n_hidden_layers=model_config['n_hidden_layers'],
                                         n_hidden_layer_units=model_config['n_hidden_layer_units'],
                                         use_dropout=model_config['policy_use_dropout'],
                                         dropout_perc=model_config['policy_dropout_perc'],
                                         nonlinear=model_config['policy_nonlinear'])
        self.stopNet = createNet(self.hidden_state_dim, 1,
                                 n_hidden_layers=model_config['n_hidden_layers'],
                                 n_hidden_layer_units=model_config['n_hidden_layer_units'],
                                 use_dropout=model_config['policy_use_dropout'],
                                 dropout_perc=model_config['policy_dropout_perc'],
                                 nonlinear=model_config['policy_nonlinear'])

        self.baselineNet = createNet(self.hidden_state_dim, 1,
                                     n_hidden_layers=model_config['baseline_n_hidden_layers'],
                                     n_hidden_layer_units=model_config['baseline_n_hidden_layer_units'],
                                     use_dropout=model_config['baseline_use_dropout'],
                                     dropout_perc=model_config['baseline_dropout_perc'],
                                     nonlinear=model_config['baseline_nonlinear']
                                     )

        self.initialize_network_weights(model_config)

    def initialize_network_weights(self, model_config):
        for l in self.filterPolicyNet:
            if l._get_name() == 'Linear':
                torch.nn.init.xavier_normal_(l.weight, gain=torch.nn.init.calculate_gain(
                    model_config['policy_nonlinear'].__name__.lower()))
                torch.nn.init.constant_(l.bias, 0.01)
        for l in self.stopNet:
            if l._get_name() == 'Linear':
                torch.nn.init.xavier_normal_(l.weight, gain=torch.nn.init.calculate_gain(
                    model_config['policy_nonlinear'].__name__.lower()))
                torch.nn.init.constant_(l.bias, 0.01)
        for l in self.baselineNet:
            if l._get_name() == 'Linear':
                torch.nn.init.xavier_normal_(l.weight, gain=torch.nn.init.calculate_gain(
                    model_config['baseline_nonlinear'].__name__.lower()))
                torch.nn.init.constant_(l.bias, 0.01)

    def freeze_agents(self):
        for param in self.filterPolicyNet.parameters():
            param.requires_grad = False
        for param in self.baselineNet.parameters():
            param.requires_grad = False

    def freeze_stopNet(self):
        for param in self.stopNet.parameters():
            param.requires_grad = False

    def unfreeze_stopNet(self):
        for param in self.stopNet.parameters():
            param.requires_grad = True

    def unfreeze_agents(self):
        for param in self.filterPolicyNet.parameters():
            param.requires_grad = True
        for param in self.baselineNet.parameters():
            param.requires_grad = True

    def initLoggers(self):
        """
        Initialize auxiiliary lists to calculate rewards at end of processing
        """

        self.stop_actions = []
        self.actions = []

        ## Log of selected action for filtering
        self.log_pi_filter = []

        ## Baseline values for states
        self.baselines = []

        ##List of filter statuses
        self.filter_statuses = []

        ##List of predictions at each chekpoint, used for stop decision training
        self.checkpoint_logits = []

        ## Used for calculation of earliness reward
        self.cumsum_filter_statuses = None

    def forward(self, x):
        """
        Return stop and filtering decisions of network
        """
        filter_action, log_pi_filter = self.filterPolicy(x)
        self.log_pi_filter.append(log_pi_filter)

        stop_action = self.stopNetwork(x.detach())

        self.stop_actions.append(stop_action)
        stop_action = (stop_action >= self.stop_action_thresh).float()

        b = self.baselineNet(x.detach()).squeeze()
        self.baselines.append(b)

        return stop_action, filter_action

    def filterPolicy(self, x):

        filterOut = torch.sigmoid(self.filterPolicyNet(x))
        distribution = Bernoulli(probs=filterOut)
        filter_action = distribution.sample()
        log_pi = distribution.log_prob(filter_action.detach())
        self.actions.append(filter_action)

        return filter_action, log_pi

    def stopNetwork(self, x):
        """
        Return the stop decision of the network
        """

        ## We assign 1 to stop, 0 to continue

        action = torch.sigmoid(self.stopNet(x)).squeeze()

        return action

    def getRewards(self, logits, labels):
        """
        Calculate the rewards for the filtering and stopping policies
        """
        y_hat = torch.softmax(logits.detach(), dim=1)
        y_hat = torch.max(y_hat, 1)[1]

        MinFilterSum = self.n_channel_slices
        MaxFilterSum = (1 + self.num_checkpoints) * self.n_channel_slices

        earl_reward = mapValue(MinFilterSum, MaxFilterSum, 1, 0, self.cumsum_filter_statuses)

        acc_reward = (2 * (
                y_hat.float().round() == labels.squeeze().float()).float() - 1)

        # Calculate final reward based on earliness and accuracy for filter decisions
        filter_reward = acc_reward + earl_reward
        filter_reward = torch.where(acc_reward == -1, -2 * torch.ones_like(acc_reward), filter_reward)
        filter_reward = filter_reward.unsqueeze(1)

        ## Calculate reward for stop decisions
        y_hat_stop = torch.stack(self.checkpoint_logits).detach().transpose(0, 1).softmax(dim=-1).max(-1)[
            1]

        stop_acc_reward = (2 * (y_hat_stop.float().round() == labels.unsqueeze(1).float()) - 1)

        stop_earl_reward = mapValue(MinFilterSum, MaxFilterSum, 1, 0,
                                    torch.stack(self.filter_statuses).transpose(0, 1).sum(-1).cumsum(-1))

        stop_reward = stop_acc_reward + stop_earl_reward
        stop_reward = torch.where(stop_acc_reward == -1, -2 * torch.ones_like(stop_acc_reward), stop_reward)

        return filter_reward.detach(), stop_reward.detach()

    def discount_rewards(self, rewards, gamma=0.99):
        """
        Discount the policy rewards with given gamma
        """

        rewards = rewards * (gamma ** torch.arange(0, rewards.shape[-1]))

        rewards = torch.flip(torch.cumsum(torch.flip(rewards, (-1,)), -1), (-1,))
        rewards = rewards / (gamma ** torch.arange(0, rewards.shape[-1]))
        return rewards

    def computeLoss(self, logits, labels):
        """
        Calculate loss using REINFORCE algorithm
        """

        ## We skip the last step since the agent cannot decide after last checkpoint, but the last slice log_pi has been added to the list
        ## (for simplicity of implementation)
        log_pi_filter = torch.stack(self.log_pi_filter).transpose(0, 1)[:, :-1]
        baselines = torch.stack(self.baselines).transpose(0, 1)[:, :-1]

        R_filter, R_stop = self.getRewards(logits, labels)

        R_filter = R_filter.repeat((1, baselines.shape[-1]))

        if self.discount:
            R_filter = self.discount_rewards(R_filter)

        adjusted_rewards = R_filter - baselines.detach() # Baseline values are subtracted from achieved rewards

        ## Baseline loss, to train baseline estimation network
        loss_b = F.mse_loss(baselines,
                            R_filter)

        ## If reward at any checkpoint is higher than future rewards, it means agent should stop
        ## The following code implements this
        stop_actions = torch.stack(self.stop_actions).transpose(0, 1)[:, :-1]
        stop_target = torch.zeros_like(stop_actions)

        for i in range(self.num_checkpoints):
            stop_target[:, i] = (R_stop[:, i:i + 1] > R_stop[:, i + 1:]).all(-1)
        ## Loss for stopping network
        loss_stop = F.binary_cross_entropy(stop_actions.flatten(), stop_target.flatten(), reduction='sum')
        loss_filter = (-log_pi_filter * adjusted_rewards.unsqueeze(-1)).sum()

        return loss_stop, loss_filter, loss_b, R_filter.detach().sum(
            -1).mean()
