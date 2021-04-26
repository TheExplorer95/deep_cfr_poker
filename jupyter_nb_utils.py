from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np


def plot_results_0(reward_history, num_actions, action_history_p0,
                 action_history_p1, preflop_history_p0, preflop_history_p1,
                 save_path=None, jupyter=True, sub_title='',
                 tick_labels=['fold/check', 'call', '2 Chips', '4 Chips'],
                 player_labels=['Player_0', 'Player_1'],
                 flop_history_p0=None, flop_history_p1=None):
    """
    Creates line plots of the reward history and cumulative reward for player_0
    and histograms of the played actions as well as played preFlop actions.
    """

    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # cumulative payoff
    cum_rew_p_0 = cumulative_reward(reward_history[:, 0])
    cum_rew_p_1 = cumulative_reward(reward_history[:, 1])
    axs[0].plot(range(0, len(cum_rew_p_0)), cum_rew_p_0, label=player_labels[0])
    axs[0].plot(range(0, len(cum_rew_p_1)), cum_rew_p_1, label=player_labels[1])
    axs[0].set_title("Player 0's cumulative Payoff", fontsize=16)
    axs[0].set_xlabel('Game')
    axs[0].set_ylabel('cumulative Payoff (Chips)')

    textstr = f'\u03C3 = {np.std(reward_history[:, 0]):.4f}\nN = {reward_history[:, 0] .flatten().shape[0]}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

    axs[0].legend(loc='upper right')

    # actions
    # 0 indicates a check or a fold, hence difficult to interprete
    bins = np.arange(num_actions+1)-0.5

    axs[1].hist(preflop_history_p0, bins=bins, alpha=.5, label=player_labels[0], density=True)
    axs[1].hist(preflop_history_p1, bins=bins, alpha=.5, label=player_labels[1], density=True)

    axs[1].set_title("Player pre-Flop action profile", fontsize=16)
    axs[1].set_xlabel('Action')
    axs[1].set_ylabel('normalized amount of plays (N)')
    axs[1].xaxis.set(ticks=range(0, num_actions),
                     ticklabels=tick_labels)
    axs[1].legend(loc='best')

    # preflop strategies
    bins = np.arange(num_actions + 1) - 0.5

    axs[2].hist(flop_history_p0, bins=bins, alpha=.5, label=player_labels[0], density=True)
    axs[2].hist(flop_history_p1, bins=bins, alpha=.5, label=player_labels[1], density=True)

    axs[2].set_title("Player Flop action profile", fontsize=16)
    axs[2].set_xlabel('Action')
    axs[2].set_ylabel('normalized amount of plays (N)')
    axs[2].xaxis.set(ticks=range(0, num_actions),
                     ticklabels=tick_labels)
    axs[2].legend(loc='best')

    # make fancy and show
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    # if jupyter:
    #     plt.show()


def plot_results(reward_history, num_actions, action_history_p0,
                 action_history_p1, preflop_history_p0, preflop_history_p1,
                 save_path=None, jupyter=True, sub_title='Results',
                 tick_labels=['fold/check', 'call', '2 Chips', '4 Chips'],
                 player_labels=['Player_0', 'Player_1']):
    """
    Creates line plots of the reward history and cumulative reward for player_0
    and histograms of the played actions as well as played preFlop actions.
    """

    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # # payoff
    # axs[0].plot(range(0, reward_history.shape[0]), reward_history[:,0])
    # axs[0].set_title("Player 0's Payoff for each Game", fontsize=16)
    # axs[0].set_xlabel('Game')
    # axs[0].set_ylabel('Payoff (Chips)')

    # cumulative payoff
    cum_rew_p_0 = cumulative_reward(reward_history[:, 0])
    axs[0].plot(range(0, len(cum_rew_p_0)), cum_rew_p_0)
    axs[0].set_title("Player 0's cumulative Payoff", fontsize=16)
    axs[0].set_xlabel('Game')
    axs[0].set_ylabel('cumulative Payoff (Chips)')

    textstr = f'\u03C3 = {np.std(reward_history[:, 0])}\nN = {reward_history[:, 0] .flatten().shape[0]}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0].text(0.95, 0.95, textstr, transform=axs[0].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    # actions
    # 0 indicates a check or a fold, hence difficult to interprete
    bins = np.arange(num_actions+1)-0.5

    axs[1].hist(action_history_p0, bins=bins, alpha=.5, label=player_labels[0], density=True)
    axs[1].hist(action_history_p1, bins=bins, alpha=.5, label=player_labels[1], density=True)

    axs[1].set_title("Player action profile ", fontsize=16)
    axs[1].set_xlabel('Action')
    axs[1].set_ylabel('normalized amount of plays (N)')
    axs[1].xaxis.set(ticks=range(0, num_actions),
                     ticklabels=tick_labels)
    axs[1].legend(loc='best')

    # preflop strategies
    bins = np.arange(num_actions+1)-0.5

    axs[2].hist(preflop_history_p0, bins=bins, alpha=.5, label=player_labels[0], density=True)
    axs[2].hist(preflop_history_p1, bins=bins, alpha=.5, label=player_labels[1], density=True)

    axs[2].set_title("Player pre-Flop action profile", fontsize=16)
    axs[2].set_xlabel('Action')
    axs[2].set_ylabel('normalized amount of plays (N)')
    axs[2].xaxis.set(ticks=range(0, num_actions),
                     ticklabels=tick_labels)
    axs[2].legend(loc='best')

    # make fancy and show
    plt.suptitle(f'{sub_title}', fontsize=24)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if jupyter:
        plt.show()


def live_plot(reward_history, round):
    """
    Dynamically plots the reward statistics for a jupyter notebook.
    """
    clear_output(wait=True)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    axs[0].plot(range(0, reward_history.shape[0]), reward_history[:, 0])
    axs[0].set_title("Player 0's payoff after game", fontsize=16)
    axs[0].set_xlabel('Game')
    axs[0].set_ylabel('Payoff')

    cum_rew_p_0 = cumulative_reward(reward_history[:, 0])
    axs[1].plot(range(0, len(cum_rew_p_0)), cum_rew_p_0)
    axs[1].set_title("Player 0's cumulative payoff", fontsize=16)
    axs[1].set_xlabel('Game')
    axs[1].set_ylabel('cumulative Payoff')

    plt.suptitle(f'Statistics after Game {round}', fontsize=24)

    plt.tight_layout()
    plt.show()


def cumulative_reward(data):
    """
    Takes a list of numbers and returns their cumulative values as a list.
    """
    for value in data:
        try:
            average.append(average[-1]+value)
        except Exception:
            average = [value]

    return average
