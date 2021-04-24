from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np


def plot_results(reward_history, num_actions, action_history_p0, action_history_p1, preflop_history_p0, preflop_history_p1, save_path=None):
    """
    Creates line plots of the reward history and cumulative reward for player_0
    and histograms of the played actions as well as played preFlop actions.
    """

    plt.style.use('ggplot')
    fig, axs = plt.subplots(4,1, figsize=(12, 15))

    # payoff
    axs[0].plot(range(0, reward_history.shape[0]), reward_history[:,0])
    axs[0].set_title("Player 0's Payoff for each Game", fontsize=16)
    axs[0].set_xlabel('Game')
    axs[0].set_ylabel('Payoff (Chips)')

    # cumulative payoff
    cum_rew_p_0 = cumulative_reward(reward_history[:,0])
    axs[1].plot(range(0, len(cum_rew_p_0)), cum_rew_p_0)
    axs[1].set_title("Player 0's cumulative Payoff", fontsize=16)
    axs[1].set_xlabel('Game')
    axs[1].set_ylabel('cumulative Payoff (Chips)')

    # actions
    # 0 indicates a check or a fold, hence difficult to interprete
    bins = np.arange(num_actions+1)-0.5

    axs[2].hist(action_history_p0, bins=bins, alpha=.5, label='Player_0', density=True)
    axs[2].hist(action_history_p1, bins=bins, alpha=.5, label='Player_1', density=True)

    axs[2].set_title("Player action profiles", fontsize=16)
    axs[2].set_xlabel('Action [0, 6]')
    axs[2].set_ylabel('normalized amount of plays (N)')
    axs[2].legend(loc='upper right')

    # preflop strategies
    bins = np.arange(num_actions+1)-0.5

    axs[3].hist(preflop_history_p0, bins=bins, alpha=.5, label='Player_0', density=True)
    axs[3].hist(preflop_history_p1, bins=bins, alpha=.5, label='Player_1', density=True)

    axs[3].set_title("Player pre-Flop strategies", fontsize=16)
    axs[3].set_xlabel('Action [0, 6]')
    axs[3].set_ylabel('normalized amount of plays (N)')
    axs[3].legend(loc='upper right')

    # make fancy and show
    plt.suptitle('Results', fontsize=24)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

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
