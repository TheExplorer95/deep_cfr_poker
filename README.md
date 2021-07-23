# Combining Reinforcement Learning, Deep Learning and Game Theory Approaches in Building a Poker Bot

This is the repository for a project undertaken by Janosch Bajorath and Mathis Pink for a deep Reinforcement Learning class (20/21) at Osnabrück University.

## Dependencies

There are some important packages, in particular gym, clubs, clubs_gym, ray and tensorflow that you need to have installed on your computer to run the code. The best way to set up the environment is by using conda to create a new environment from the environment.yml file and then install some further requirements from the requirements.txt file via pip.

We use [Clubs Gym](https://github.com/fschlatt/clubs_gym) as the poker game environment, which allows for designing any poker game from Kuhn Poker to Full Table NLHE. Ray is used for multi-threaded sampling during CFR traversals.

If something does not work feel free to contact one of us.

## Content

We provide **2 python scripts** and **2 jupyter notbooks** that are intended to be modified to run experiments with our implementation:

1. main
> Runs the core Deep CFR algorithm implemented according to the [deep CFR paper](https://arxiv.org/abs/1811.00164]). Experiment parameters can be changed in this script (preset to the ones we used). It will train and save both players trained advantage models and loss during each CFR iteration as well as the strategy model at the end of the CFR algorithm. Running the code will create reservoir sampled memory files for the data that is produced during the game traversal.

2. test_poker_bot
> A Jupyter Notebook that is used to evaluate pretrained bots, just define your model paths and you are good to go.
> Includes different **rendering modi**: The *Terminal* one outputs the current game into the Notebook output, in the *Display* one the browser can be used to render the current game (with live plot of rewards) which you can also switch off.

3. play_against_bot
> With this Notebook you can play against a trained bot, while getting 'assistance' from the bot by receiving information about the choices he would have made in your situation.
> Also includes different rendering modi. 

4. eval_bot
> Python script to evaluate several bots at once. Just set your parameters at the top and you are good to got (preset parameters also work).

**Other important scripts for modifying or extending the algorithm**
1. deep_CFR_algorithm
> Ray implementation of the deep CFR Algorithm from the [deep CFR paper](https://arxiv.org/abs/1811.00164]).

2. deep_CFR_model
> The tensorflow models used by the poker agents.

3. poker_agent
> Clubs_gym poker agents.

**Utils:**
1. utils
2. memory_utils
3. training_utils
4. jupyter_nb_utils
5. eval_utils

**Folders:**
1. trained_models
> Contains tensorflow models of pretrained bots.

2. results/eval
> Contains evaluation data of trained advantage and strategy networks. (When train was used even more)

**Documentation**

1. project_report

