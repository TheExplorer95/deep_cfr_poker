# Combining Reinforcement Learning, Deep Learning and Game theory approaches in building a poker bot

This is Janosch Bajoraths and Mathis Pinks project for the the deep Reinforcement Learning class (WS20/21) at the University of Osnabrück.

## Dependencies

There are some important packages like gym, clubs, clubs_gym, ray and tensorflow that you should have installed on your computer to run the code. The best way to set up your environment is by using conda to create a new environment from the environment.yml file and then install some further requirements from the requirements.txt file via pip.

We use a poker game implementation from the clubs_gym package and ray for multi-threaded sampling during CFR traversals.

If something does not work feel free to contact us!!!

## Content

We have **2 python scripts** and **2 jupyter notbooks** that are intended for **end user application**:
1. main
> The core deep CFR algorithm implemented according to the [deep CFR paper](https://arxiv.org/abs/1811.00164]). Set preferred parameters within the script (preset ones are standard ones used by us) and run it from your terminal. It will train and save both players trained advantage models and loss during each CFR iteration as well as the strategy model at the end of the CFR algorithm. Creates memory files for the data that was produced during the game traversal. 

2. test_poker_bot
> A Jupyter Notebook that is used to evaluate pretrained bots, just define your model paths and you are good to go.
> Includes different **rendering modis**: The *Terminal* one outputs the current game into the Notebook terminal, in the *Display* one a browser can be used to render the current game (+ live plot of rewards in terminal) and you can also switch it off.

3. play_against_bot
> With this Notebook you can play against a trained bot, while getting 'assistance' from the bot by receiving information about the choices he would have made in your situation.
> Also includes different rendering modis. 

4. eval_bot
> Python script to evaluate several bots at once. Just set your parameters at the top and you are good to got (predefined parameters also work).

**Oother important scripts:**
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

**Documentaion**

1. project_report

