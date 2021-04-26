# Combining Reinforcement Learning, Deep Learning and Game theory approaches in building a poker bot

This is the project repository for the the deep Reinforcement Learning class (WS20/21) at the university of Osnabrück by Mathis Pink and Janosch Bajorath.

## Dependencies
clubs, clubs_gym, environment.yml, requirements.txt

## The Code
We have 2 python scripts and 2 jupyter notbooks that are intended for end user application:
1. main
> The core deep CFR algorithm implemented according to (DEEP CFR REF). Set preferred parameters within the script (preset ones are standart ones used by us) and run it from your terminal. It will save both players trained advantage models uring each CFR iteration

2. test_poker_bot
> A Jupyter Notebook that is used to evaluate pretrained bots, just define your model paths and you are good to go.
> Includes different **rendering modis**: The *Terminal* one outputs the current game into the Notebook terminal, in the *Display* one a browser can be used to render the current game (+ live plot of rewards in terminal) and you can also switch it off.

3. play_against_bot
> With this Notebook you can play against a trained bot, while getting 'assistance' from the bot by receiving information about the choices he would have made in your situation.
> Also includes different rendering modis. 

4. eval_bot
> Python script to evaluate several bots at once. Just set your parameters at the top and you are good to got (predefined parameters also work).

Oother important scripts:
1. deep_CFR_algorithm
> Ray implementation of the deep CFR Algorithm from the paper [DEEP CFR REFERENCE].

2. deep_CFR_model
> The tensorflow models used by the poker agents.

3. poker_agent
> Clubs_gym poker agents.

Utils:
1. utils
2. memory_utils
3. training_utils
4. jupyter_nb_utils
5. eval_utils

Folders:
1. trained_models
> Bla

2. results/eval
> Bla

## Documentaion

Links for the documentaiton.
