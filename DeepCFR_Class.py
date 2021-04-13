class DeepCFR(object):
    def __init__(self, main_path, environment_args, mode):

        # initiate (create paths, private variables)

        # set up environment for game mode (environment_args is passed to the dealer etc)

        # mode: flop only, hole cards only, flop + turn, full poker

        # set path variables (self.memory_p1_path etc.)

        # create initial models and save them to a folder

        #

        pass

    def train(self, CFR_iterations, traverse_iterations):

        for t in range(CFR_iterations):
            # re-register newly trained agents (TODO)

            for p in range(num_players):
                for k in range(num_traversals):
                    # collect data from env via external sampling
                    env = create_env(...)
                    obs = env.reset()
                    self.traverse(env, obs, history, p, t)

                # initialize new value network (if not first iteration) and train with val_mem_p
                # (only used for prediction of the next regret values)
                train_val_net()

                ### (TODO): re-register trained agent

            # train the strat_net with strat_mem
            train_strat_net()

        # load strategy network
        return strategy_network


    def traverse(self, env, obs, history, p, t):
        """
        Does one full traversal with recursions.


        Parameters
        ----------
        env : clubs_gym env
            Has 2 agents in its dealer.
            Each agent has a value network (val_net_p1, val_net_p2)

        history : list
            Betting history [10, 10, 0, 0, 23]
            Each value indicated the amount of money played in one turn.
            0 indicates a check or a fold. 1 - inf indicates a check or a call
            (can be deduced from previous entry).

        traverser : int()
            Heads up -> 0 or 1

        val_net_p1 :

        val_net_p2 :

        val_mem_traverser :

        strat_mem :

        CFR_iteration :

        """
        pass
