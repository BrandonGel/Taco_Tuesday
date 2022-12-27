from simulator import PrisonerBlueEnv, PrisonerBothEnv
from fugitive_policies.heuristic import HeuristicPolicy
from blue_policies.heuristic import BlueHeuristic
from simulator.gnn_wrapper import PrisonerGNNEnv
from simulator import initialize_prisoner_environment

if __name__ == "__main__":
    map_num = 0
    seed = 0
    epsilon = 0.1
    observation_step_type = "Blue"
    env = initialize_prisoner_environment(map_num, 
                                        observation_step_type = observation_step_type, 
                                        epsilon=epsilon, seed=seed)
    env = PrisonerGNNEnv(env)

    gnn_obs, blue_obs = env.reset()
    # blue_obs = env._blue_observation
    blue_heuristic = BlueHeuristic(env, debug=False)
    blue_heuristic.reset()
    blue_heuristic.init_behavior()

    i = 0
    done=False
    while not done:
        i += 1

        blue_actions = blue_heuristic.predict(blue_obs)
        gnn_obs, blue_obs, reward, done, _ = env.step(blue_actions)
        # blue_obs = env._blue_observation

        
        # game_img = env.render('Policy', show=True, fast=True)
