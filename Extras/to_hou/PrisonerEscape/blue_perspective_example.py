from simulator import PrisonerBlueEnv, PrisonerBothEnv
from fugitive_policies.heuristic import HeuristicPolicy
from blue_policies.heuristic import BlueHeuristic

if __name__ == "__main__":
    
    variation = 0
    terrain_map = f'simulator/forest_coverage/map_set/{variation}.npy'
    mountain_locations = [(400, 300), (1600, 1800)] # original mountain setup

    camera_configuration="simulator/camera_locations/original_and_more.txt"
    observation_step_type="Blue" 
    step_reset=True 
    terrain=None
    env = PrisonerBothEnv(terrain=terrain,
                        spawn_mode='corner',
                        observation_step_type=observation_step_type,
                        random_cameras=False,
                        camera_file_path=camera_configuration,
                        mountain_locations=mountain_locations,
                        camera_range_factor=1.0,
                        observation_terrain_feature=False,
                        random_hideout_locations=False,
                        spawn_range=350,
                        helicopter_battery_life=200,
                        helicopter_recharge_time=40,
                        num_search_parties=5,
                        terrain_map=terrain_map,
                        step_reset = step_reset
                        )
    
    epsilon = 0.1
    policy = HeuristicPolicy(env, epsilon=epsilon)
    env = PrisonerBlueEnv(env, policy)
    print(env.observation_space.shape)

    blue_obs = env.reset()

    blue_heuristic = BlueHeuristic(env, debug=False)
    blue_heuristic.reset()

    blue_heuristic.init_behavior()

    i = 0
    done=False
    while not done:
        i += 1

        blue_actions = blue_heuristic.predict(blue_obs)
        blue_obs, reward, done, _ = env.step(blue_actions)
        game_img = env.render('Policy', show=True, fast=True)
