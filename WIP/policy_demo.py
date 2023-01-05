import time
import gym
import argparse
import gym_cap
import gym_cap.heuristic as policy

import numpy as np


description = "Evaluate two different policy."
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--episode', type=int, help='number of episodes to run', default=30)
# parser.add_argument('--blue_policy', type=str, help='blue policy', default='AStar') #AStar policy for Blue
# parser.add_argument('--blue_policy', type=str, help='blue policy', default='Defense') #Defense policy for Blue
# parser.add_argument('--blue_policy', type=str, help='blue policy', default='Patrol') #Patrol policy for Blue
parser.add_argument('--blue_policy', type=str, help='blue policy', default='AStar') #Patrol policy for Blue
parser.add_argument('--red_policy', type=str, help='red policy', default='Invariant')
parser.add_argument('--config_path', type=str, help='configuration path', default='base_settings.ini')
parser.add_argument('--map_size', type=int, help='size of the board', default=20)
parser.add_argument('--time_step', type=int, help='maximum time step', default=100)
parser.add_argument('--fair_map', help='run on fair map', action='store_true')
parser.add_argument('--cores', type=int, help='number of cores (-1 to use all)', default=1)
args = parser.parse_args()

# Initialize the environment
env = gym.make("cap-v0")

# Reset the environment and select the policies for each of the team
blue_policy = getattr(policy, args.blue_policy)()
red_policy = getattr(policy, args.red_policy)()
observation = env.reset(
        map_size=args.map_size,
        policy_blue=blue_policy,
        policy_red=red_policy,
        config_path=args.config_path
    )

agent_num = env.NUM_BLUE + env.NUM_BLUE_UAV
hier_high_obs_dim_each = 3
hier_high_obs_dim = agent_num * hier_high_obs_dim_each
hier_low_obs_dim_each = 3
hier_low_obs_dim = agent_num * hier_low_obs_dim_each

num_match = 1
render = True

rscore = []
start_time = time.time()
for n in range(num_match):
    done = False
    rewards = []
    while not done:
        #you are free to select a random action
        # or generate an action using the policy
        # or select an action manually
        # and the apply the selected action to blue team
        # or use the policy selected and provided in env.reset
        #action = env.action_space.sample()  # choose random action
        #action = policy_blue.gen_action(env.team1,observation,map_only=env.team_home)
        #action = [0, 0, 0, 0]
        #observation, reward, done, info = env.step(action)

        # Take a step and receive feedback
        # Action does not have to be explicitly given if policy is passed during reset.
        # Any provided actions override policy actions.
        # Observation Space = [Board Size X, Board Size Y, 6]
        # Observation Space[:,:,0] = explored spaces (-1: explored, 0: exploring, 1: unexplored)
        # Observation Space[:,:,1] = explored areas (-1: red, 0: NONE, 1: blue)
        # Observation Space[:,:,2] = flag position (-1: red, 0: NONE, 1: blue)
        # Observation Space[:,:,3] = explored obstruction (0: NOT obstruction, 1: obstruction)
        # Observation Space[:,:,4] = UGV position (-1: red, 0: NONE, 1: blue)
        # Observation Space[:,:,5] = UAV position (-1: red, 0: NONE, 1: blue)


        actions = blue_policy.gen_action(env.get_team_blue, observation)
        observation, reward, done, info = env.step(actions)
        rewards.append(reward)

        # Render and sleep (not needed for score analysis)
        if render:
            env.render()
            time.sleep(.05)

    # Reset the game state
    env.reset()

    # Statistics
    rscore.append(sum(rewards))
    duration = time.time() - start_time
    print("Time: %.2f s, Score: %.2f" % (duration, rscore[-1]))

print("Average Time: %.2f s, Average Score: %.2f"
        % (duration/num_match, sum(rscore)/num_match))
# print(rewards)
env.close()

