import time
import gym
import gym_cap
import gym_cap.heuristic as policy

# Initialize the environment
env = gym.make("cap-v0")

# Reset the environment and select the policies for each of the team
observation = env.reset(
        map_size=20,
        config_path='base_settings.ini',
        # config_path='demo/base_settings.ini',
        policy_blue=policy.Defense(),
        policy_red=policy.Random()
    )

num_match = 100
render = False

rscore = []
start_time = time.time()
for n in range(num_match):
    done = False
    rewards = []
    while not done:
        # Take a step and receive feedback
        observation, reward, done, info = env.step([1,1])
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
