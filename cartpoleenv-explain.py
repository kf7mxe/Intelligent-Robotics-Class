
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

total_iterations = 5000
logging_interval = 500
aggressive_right_mode = False


#https://gymnasium.farama.org/environments/classic_control/cart_pole/

# set up the envriroment and record video
env = gym.wrappers.RecordVideo(gym.make('CartPole-v1', render_mode='rgb_array'),
                               video_folder="./videos",
                               name_prefix="non_aggressive_right",
                               episode_trigger=lambda x: x == total_iterations - 1)

## code that makes the decision as to which way to move the cart based on the environment
agent = CartPoleAgent()
learning = []

# how many times to run the simulation
for iteration in range(1, total_iterations+1):
    steps = 0
    done = False
    state = env.reset()

    # the environment state. Which is an array of 4 values. The first value is cart position between -4.8 and 4.8.
    # the second value is cart velocity between -inf and inf. The third value is pole angle between -24 and 24 degrees.
    # the fourth value is pole velocity at tip between -inf and inf.
    state = state[0]

    # reinforcement learning reward and action probabilities
    cumulative_reward = 0
    action_probabilities = []

    # each step in the simulation. If the pole falls over, the simulation is over
    while not done:
        steps += 1
        
        # pass in the state to the agent which will output the action which is 0 or 1 for left or right
        # don't worry about the action_prob. It is used in the reinforcement learning agent.
        action, action_prob = agent.act(state)
        action_probabilities.append(action_prob)

        # Used for one of the reinforcement requirements you can just ignore it
        if aggressive_right_mode:
            if action == 1:
                env.step(action)

        # increment the environment by one state using the action from the agent. Basically increase time by one unit
        # and get the updated environment state, status of the simulation ie is the pole still up and a reward value, which is used in reinforcement learning
        state, reward, done, *_ = env.step(action)

        # cumulative reward for reinforcement learning
        cumulative_reward += reward

    # used for reinforcement learning to calculate the loss or how much to change the neural network weights in reinforcement learning
    policy_loss = [-log_prob * cumulative_reward for log_prob in action_probabilities]
    policy_loss = torch.cat(policy_loss).sum()

    # Reinforcement learning code updates the neural network weights, perform backpropoagation
    agent.optimizer.zero_grad()
    policy_loss.backward()
    agent.optimizer.step()

    # logging 
    learning.append(steps)
    if iteration % logging_interval == 0:
        print("Iteration: {}, Score: {}".format(iteration, steps))
# logging 
x = np.arange(1, len(learning),25)
y = np. add.reduceat(learning,x) / 25

# Logging 
sns.lineplot(x=x, y=y)
plt.xlabel("Episode")
plt.ylabel("Learning Curve / Total Reward")
if aggressive_right_mode:
    plt.title("Aggressive Right")
else:
    plt.title("Non Aggressive Right")
plt.show()
if aggressive_right_mode:
    plt.savefig("Aggressive_right.png")
else:
    plt.savefig("Non_Aggressive_right.png")
