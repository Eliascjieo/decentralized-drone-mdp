
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DT = 0.2
DRONE_RADIUS = 0.3


ACTIONS = {
    0: np.array([0.0, 0.0]),
    1: np.array([1.0, 0.0]),
    2: np.array([-1.0, 0.0]),
    3: np.array([0.0, 1.0]),
    4: np.array([0.0, -1.0]),
    5: np.array([1.0, 1.0]),
    6: np.array([1.0, -1.0]),
    7: np.array([-1.0, 1.0]),
    8: np.array([-1.0, -1.0])
}


class Drone:

    def __init__(self, position, goal, priority):

        self.position = np.array(
            position,
            dtype=float
        )

        self.goal = np.array(
            goal,
            dtype=float
        )

        self.priority = priority

        self.velocity = np.array(
            [0.0, 0.0]
        )

    def distance_to_goal(self):

        return np.linalg.norm(
            self.goal -
            self.position
        )



def check_collision(drone1, drone2):

    distance = np.linalg.norm(
        drone1.position -
        drone2.position
    )

    return distance < 2 * DRONE_RADIUS



def get_observation(self_drone,other_drone):

    goal_vector = (
        self_drone.goal -
        self_drone.position
    )

    relative_position = (
        other_drone.position -
        self_drone.position
    )

    relative_velocity = (
        other_drone.velocity -
        self_drone.velocity
    )

    distance = np.linalg.norm(
        relative_position
    )

    priority_difference = (
        self_drone.priority -
        other_drone.priority
    )

    observation = np.concatenate([
        goal_vector,
        self_drone.velocity,
        relative_position,
        relative_velocity,
        [distance],
        [self_drone.priority],
        [other_drone.priority],
        [priority_difference]
    ])

    return observation

def discretize_direction(vector):

    angle = np.arctan2(
        vector[1],
        vector[0]
    )

    angle = np.degrees(angle)

    if -22.5 <= angle < 22.5:
        return "right"

    elif 22.5 <= angle < 67.5:
        return "front_right"

    elif 67.5 <= angle < 112.5:
        return "front"

    elif 112.5 <= angle < 157.5:
        return "front_left"

    elif angle >= 157.5 or angle < -157.5:
        return "left"

    elif -157.5 <= angle < -112.5:
        return "back_left"

    elif -112.5 <= angle < -67.5:
        return "back"

    else:
        return "back_right"

def discretize_distance(distance):

    if distance < 0.6:
        return "collision"

    elif distance < 1.5:
        return "critical"

    elif distance < 3.0:
        return "near"

    elif distance < 6.0:
        return "medium"

    else:
        return "far"


def discretize_relative_speed(relative_velocity):

    speed = np.linalg.norm(relative_velocity)

    if speed < 0.5:
        return "slow"

    elif speed < 1.5:
        return "medium"

    else:
        return "fast"


def discretize_closing_speed(relative_position,relative_velocity):

    distance = np.linalg.norm(relative_position)

    if distance == 0:
        return "Distance is 0"

    closing_speed= -np.dot(relative_position,relative_velocity)/distance

    if closing_speed > 1.5:
        return "strongly_approaching"

    elif closing_speed > 0.3:
        return "approaching"

    elif closing_speed > -0.3:
        return "neutral"

    elif closing_speed > -1.5:
        return "separating"

    else:
        return "strongly_separating"


def discretize_priority_difference(delta_priority):

    if delta_priority > 0:
        return "higher"

    elif delta_priority < 0:
        return "lower"

    else:
        return "equal"

def get_goal_direction(drone):

    direction = (
        drone.goal -
        drone.position
    )

    norm = np.linalg.norm(direction)

    if norm == 0:
        return np.array([1.0, 0.0])

    return direction / norm


def discretize_observation(observation):

    goal_vector = observation[0:2]

    velocity = observation[2:4]

    relative_position = observation[4:6]

    relative_velocity = observation[6:8]

    distance = observation[8]

    priority_difference = observation[11]

    goal_direction = discretize_direction(
        goal_vector
    )

    relative_direction = discretize_direction(
        relative_position
    )

    relative_speed = discretize_relative_speed(
        relative_velocity
    )

    closing_speed_category = discretize_closing_speed(
        relative_position,
        relative_velocity
    )

    
    distance_category = discretize_distance(
        distance
    )

    priority_category = discretize_priority_difference(
        priority_difference
    )
    own_speed = discretize_relative_speed(velocity)

    state = (
        goal_direction,
        relative_direction,
        distance_category,
        relative_speed,
        own_speed,
        closing_speed_category,
        priority_category
    )

    return state




class Environment:

    def __init__(self):

        self.drone1 = Drone(
            [-5, 0],
            [5, 0],
            1.0
        )

        self.drone2 = Drone(
            [5, 0],
            [-5, 0],
            1.0
        )

    def reset(self):
        self.drone1.position = np.array([-5.0, 0.0])
        self.drone2.position = np.array([5.0, 0.0])

        self.drone1.velocity = np.array([0.0, 0.0])
        self.drone2.velocity = np.array([0.0, 0.0])

        # Randomize priorities at the start of every episode
        priority_options = [0.0, 0.5, 1.0]
        self.drone1.priority = random.choice(priority_options)
        self.drone2.priority = random.choice(priority_options)

        return self.get_observations()

    def get_observations(self):

        obs1 = get_observation(self.drone1,self.drone2)

        obs2 = get_observation(self.drone2,self.drone1)

        return obs1, obs2

    def step(self, action1, action2):
        old_dist1 = self.drone1.distance_to_goal()
        old_dist2 = self.drone2.distance_to_goal()

        self.drone1.velocity = ACTIONS[action1]
        self.drone2.velocity = ACTIONS[action2]

        self.drone1.position += self.drone1.velocity * DT
        self.drone2.position += self.drone2.velocity * DT

        collision = check_collision(self.drone1, self.drone2)
        goal1 = self.drone1.distance_to_goal() < DRONE_RADIUS
        goal2 = self.drone2.distance_to_goal() < DRONE_RADIUS
        done = collision or (goal1 and goal2)
        
        new_dist1 = self.drone1.distance_to_goal()
        new_dist2 = self.drone2.distance_to_goal()

        # ---------------- SIMPLE PRIORITY LOGIC ----------------
        WARNING_DISTANCE = 2.0
        dist_between = np.linalg.norm(self.drone1.position - self.drone2.position)

        # Base progress reward
        reward1 = (old_dist1 - new_dist1) * 10.0
        reward2 = (old_dist2 - new_dist2) * 10.0

        # If they get too close, punish the low-priority drone heavily
        if dist_between < WARNING_DISTANCE:
            reward1 -= 10.0 * (1.0 - self.drone1.priority)
            reward2 -= 10.0 * (1.0 - self.drone2.priority)

        # Terminal states
        if collision:
            reward1 -= 100.0
            reward2 -= 100.0
        
        if goal1:
            reward1 += 100.0
        if goal2:
            reward2 += 100.0
        # -------------------------------------------------------

        observations = self.get_observations()
        info = {
            "collision": collision,
            "goal1": goal1,
            "goal2": goal2
        }

        return observations, (reward1, reward2), done, info

class Agent:
    def __init__(self, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        
        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        return self.q_table[state]

    def choose_action(self, state):
        # Epsilon-greedy: Explore randomly or exploit the Q-table
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(ACTIONS.keys()))
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state):
        current_q = self.get_q_values(state)[action]
        next_q = np.max(self.get_q_values(next_state))
        
        new_q = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        # Reduce exploration over time as the agent learns
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


env = Environment()

# Instantiate two independent brains
agent1 = Agent()
agent2 = Agent()

NUM_EPISODES = 2000
MAX_STEPS = 200  # Prevent infinite loops if they spin in circles

print("Starting training...")

for episode in range(NUM_EPISODES):
    obs1, obs2 = env.reset()
    state1 = discretize_observation(obs1)
    state2 = discretize_observation(obs2)
    
    total_reward1 = 0
    total_reward2 = 0

    for step_num in range(MAX_STEPS):
        # 1. Agents choose actions
        action1 = agent1.choose_action(state1)
        action2 = agent2.choose_action(state2)

        # 2. Environment steps forward
        (next_obs1, next_obs2), (reward1, reward2), done, info = env.step(action1, action2)

        # 3. Discretize new states
        next_state1 = discretize_observation(next_obs1)
        next_state2 = discretize_observation(next_obs2)

        # 4. Agents learn from the consequences
        agent1.update(state1, action1, reward1, next_state1)
        agent2.update(state2, action2, reward2, next_state2)

        state1 = next_state1
        state2 = next_state2
        total_reward1 += reward1
        total_reward2 += reward2

        if done:
            break
            
    # Decay exploration rate at the end of each episode
    agent1.decay_epsilon()
    agent2.decay_epsilon()

    # Print progress every 100 episodes
    if episode % 100 == 0:
        print(f"Episode: {episode} | Epsilon: {agent1.epsilon:.2f} | "
              f"Reward1: {total_reward1:.1f} | Reward2: {total_reward2:.1f} | "
              f"Collision: {info['collision']}")

print("Training finished!")


def test_and_visualize(env, agent1, agent2, force_p1=0,force_p2=1):
    print("Starting visualization test...")
    
    # 1. Reset environment and capture random priorities
    env.reset()

    if force_p1 is not None:
        env.drone1.priority = force_p1
    if force_p2 is not None:
        env.drone2.priority = force_p2

    obs1, obs2 = env.get_observations()
    state1 = discretize_observation(obs1)
    state2 = discretize_observation(obs2)


    
    p1 = env.drone1.priority
    p2 = env.drone2.priority
    
    # 2. Track histories for plotting
    history_drone1 = [env.drone1.position.copy()]
    history_drone2 = [env.drone2.position.copy()]
    
    done = False
    steps = 0
    
    # 3. Run one episode using 100% exploitation (no random moves)
    while not done and steps < 200:
        action1 = np.argmax(agent1.get_q_values(state1))
        action2 = np.argmax(agent2.get_q_values(state2))
        
        (obs1, obs2), _, done, info = env.step(action1, action2)
        
        state1 = discretize_observation(obs1)
        state2 = discretize_observation(obs2)
        
        history_drone1.append(env.drone1.position.copy())
        history_drone2.append(env.drone2.position.copy())
        steps += 1

    print(f"Test finished in {steps} steps. Collision: {info['collision']}")

    # 4. Set up the Matplotlib plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-7, 7)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Title showing the randomized priorities
    ax.set_title(f"Learned Trajectories\nDrone 1 (Blue) Priority: {p1} | Drone 2 (Red) Priority: {p2}")
    
    # Plot Goal locations
    ax.plot(env.drone1.goal[0], env.drone1.goal[1], 'bX', markersize=12, label='Goal 1')
    ax.plot(env.drone2.goal[0], env.drone2.goal[1], 'rX', markersize=12, label='Goal 2')
    
    # Initialize Drone markers and trajectory lines
    drone1_dot, = ax.plot([], [], 'bo', markersize=10, label='Drone 1')
    drone2_dot, = ax.plot([], [], 'ro', markersize=10, label='Drone 2')
    line1, = ax.plot([], [], 'b-', alpha=0.4, linewidth=2)
    line2, = ax.plot([], [], 'r-', alpha=0.4, linewidth=2)
    
    ax.legend(loc='upper right')
    
    # 5. Animation update function
    def update(frame):
        # Update current positions
        d1_pos = history_drone1[frame]
        d2_pos = history_drone2[frame]
        drone1_dot.set_data([d1_pos[0]], [d1_pos[1]])
        drone2_dot.set_data([d2_pos[0]], [d2_pos[1]])
        
        # Update trailing lines
        line1.set_data([p[0] for p in history_drone1[:frame+1]], 
                       [p[1] for p in history_drone1[:frame+1]])
        line2.set_data([p[0] for p in history_drone2[:frame+1]], 
                       [p[1] for p in history_drone2[:frame+1]])
        
        return drone1_dot, drone2_dot, line1, line2

    # 6. Generate and show the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(history_drone1), 
        interval=50, blit=True, repeat=False
    )
    
    plt.show()

# Call the function to watch them fly
test_and_visualize(env, agent1, agent2)