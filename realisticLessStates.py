import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DT = 0.2
DRONE_RADIUS = 0.3
NUM_DRONES = 4

# These now represent acceleration directions, not instant velocity
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
        self.position = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.priority = priority
        self.velocity = np.array([0.0, 0.0])

    def distance_to_goal(self):
        return np.linalg.norm(self.goal - self.position)

def get_closest_drone(self_drone, all_drones):
    closest = None
    min_dist = float('inf')
    for other in all_drones:
        if other is self_drone: 
            continue
        dist = np.linalg.norm(self_drone.position - other.position)
        if dist < min_dist:
            min_dist = dist
            closest = other
    return closest

# ---------------- UPDATED DISCRETIZATION STATE LOGIC (2,880 States) ----------------
def discretize_direction(vector):
    if np.linalg.norm(vector) == 0:
        return "front"
    angle = np.degrees(np.arctan2(vector[1], vector[0]))
    if -22.5 <= angle < 22.5: return "right"
    elif 22.5 <= angle < 67.5: return "front_right"
    elif 67.5 <= angle < 112.5: return "front"
    elif 112.5 <= angle < 157.5: return "front_left"
    elif angle >= 157.5 or angle < -157.5: return "left"
    elif -157.5 <= angle < -112.5: return "back_left"
    elif -112.5 <= angle < -67.5: return "back"
    else: return "back_right"

def discretize_ttc(relative_position, relative_velocity):
    distance = np.linalg.norm(relative_position)
    if distance == 0: return "critical"
    closing_speed = -np.dot(relative_position, relative_velocity) / distance
    if closing_speed <= 0.1: return "safe"
    ttc = distance / closing_speed
    if ttc < 2.0: return "critical"
    elif ttc < 5.0: return "warning"
    else: return "safe"

def discretize_priority_difference(delta_priority):
    if delta_priority > 0: return "higher"
    elif delta_priority < 0: return "lower"
    else: return "equal"

def discretize_own_velocity(velocity):
    speed = np.linalg.norm(velocity)
    if speed < 0.1: return "hover"
    angle = np.degrees(np.arctan2(velocity[1], velocity[0]))
    if -45 <= angle < 45: return "right"
    elif 45 <= angle < 135: return "front"
    elif -135 <= angle < -45: return "back"
    else: return "left"

def get_observation(self_drone, other_drone):
    return {
        "goal_vector": self_drone.goal - self_drone.position,
        "relative_position": other_drone.position - self_drone.position,
        "relative_velocity": other_drone.velocity - self_drone.velocity,
        "priority_diff": self_drone.priority - other_drone.priority,
        "own_velocity": self_drone.velocity
    }

def discretize_observation(obs):
    return (
        discretize_direction(obs["goal_vector"]),
        discretize_direction(obs["relative_position"]),
        discretize_ttc(obs["relative_position"], obs["relative_velocity"]),
        discretize_priority_difference(obs["priority_diff"]),
        discretize_own_velocity(obs["own_velocity"])
    )
# -------------------------------------------------------------------

class Environment:
    def __init__(self, num_drones):
        self.num_drones = num_drones
        self.drones = [Drone([0,0], [0,0], 1.0) for _ in range(num_drones)]

    def reset(self, randomize_positions=True):
        R = random.uniform(3.0, 6.0) if randomize_positions else 5.0
        base_theta = random.uniform(0, 2 * np.pi) if randomize_positions else 0.0
        priority_options = [0.0, 0.5, 1.0]

        for i, drone in enumerate(self.drones):
            theta = base_theta + i * (2 * np.pi / self.num_drones)
            drone.position = np.array([R * np.cos(theta), R * np.sin(theta)])
            drone.goal = -drone.position.copy()
            drone.velocity = np.array([0.0, 0.0])
            drone.priority = random.choice(priority_options)

        return self.get_observations()

    def get_observations(self):
        obs_list = []
        for drone in self.drones:
            closest = get_closest_drone(drone, self.drones)
            obs = get_observation(drone, closest)
            obs_list.append(discretize_observation(obs))
        return obs_list

    def step(self, actions):
        old_dists = [d.distance_to_goal() for d in self.drones]
        goals_reached = [dist < DRONE_RADIUS for dist in old_dists]

        V_MAX = 0.7
        A_MAX = 2.0

        # Apply acceleration physics
        for i, drone in enumerate(self.drones):
            if goals_reached[i]:
                drone.velocity = np.array([0.0, 0.0])
            else:
                acceleration = ACTIONS[actions[i]] * A_MAX
                drone.velocity += acceleration * DT
                
                speed = np.linalg.norm(drone.velocity)
                if speed > V_MAX:
                    drone.velocity = (drone.velocity / speed) * V_MAX
                    
            drone.position += drone.velocity * DT

        new_dists = [d.distance_to_goal() for d in self.drones]
        
        observations = []
        rewards = []
        collisions = []
        WARNING_DISTANCE = 2.5  # Increased to allow for braking distance

        for i, drone in enumerate(self.drones):
            closest = get_closest_drone(drone, self.drones)
            dist_between = np.linalg.norm(drone.position - closest.position)
            
            collision = dist_between < 2 * DRONE_RADIUS
            collisions.append(collision)

            progress = old_dists[i] - new_dists[i]
            reward = progress * 50.0  
            reward -= 1.0

            # WARNING: Base penalty of -5, plus up to -15 extra for low priority
            if dist_between < WARNING_DISTANCE:
                reward -= 5.0 + 15.0 * (1.0 - drone.priority)

            # COLLISION: Base penalty of -500, plus up to -500 extra for low priority
            if collision:
                reward -= 500.0 + 500.0 * (1.0 - drone.priority)

            if new_dists[i] < DRONE_RADIUS and not goals_reached[i]:
                reward += 1000.0

            rewards.append(reward)
            observations.append(discretize_observation(get_observation(drone, closest)))
            
        done = all([d < DRONE_RADIUS for d in new_dists]) or any(collisions)
        info = {"collision": any(collisions)}

        return observations, rewards, done, info

class Agent:
    def __init__(self, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.05):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        return self.q_table[state]

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(ACTIONS.keys()))
        else:
            return np.argmax(self.get_q_values(state))

    def update(self, state, action, reward, next_state):
        current_q = self.get_q_values(state)[action]
        next_q = np.max(self.get_q_values(next_state))
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ========================================================
# TRAINING LOOP
# ========================================================
env = Environment(NUM_DRONES)
shared_agent = Agent()

NUM_EPISODES = 8000
MAX_STEPS = 200

print(f"Starting training with {NUM_DRONES} drones...")

for episode in range(NUM_EPISODES):
    states = env.reset(randomize_positions=True)
    total_rewards = [0] * NUM_DRONES

    for step_num in range(MAX_STEPS):
        actions = [shared_agent.choose_action(s) for s in states]
        next_states, rewards, done, info = env.step(actions)

        for i in range(NUM_DRONES):
            shared_agent.update(states[i], actions[i], rewards[i], next_states[i])
            total_rewards[i] += rewards[i]

        states = next_states
        if done: break
            
    shared_agent.decay_epsilon()

    if episode % 500 == 0:
        avg_reward = sum(total_rewards) / NUM_DRONES
        print(f"Episode: {episode} | Epsilon: {shared_agent.epsilon:.2f} | "
              f"Avg Reward: {avg_reward:.1f} | Collision: {info['collision']}")

print("Training finished!")

# ========================================================
# VISUALIZATION
# ========================================================
def test_and_visualize(env, agent):
    print("Starting visualization test...")
    states = env.reset(randomize_positions=True)
    histories = [[d.position.copy()] for d in env.drones]
    
    done = False
    steps = 0
    
    while not done and steps < 200:
        actions = [np.argmax(agent.get_q_values(s)) for s in states]
        states, _, done, info = env.step(actions)
        
        for i, d in enumerate(env.drones):
            histories[i].append(d.position.copy())
            
        steps += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f"Learned Trajectories with Acceleration ({env.num_drones} Drones)")
    
    cmap = plt.get_cmap('tab10')
    dots, lines = [], []
    
    for i, drone in enumerate(env.drones):
        c = cmap(i % 10)
        ax.plot(drone.goal[0], drone.goal[1], marker='X', color=c, markersize=12, label=f'Goal {i+1}')
        dot, = ax.plot([], [], marker='o', color=c, markersize=10, label=f'Drone {i+1} (P: {drone.priority})')
        line, = ax.plot([], [], color=c, alpha=0.4, linewidth=2)
        dots.append(dot)
        lines.append(line)
        
    ax.legend(loc='upper right', fontsize='small')
    
    def update(frame):
        for i in range(env.num_drones):
            pos = histories[i][frame]
            dots[i].set_data([pos[0]], [pos[1]])
            x_data = [p[0] for p in histories[i][:frame+1]]
            y_data = [p[1] for p in histories[i][:frame+1]]
            lines[i].set_data(x_data, y_data)
        return dots + lines

    ani = animation.FuncAnimation(
        fig, update, frames=len(histories[0]), 
        interval=50, blit=True, repeat=False
    )
    plt.show()

test_and_visualize(env, shared_agent)