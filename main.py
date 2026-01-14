import multiprocessing as mp
import os
import warnings
import numpy as np
import gymnasium as gym
from dataclasses import dataclass

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")


class LegContactBonus(gym.RewardWrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        leg1, leg2 = obs[6], obs[7]
        if leg1 > 0.5 and leg2 > 0.5 and obs[1] > 0.1:
            reward += 0.5
        return obs, reward, terminated, truncated, info


class AccelPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, max_acc=5.0, penalty_weight=1.0, kill_on_violation=False, dt=1.0):
        super().__init__(env)
        self.max_acc = max_acc
        self.penalty_weight = penalty_weight
        self.kill_on_violation = kill_on_violation
        self.dt = dt
        self.prev_vx = None
        self.prev_vy = None
        self.reward_range = (-np.inf, np.inf)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        vx = obs[2]
        vy = obs[3]
        self.prev_vx = vx
        self.prev_vy = vy
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vx = obs[2]
        vy = obs[3]

        ax = (vx - self.prev_vx) / self.dt
        ay = (vy - self.prev_vy) / self.dt
        a = np.sqrt(ax ** 2 + ay ** 2)

        self.prev_vx = vx
        self.prev_vy = vy

        if a > self.max_acc:
            penalty = self.penalty_weight * (a - self.max_acc)
            reward -= penalty
            if self.kill_on_violation:
                terminated = True
                info["accel_violation"] = True

        info["accel"] = a
        info["accel_vec"] = (ax, ay)

        return obs, reward, terminated, truncated, info


class TouchdownVelocityWrapper(gym.RewardWrapper):
    def __init__(self, env, max_touchdown_speed=1.0, kill_on_violation=False, penalty=-100.0):
        super().__init__(env)
        self.max_touchdown_speed = max_touchdown_speed
        self.kill_on_violation = kill_on_violation
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vx = obs[2]
        vy = obs[3]
        speed = np.sqrt(vx ** 2 + vy ** 2)

        leg1 = obs[6] > 0.5
        leg2 = obs[7] > 0.5
        touchdown = leg1 or leg2

        if touchdown and not info.get("touchdown_checked", False):
            info["touchdown_checked"] = True
            info["touchdown_speed"] = speed

            if speed > self.max_touchdown_speed:
                reward += self.penalty
                info["touchdown_too_fast"] = True
                if self.kill_on_violation:
                    terminated = True

        return obs, reward, terminated, truncated, info


ENV_ID = "LunarLander-v3"
POP_SIZE = 50
N_GENERATIONS = 200
ELITE_FRACTION = 0.1
MUTATION_STD = 0.1
EPISODES_PER_INDIVIDUAL = 5
MAX_STEPS_PER_EPISODE = 1000
TEMP_START = 1.0
TEMP_END = 0.1
TEMP_DECAY = (TEMP_END / TEMP_START) ** (1 / N_GENERATIONS)

INPUT_SIZE = 8
HIDDEN_SIZE = 32
OUTPUT_SIZE = 4


@dataclass
class Individual:
    weights: np.ndarray
    fitness: float = -np.inf


def create_env(render_mode=None):
    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        continuous=False,
        gravity=-3.8,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
    )
    env = AccelPenaltyWrapper(
        env,
        max_acc=1.0,
        penalty_weight=1.0,
        kill_on_violation=False,
        dt=1 / 50.0
    )
    env = LegContactBonus(env)
    return env


def get_param_sizes():
    w1_size = INPUT_SIZE * HIDDEN_SIZE
    b1_size = HIDDEN_SIZE
    w2_size = HIDDEN_SIZE * OUTPUT_SIZE
    b2_size = OUTPUT_SIZE
    total = w1_size + b1_size + w2_size + b2_size
    return (w1_size, b1_size, w2_size, b2_size, total)


W1_SIZE, B1_SIZE, W2_SIZE, B2_SIZE, N_PARAMS = get_param_sizes()


def unpack_weights(theta: np.ndarray):
    assert theta.size == N_PARAMS
    idx = 0
    w1 = theta[idx: idx + W1_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
    idx += W1_SIZE
    b1 = theta[idx: idx + B1_SIZE].reshape(HIDDEN_SIZE)
    idx += B1_SIZE
    w2 = theta[idx: idx + W2_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    idx += W2_SIZE
    b2 = theta[idx: idx + B2_SIZE].reshape(OUTPUT_SIZE)
    return w1, b1, w2, b2


def policy_act(theta: np.ndarray, obs: np.ndarray, temp=1.0, deterministic=False):
    w1, b1, w2, b2 = unpack_weights(theta)
    x = obs.astype(np.float32)
    h = np.maximum(0, x @ w1 + b1)
    logits = h @ w2 + b2

    if deterministic:
        return np.argmax(logits)
    else:
        logits = np.clip(logits, -10*temp, 10*temp)
        exp_logits = np.exp(logits / temp)
        probs = exp_logits / np.sum(exp_logits)
        return np.random.choice(4, p=probs)


def init_population():
    pop = []
    for _ in range(POP_SIZE):
        theta = np.random.randn(N_PARAMS) * 0.1
        pop.append(Individual(weights=theta))
    return pop


def evaluate_individual(individual: Individual, env, temp=1.0):
    rewards = []
    for _ in range(EPISODES_PER_INDIVIDUAL):
        obs, info = env.reset(seed=None)
        done = False
        total_r = 0.0
        steps = 0
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = policy_act(individual.weights, obs, temp=temp)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            done = terminated or truncated
            steps += 1
        rewards.append(total_r)
    return float(np.mean(rewards))


def worker_evaluate(args):
    ind_idx, individual, env_id, current_temp = args
    env = create_env(render_mode=None)
    fitness = evaluate_individual(individual, env, temp=current_temp)
    env.close()
    return ind_idx, fitness


def evaluate_population(population, current_temp):
    n_cores = min(mp.cpu_count(), POP_SIZE)
    args_list = [(i, ind, ENV_ID, current_temp) for i, ind in enumerate(population)]

    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(worker_evaluate, args_list)

    results.sort(key=lambda x: x[0])
    for i, (ind_idx, fitness) in enumerate(results):
        population[i].fitness = fitness


def select_elites(population):
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    n_elite = max(1, int(ELITE_FRACTION * POP_SIZE))
    return population[:n_elite]


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    mask = np.random.rand(N_PARAMS) < 0.5
    child_weights = np.where(mask, parent1.weights, parent2.weights)
    return Individual(weights=child_weights)


def mutate(individual, gen):
    if gen < 50:
        std = 0.15
    elif gen < 150:
        std = 0.08
    else:
        std = 0.03 + 0.1 * np.random.random()
    noise = np.random.randn(N_PARAMS) * std
    individual.weights += noise


def create_next_generation(elites, gen):
    new_population = []
    for elite in elites:
        new_population.append(Individual(weights=elite.weights.copy(), fitness=elite.fitness))

    while len(new_population) < POP_SIZE:
        parents = np.random.choice(elites, size=2, replace=True)
        child = crossover(parents[0], parents[1])
        mutate(child, gen)
        new_population.append(child)
    return new_population


def load_policy(path="best_policy.npy") -> np.ndarray:
    return np.load(path)


def watch_saved_policy(path="best_policy.npy", n_episodes=10):
    theta = load_policy(path)
    env = create_env(render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=None)
        done = False
        total_r = 0.0
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = policy_act(theta, obs, temp=0.37, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            done = terminated or truncated
            steps += 1

        print(f"Ep {ep} | R={total_r:.1f} | Steps={steps}")

    env.close()


def main():
    np.random.seed(0)
    population = init_population()
    print(f"Initialized population with {POP_SIZE} individuals and {N_PARAMS} parameters each.")

    best_overall = None
    current_temp = TEMP_START

    for gen in range(N_GENERATIONS):
        evaluate_population(population, current_temp)

        fitnesses = np.array([ind.fitness for ind in population])
        best_idx = int(np.argmax(fitnesses))
        best_gen = population[best_idx]

        if best_overall is None or best_gen.fitness > best_overall.fitness:
            best_overall = Individual(weights=best_gen.weights.copy(), fitness=best_gen.fitness)

        print(f"Gen {gen:03d} | "
              f"mean: {fitnesses.mean():7.2f} | "
              f"std: {fitnesses.std():6.2f} | "
              f"best: {best_gen.fitness:7.2f} | "
              f"best_overall: {best_overall.fitness:7.2f} | "
              f"... | T={current_temp:.3f}")
        population = create_next_generation(select_elites(population), gen)
        current_temp = max(TEMP_END, current_temp * TEMP_DECAY)

    print("\nTraining finished.")
    print(f"Best overall fitness: {best_overall.fitness:.2f}")

    env = create_env(render_mode="human")
    obs, info = env.reset(seed=42)
    done = False
    total_r = 0.0
    steps = 0

    while not done and steps < MAX_STEPS_PER_EPISODE:
        action = policy_act(best_overall.weights, obs, temp=0.01, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        done = terminated or truncated
        steps += 1

    print(f"Episode reward of best individual: {total_r:.2f}")
    input("Press Enter to close the window...")
    np.save("best_policy.npy", best_overall.weights)
    print("Saved best policy to best_policy.npy")
    env.close()


if __name__ == "__main__":
    mode = "play"
    if mode == "train":
        mp.set_start_method('spawn', force=True)
        main()
    elif mode == "play":
        watch_saved_policy("best_policy.npy")
