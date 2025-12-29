import gymnasium as gym
import numpy as np
import multiprocessing as mp
from functools import partial
import warnings
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")


class LegContactBonus(gym.RewardWrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        leg1, leg2 = obs[6], obs[7]
        if leg1 > 0.5 and leg2 > 0.5:  # Both legs down
            reward += 2.0  # Small bonus for stable hover
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

        # optional: Info im observation_space unverändert lassen
        self.reward_range = (-np.inf, np.inf)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Geschwindigkeiten aus Observation holen
        vx = obs[2]
        vy = obs[3]
        self.prev_vx = vx
        self.prev_vy = vy
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vx = obs[2]
        vy = obs[3]

        # diskrete Beschleunigung
        ax = (vx - self.prev_vx) / self.dt
        ay = (vy - self.prev_vy) / self.dt
        a = np.sqrt(ax ** 2 + ay ** 2)

        # für nächsten Schritt merken
        self.prev_vx = vx
        self.prev_vy = vy

        # Penalty/Abbruch
        if a > self.max_acc:
            # negative Belohnung proportional zur Überschreitung
            penalty = self.penalty_weight * (a - self.max_acc)
            reward -= penalty

            # optional: Episode beenden, Astronaut „stirbt“
            if self.kill_on_violation:
                terminated = True
                info["accel_violation"] = True

        # Beschleunigung im info mitgeben
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
            # Erstes Mal Bodenkontakt in dieser Episode
            info["touchdown_checked"] = True
            info["touchdown_speed"] = speed

            if speed > self.max_touchdown_speed:
                # zu schnell aufgesetzt
                reward += self.penalty  # z.B. -100 dazu
                info["touchdown_too_fast"] = True
                if self.kill_on_violation:
                    terminated = True

        return obs, reward, terminated, truncated, info


import gymnasium as gym
import numpy as np
from dataclasses import dataclass

# =============================
# Config
# =============================

ENV_ID = "LunarLander-v3"  # if your gymnasium has only v2, change to "LunarLander-v2"
POP_SIZE = 50  # population size
N_GENERATIONS = 200  # number of generations
ELITE_FRACTION = 0.2  # fraction kept as elites
MUTATION_STD = 0.1  # std-dev of Gaussian mutation
EPISODES_PER_INDIVIDUAL = 5  # fitness = mean reward over these episodes
MAX_STEPS_PER_EPISODE = 1000

INPUT_SIZE = 8  # LunarLander observation dim (x, y, vx, vy, angle, ang_vel, leg1, leg2)
HIDDEN1_SIZE = 32
HIDDEN2_SIZE = 16
OUTPUT_SIZE = 4  # 4 discrete actions


# =============================
# Policy / Genome definition
# =============================

@dataclass
class Individual:
    weights: np.ndarray  # flattened parameter vector
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
    # Astronauten-Schutz einbauen
    env = AccelPenaltyWrapper(
        env,
        max_acc=1.0,  # tuning-Parameter
        penalty_weight=1.0,  # tuning-Parameter
        kill_on_violation=False,
        dt=1 / 50.0  # time interval (1 frame)
    )
    return env


def get_param_sizes():
    # Two-layer MLP: input -> hidden -> output
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

    if deterministic:  # NEW: argmax for demos
        return np.argmax(logits)
    else:  # softmax for training
        exp_logits = np.exp(logits / temp)
        probs = exp_logits / np.sum(exp_logits)
        return np.random.choice(4, p=probs)


# =============================
# GA operators
# =============================

def init_population():
    pop = []
    for _ in range(POP_SIZE):
        # small random init around 0
        theta = np.random.randn(N_PARAMS) * 0.1
        pop.append(Individual(weights=theta))
    return pop


def evaluate_individual(individual: Individual, env):  # ADDED THIS
    rewards = []
    for _ in range(EPISODES_PER_INDIVIDUAL):
        obs, info = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = policy_act(individual.weights, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            done = terminated or truncated
            steps += 1
        rewards.append(total_r)
    return float(np.mean(rewards))


def worker_evaluate(args):
    ind_idx, individual, env_id = args
    env = create_env(render_mode=None)
    fitness = evaluate_individual(individual, env)  # Your single-threaded version
    env.close()
    return ind_idx, fitness  # ONLY 2 VALUES


def evaluate_population(population):
    n_cores = min(mp.cpu_count(), POP_SIZE)
    # print(f"Evaluating {len(population)} individuals on {n_cores} cores...")

    args_list = [(i, ind, ENV_ID) for i, ind in enumerate(population)]

    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(worker_evaluate, args_list)

    results.sort(key=lambda x: x[0])
    for i, (ind_idx, fitness) in enumerate(results):  # FIXED: 2 values only
        population[i].fitness = fitness


def select_elites(population):
    # sort by fitness descending
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    n_elite = max(1, int(ELITE_FRACTION * POP_SIZE))
    return population[:n_elite]


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    # uniform crossover
    mask = np.random.rand(N_PARAMS) < 0.5
    child_weights = np.where(mask, parent1.weights, parent2.weights)
    return Individual(weights=child_weights)


def mutate(individual: Individual):
    noise = np.random.randn(N_PARAMS) * MUTATION_STD
    individual.weights = individual.weights + noise


def create_next_generation(elites):
    new_population = []

    # keep elites (elitism)
    for elite in elites:
        # copy to avoid modifying original
        new_population.append(Individual(weights=elite.weights.copy(), fitness=elite.fitness))

    # fill the rest with crossover + mutation
    while len(new_population) < POP_SIZE:
        parents = np.random.choice(elites, size=2, replace=True)
        child = crossover(parents[0], parents[1])
        mutate(child)
        new_population.append(child)

    return new_population


def load_policy(path="best_policy.npy") -> np.ndarray:
    return np.load(path)


def watch_saved_policy(path="best_policy.npy", n_episodes=5):  # More episodes
    theta = load_policy(path)
    env = create_env(render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False;
        total_r = 0.0;
        steps = 0
        accel_violations = 0;
        touchdown_speed = None

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = policy_act(theta, obs, temp=0.01, deterministic=True)  # FIXED
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            done = terminated or truncated
            steps += 1

            # LOG WEAKNESS DETECTION
            if info.get("accel_violation", False):
                accel_violations += 1
            if "touchdown_speed" in info:
                touchdown_speed = info["touchdown_speed"]

        print(f"Ep {ep} | R={total_r:.1f} | Steps={steps} | "
              f"AccelViol={accel_violations} | TD_speed={touchdown_speed:.2f if touchdown_speed else 'N/A'}")

    env.close()


# =============================
# Main GA loop
# =============================

def main():
    np.random.seed(0)

    population = init_population()
    print(f"Initialized population with {POP_SIZE} individuals and {N_PARAMS} parameters each.")

    best_overall = None

    for gen in range(N_GENERATIONS):
        evaluate_population(population)

        # track stats
        fitnesses = np.array([ind.fitness for ind in population])
        best_idx = int(np.argmax(fitnesses))
        best_gen = population[best_idx]

        if best_overall is None or best_gen.fitness > best_overall.fitness:
            best_overall = Individual(weights=best_gen.weights.copy(), fitness=best_gen.fitness)

        print(
            f"Gen {gen:03d} | "
            f"mean: {fitnesses.mean():7.2f} | "
            f"std: {fitnesses.std():6.2f} | "
            f"best: {best_gen.fitness:7.2f} | "
            f"best_overall: {best_overall.fitness:7.2f}"
        )

        population = create_next_generation(select_elites(population))

    print("\nTraining finished.")
    print(f"Best overall fitness: {best_overall.fitness:.2f}")

    # =========================
    # Watch best individual
    # =========================
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

        # hier: Beschleunigung in der Konsole zeigen
        # if "accel" in info:
        # print(f"Step {steps:4d} | a = {info['accel']:.3f}  ax, ay = {info['accel_vec']}")

    print(f"Episode reward of best individual: {total_r:.2f}")
    input("Press Enter to close the window...")
    print("\nTraining finished.")
    print(f"Best overall fitness: {best_overall.fitness:.2f}")

    # Gewichte speichern
    np.save("best_policy.npy", best_overall.weights)
    print("Saved best policy to best_policy.npy")
    env.close()


if __name__ == "__main__":
    mode = "train"
    if mode == "train":
        mp.set_start_method('spawn', force=True)  # Essential for Gym + multiprocessing
        main()
    elif mode == "play":
        watch_saved_policy("best_policy.npy")
