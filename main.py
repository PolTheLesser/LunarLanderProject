import gymnasium as gym
import numpy as np
from dataclasses import dataclass

# =============================
# Config
# =============================

ENV_ID = "LunarLander-v3"   # if your gymnasium has only v2, change to "LunarLander-v2"
POP_SIZE = 50               # population size
N_GENERATIONS = 200         # number of generations
ELITE_FRACTION = 0.2        # fraction kept as elites
MUTATION_STD = 0.1          # std-dev of Gaussian mutation
EPISODES_PER_INDIVIDUAL = 3 # fitness = mean reward over these episodes
MAX_STEPS_PER_EPISODE = 1000

INPUT_SIZE = 8   # LunarLander observation dim (x, y, vx, vy, angle, ang_vel, leg1, leg2)
HIDDEN_SIZE = 16
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
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
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


def policy_act(theta: np.ndarray, obs: np.ndarray) -> int:
    """Forward pass through MLP and pick argmax action."""
    w1, b1, w2, b2 = unpack_weights(theta)
    x = obs.astype(np.float32)
    h = np.tanh(x @ w1 + b1)
    logits = h @ w2 + b2
    action = int(np.argmax(logits))
    return action


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


def evaluate_individual(individual: Individual, env: gym.Env) -> float:
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
    individual.fitness = float(np.mean(rewards))
    return individual.fitness


def evaluate_population(population):
    env = create_env(render_mode=None)
    for ind in population:
        evaluate_individual(ind, env)
    env.close()


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
        action = policy_act(best_overall.weights, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        done = terminated or truncated
        steps += 1

    print(f"Episode reward of best individual: {total_r:.2f}")
    env.close()


if __name__ == "__main__":
    main()
