#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import math
import warnings
import argparse
import multiprocessing as mp
from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import gymnasium as gym

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")


# -----------------------------
# Reward wrappers (dein Stil)
# -----------------------------

class LegContactBonus(gym.RewardWrapper):
    """Bonus if both legs contact ground while still above ground a bit."""
    def __init__(self, env, bonus=0.5, min_y=0.1):
        super().__init__(env)
        self.bonus = float(bonus)
        self.min_y = float(min_y)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        leg1, leg2 = obs[6], obs[7]  # LunarLander state: legs contact in indices 6/7 [web:77][web:78]
        if leg1 > 0.5 and leg2 > 0.5 and obs[1] > self.min_y:
            reward += self.bonus
        return obs, reward, terminated, truncated, info


class AccelPenaltyWrapper(gym.RewardWrapper):
    """Penalize acceleration magnitude computed from velocity delta / dt."""
    def __init__(self, env, max_acc=1.0, penalty_weight=1.0, kill_on_violation=False, dt=1 / 50.0):
        super().__init__(env)
        self.max_acc = float(max_acc)
        self.penalty_weight = float(penalty_weight)
        self.kill_on_violation = bool(kill_on_violation)
        self.dt = float(dt)
        self.prev_vx = None
        self.prev_vy = None
        self.reward_range = (-np.inf, np.inf)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_vx = float(obs[2])
        self.prev_vy = float(obs[3])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vx, vy = float(obs[2]), float(obs[3])
        ax = (vx - self.prev_vx) / self.dt
        ay = (vy - self.prev_vy) / self.dt
        a = float(np.sqrt(ax * ax + ay * ay))

        self.prev_vx, self.prev_vy = vx, vy

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
    """Penalize if touchdown speed too high (checked once per episode)."""
    def __init__(self, env, max_touchdown_speed=1.0, kill_on_violation=False, penalty=-100.0):
        super().__init__(env)
        self.max_touchdown_speed = float(max_touchdown_speed)
        self.kill_on_violation = bool(kill_on_violation)
        self.penalty = float(penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vx, vy = float(obs[2]), float(obs[3])
        speed = float(np.sqrt(vx * vx + vy * vy))

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


# -----------------------------
# Config / ES structures
# -----------------------------

@dataclass
class ESConfig:
    # env
    env_id: str = "LunarLander-v3"
    continuous: bool = False
    gravity: float = -3.8
    enable_wind: bool = False
    wind_power: float = 15.0
    turbulence_power: float = 1.5

    # wrappers
    wrapper_variant: str = "all"  # baseline | accel_only | all
    max_acc: float = 1.0
    acc_penalty_weight: float = 1.0
    acc_kill: bool = False
    dt: float = 1 / 50.0

    leg_bonus: float = 0.5
    leg_min_y: float = 0.1

    max_touchdown_speed: float = 1.0
    touchdown_penalty: float = -100.0
    touchdown_kill: bool = False

    # ES hyperparams
    pop_size: int = 50
    n_generations: int = 200
    elite_fraction: float = 0.1
    episodes_per_individual: int = 5
    max_steps_per_episode: int = 1000

    # policy network
    input_size: int = 8
    hidden_size: int = 32
    output_size: int = 4

    # action sampling temperature schedule
    temp_start: float = 1.0
    temp_end: float = 0.1

    # mutation schedule (like your code)
    # if you want factorial tuning of mutation, tune "mut_scale" which scales stds
    mut_scale: float = 1.0

    # compute
    n_cores: int = 0  # 0 => auto
    seed: int = 0

    # output
    out_dir: str = "results"
    run_name: str = "run"


@dataclass
class Individual:
    weights: np.ndarray
    fitness: float = -np.inf


# -----------------------------
# Env factory
# -----------------------------

def create_env(cfg: ESConfig, render_mode=None):
    env = gym.make(
        cfg.env_id,
        render_mode=render_mode,
        continuous=cfg.continuous,
        gravity=cfg.gravity,
        enable_wind=cfg.enable_wind,
        wind_power=cfg.wind_power,
        turbulence_power=cfg.turbulence_power,
    )

    if cfg.wrapper_variant in ("accel_only", "all"):
        env = AccelPenaltyWrapper(
            env,
            max_acc=cfg.max_acc,
            penalty_weight=cfg.acc_penalty_weight,
            kill_on_violation=cfg.acc_kill,
            dt=cfg.dt,
        )

    if cfg.wrapper_variant == "all":
        env = LegContactBonus(env, bonus=cfg.leg_bonus, min_y=cfg.leg_min_y)
        env = TouchdownVelocityWrapper(
            env,
            max_touchdown_speed=cfg.max_touchdown_speed,
            kill_on_violation=cfg.touchdown_kill,
            penalty=cfg.touchdown_penalty,
        )

    return env


# -----------------------------
# Policy network utilities
# -----------------------------

def get_param_sizes(cfg: ESConfig):
    w1 = cfg.input_size * cfg.hidden_size
    b1 = cfg.hidden_size
    w2 = cfg.hidden_size * cfg.output_size
    b2 = cfg.output_size
    total = w1 + b1 + w2 + b2
    return w1, b1, w2, b2, total


def unpack_weights(theta: np.ndarray, cfg: ESConfig):
    w1_size, b1_size, w2_size, b2_size, total = get_param_sizes(cfg)
    assert theta.size == total
    idx = 0
    w1 = theta[idx: idx + w1_size].reshape(cfg.input_size, cfg.hidden_size)
    idx += w1_size
    b1 = theta[idx: idx + b1_size].reshape(cfg.hidden_size)
    idx += b1_size
    w2 = theta[idx: idx + w2_size].reshape(cfg.hidden_size, cfg.output_size)
    idx += w2_size
    b2 = theta[idx: idx + b2_size].reshape(cfg.output_size)
    return w1, b1, w2, b2


def policy_act(theta: np.ndarray, obs: np.ndarray, cfg: ESConfig, temp=1.0, deterministic=False):
    w1, b1, w2, b2 = unpack_weights(theta, cfg)
    x = obs.astype(np.float32)
    h = np.maximum(0, x @ w1 + b1)
    logits = h @ w2 + b2

    if deterministic:
        return int(np.argmax(logits))

    # softmax with temperature
    temp = float(max(1e-6, temp))
    clipped = np.clip(logits, -10.0 * temp, 10.0 * temp)
    exp_logits = np.exp(clipped / temp)
    probs = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(cfg.output_size, p=probs))


# -----------------------------
# ES operators
# -----------------------------

def init_population(cfg: ESConfig):
    _, _, _, _, n_params = get_param_sizes(cfg)
    pop = []
    for _ in range(cfg.pop_size):
        theta = np.random.randn(n_params) * 0.1
        pop.append(Individual(weights=theta))
    return pop


def mutation_std_for_gen(cfg: ESConfig, gen: int):
    # your schedule, scaled by mut_scale (factorial-tunable)
    if gen < 50:
        std = 0.15
    elif gen < 150:
        std = 0.08
    else:
        std = 0.03 + 0.1 * np.random.random()
    return float(std * cfg.mut_scale)


def mutate(cfg: ESConfig, individual: Individual, gen: int):
    _, _, _, _, n_params = get_param_sizes(cfg)
    std = mutation_std_for_gen(cfg, gen)
    individual.weights += np.random.randn(n_params) * std


def crossover(cfg: ESConfig, parent1: Individual, parent2: Individual):
    _, _, _, _, n_params = get_param_sizes(cfg)
    mask = np.random.rand(n_params) < 0.5
    child_weights = np.where(mask, parent1.weights, parent2.weights)
    return Individual(weights=child_weights)


def select_elites(cfg: ESConfig, population):
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    n_elite = max(1, int(cfg.elite_fraction * cfg.pop_size))
    return population[:n_elite]


def create_next_generation(cfg: ESConfig, elites, gen: int):
    new_pop = [Individual(weights=e.weights.copy(), fitness=e.fitness) for e in elites]
    while len(new_pop) < cfg.pop_size:
        parents = np.random.choice(elites, size=2, replace=True)
        child = crossover(cfg, parents[0], parents[1])
        mutate(cfg, child, gen)
        new_pop.append(child)
    return new_pop


# -----------------------------
# Evaluation (parallel)
# -----------------------------

def evaluate_individual(cfg: ESConfig, theta: np.ndarray, temp: float, seed_base: int):
    env = create_env(cfg, render_mode=None)
    rewards = []
    for ep in range(cfg.episodes_per_individual):
        obs, info = env.reset(seed=seed_base + ep)
        done = False
        total_r = 0.0
        steps = 0
        while not done and steps < cfg.max_steps_per_episode:
            action = policy_act(theta, obs, cfg, temp=temp, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += float(reward)
            done = bool(terminated or truncated)  # Gymnasium API [web:84]
            steps += 1
        rewards.append(total_r)
    env.close()
    return float(np.mean(rewards))


def _worker_eval(args):
    idx, theta, cfg_dict, temp, seed_base = args
    cfg = ESConfig(**cfg_dict)
    fit = evaluate_individual(cfg, theta, temp, seed_base=seed_base)
    return idx, fit


def evaluate_population(cfg: ESConfig, population, temp: float, gen: int):
    n_cores = cfg.n_cores if cfg.n_cores > 0 else min(mp.cpu_count(), cfg.pop_size)
    cfg_dict = asdict(cfg)

    args = []
    # seed_base per individual (stable within gen)
    for i, ind in enumerate(population):
        args.append((i, ind.weights, cfg_dict, temp, cfg.seed + gen * 100000 + i * 1000))

    if n_cores == 1:
        results = [_worker_eval(a) for a in args]
    else:
        with mp.Pool(processes=n_cores) as pool:
            results = pool.map(_worker_eval, args)

    results.sort(key=lambda x: x[0])
    for i, (_, fit) in enumerate(results):
        population[i].fitness = fit


# -----------------------------
# Temperature schedule
# -----------------------------

def temp_for_gen(cfg: ESConfig, gen: int):
    # exponential decay from start to end
    if cfg.n_generations <= 1:
        return cfg.temp_end
    ratio = cfg.temp_end / cfg.temp_start
    decay = ratio ** (1.0 / (cfg.n_generations - 1))
    t = cfg.temp_start * (decay ** gen)
    return float(max(cfg.temp_end, t))


# -----------------------------
# Logging helpers
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_csv(path, rows, header):
    import csv
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# Single run (train)
# -----------------------------

def run_training(cfg: ESConfig):
    np.random.seed(cfg.seed)

    ensure_dir(cfg.out_dir)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    ensure_dir(run_dir)

    # save config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    pop = init_population(cfg)
    best_overall = None

    t0 = time.time()
    history_rows = []
    for gen in range(cfg.n_generations):
        temp = temp_for_gen(cfg, gen)
        evaluate_population(cfg, pop, temp=temp, gen=gen)

        fits = np.array([ind.fitness for ind in pop], dtype=np.float64)
        best_idx = int(np.argmax(fits))
        best_gen = pop[best_idx]

        if best_overall is None or best_gen.fitness > best_overall.fitness:
            best_overall = Individual(weights=best_gen.weights.copy(), fitness=float(best_gen.fitness))

        row = {
            "gen": gen,
            "temp": temp,
            "mean": float(fits.mean()),
            "std": float(fits.std()),
            "best_gen": float(best_gen.fitness),
            "best_overall": float(best_overall.fitness),
        }
        history_rows.append(row)

        elites = select_elites(cfg, pop)
        pop = create_next_generation(cfg, elites, gen)

    elapsed = time.time() - t0

    # write history
    write_csv(os.path.join(run_dir, "history.csv"), history_rows,
              header=["gen", "temp", "mean", "std", "best_gen", "best_overall"])

    # save best policy
    np.save(os.path.join(run_dir, "best_policy.npy"), best_overall.weights)

    # summary
    summary = {
        "best_overall_fitness": float(best_overall.fitness),
        "elapsed_sec": float(elapsed),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# -----------------------------
# Watch a saved policy
# -----------------------------

def watch_policy(cfg: ESConfig, policy_path: str, n_episodes=10, temp=0.37, deterministic=False, seed=123):
    theta = np.load(policy_path)
    env = create_env(cfg, render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        steps = 0
        while not done and steps < cfg.max_steps_per_episode:
            action = policy_act(theta, obs, cfg, temp=temp, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += float(reward)
            done = bool(terminated or truncated)  # Gymnasium API [web:84]
            steps += 1
        print(f"Ep {ep:02d} | R={total_r:8.2f} | Steps={steps:4d}")
    env.close()


# -----------------------------
# Factorial experiment runner
# -----------------------------

def factorial_grid():
    """
    2^k Levels (edit here).
    Keep it small enough for your compute.
    """
    return {
        "pop_size": [30, 70],
        "elite_fraction": [0.05, 0.15],
        "mut_scale": [0.7, 1.3],          # scales your mutation schedule
        "temp_start": [0.8, 1.2],
        "max_acc": [0.5, 1.5],
        "acc_penalty_weight": [0.5, 1.5],
    }


def run_factorial(cfg_base: ESConfig, replications: int, only_variant: str | None = None):
    ensure_dir(cfg_base.out_dir)
    grid = factorial_grid()
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(product(*values))

    all_rows = []
    exp_csv = os.path.join(cfg_base.out_dir, "factorial_results.csv")
    header = keys + ["rep", "run_name", "seed", "best_overall_fitness", "elapsed_sec", "wrapper_variant"]

    run_counter = 0
    for combo_idx, combo in enumerate(combos):
        combo_dict = dict(zip(keys, combo))
        for rep in range(replications):
            run_counter += 1

            cfg = ESConfig(**asdict(cfg_base))
            for k, v in combo_dict.items():
                setattr(cfg, k, v)

            if only_variant is not None:
                cfg.wrapper_variant = only_variant

            cfg.seed = int(cfg_base.seed + combo_idx * 1000 + rep)
            cfg.run_name = f"combo_{combo_idx:03d}_rep_{rep:02d}"

            print(f"\n[{run_counter:04d}/{len(combos)*replications}] {cfg.run_name} | {combo_dict} | variant={cfg.wrapper_variant}")
            summary = run_training(cfg)

            row = {**combo_dict,
                   "rep": rep,
                   "run_name": cfg.run_name,
                   "seed": cfg.seed,
                   "best_overall_fitness": summary["best_overall_fitness"],
                   "elapsed_sec": summary["elapsed_sec"],
                   "wrapper_variant": cfg.wrapper_variant}
            all_rows.append(row)

            # append incremental
            write_csv(exp_csv, [row], header=header)

    return all_rows


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "watch", "factorial", "ablation_factorial"])
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--run_name", type=str, default="run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_cores", type=int, default=0)

    # quick overrides
    p.add_argument("--pop_size", type=int, default=50)
    p.add_argument("--n_generations", type=int, default=200)
    p.add_argument("--elite_fraction", type=float, default=0.1)
    p.add_argument("--episodes_per_individual", type=int, default=5)
    p.add_argument("--temp_start", type=float, default=1.0)
    p.add_argument("--temp_end", type=float, default=0.1)
    p.add_argument("--mut_scale", type=float, default=1.0)

    p.add_argument("--wrapper_variant", type=str, default="all", choices=["baseline", "accel_only", "all"])

    # factorial
    p.add_argument("--replications", type=int, default=3)

    # watch
    p.add_argument("--policy_path", type=str, default="results/run/best_policy.npy")
    p.add_argument("--watch_episodes", type=int, default=10)
    p.add_argument("--watch_temp", type=float, default=0.37)
    p.add_argument("--watch_deterministic", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = ESConfig(
        out_dir=args.out_dir,
        run_name=args.run_name,
        seed=args.seed,
        n_cores=args.n_cores,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        elite_fraction=args.elite_fraction,
        episodes_per_individual=args.episodes_per_individual,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        mut_scale=args.mut_scale,
        wrapper_variant=args.wrapper_variant,
    )

    # safer default for multiprocessing on some systems
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.mode == "train":
        summary = run_training(cfg)
        print("\nDONE:", summary)

    elif args.mode == "watch":
        watch_policy(cfg, args.policy_path,
                     n_episodes=args.watch_episodes,
                     temp=args.watch_temp,
                     deterministic=args.watch_deterministic,
                     seed=123)

    elif args.mode == "factorial":
        run_factorial(cfg, replications=args.replications, only_variant=None)

    elif args.mode == "ablation_factorial":
        # runs factorial once per wrapper variant
        for variant in ["baseline", "accel_only", "all"]:
            print(f"\n=== Running factorial for wrapper_variant={variant} ===")
            cfg_v = ESConfig(**asdict(cfg))
            cfg_v.out_dir = os.path.join(cfg.out_dir, f"ablation_{variant}")
            run_factorial(cfg_v, replications=args.replications, only_variant=variant)


if __name__ == "__main__":
    main()
