"""
Student Agent Implementation for River and Stones Game

This file contains the essential utilities and template for implementing your AI agent.
Your task is to complete the StudentAgent class with intelligent move selection.

Game Rules:
- Goal: Get 4 of your stones into the opponent's scoring area
- Pieces can be stones or rivers (horizontal/vertical orientation)  
- Actions: move, push, flip (stone↔river), rotate (river orientation)
- Rivers enable flow-based movement across the board

Your Task:
Implement the choose() method in the StudentAgent class to select optimal moves.
You may add any helper methods and modify the evaluation function as needed.
"""
from student_agent import *
import math
import time
import random
import copy

# ---------- lightweight helpers for rollouts ----------

def is_terminal_fast(board, rows, cols, score_cols):
    """Return 'circle' or 'square' if either has 4 stones scored, else None."""
    for side in ("circle", "square"):
        if count_stones_in_scoring_area(board, side, rows, cols, score_cols) >= 4:
            return side
    return None

def distance_to_goal_rowsum(board, player, rows, cols, score_cols):
    """Sum of row distances for player's stones to their goal row (smaller is better)."""
    goal_row = bottom_score_row(rows) if player == "square" else top_score_row()
    s = 0
    for y in range(rows):
        for x in range(cols):
            p = board[y][x]
            if p and p.owner == player and getattr(p, "side", "stone") == "stone":
                s += abs(goal_row - y)
    return s

def softmax(xs):
    m = max(xs) if xs else 0.0
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es] if z else [1.0 / len(xs)] * len(xs) if xs else []

class _Policy:
    """
    Cheap biased playout policy:
      - prefer moves that advance toward goal in scoring columns
      - small nudge for pushes and river flips/rotations
    """
    def __init__(self, rows, cols, score_cols):
        self.rows = rows
        self.cols = cols
        self.score_set = set(score_cols)

    def _forward_bonus(self, move, player):
        if move["action"] == "move":
            fx, fy = move["from"]; tx, ty = move["to"]
            goal_row = bottom_score_row(self.rows) if player == "square" else top_score_row()
            before = abs(goal_row - fy); after = abs(goal_row - ty)
            b = 0.0
            if (tx in self.score_set) and (after < before): b += 1.0
            if  after < before: b += 0.4
            return b
        if move["action"] == "push":
            tx, ty = move["to"]
            goal_row = bottom_score_row(self.rows) if player == "square" else top_score_row()
            if abs(goal_row - ty) <= 2 and (tx in self.score_set):
                return 0.8
            return 0.3
        if move["action"] in ("flip", "rotate"):
            return 0.15
        return 0.0

    def pick(self, board, moves, player):
        if not moves: return None
        scores = [self._forward_bonus(m, player) for m in moves]
        probs = softmax(scores)
        r, acc = random.random(), 0.0
        for m, p in zip(moves, probs):
            acc += p
            if r <= acc: return m
        return moves[-1]

# ---------- Monte Carlo StudentAgent (drop-in replacement) ----------

class ManInBlack(BaseAgent):
    def __init__(self, player: str):
        super().__init__(player)
        # rollout parameters safe for laptops
        self.rollout_depth_cap = 12          # max plies per rollout
        self.samples_per_move_cap = 64       # hard cap per candidate move
        self.min_samples_per_move = 8        # first-pass samples per move
        self.random_seed = None              # set to int for reproducibility

    def _rollout_once(self, board, to_move, rows, cols, score_cols, depth_cap, deadline) -> float:
        """Single playout; returns score from self.player perspective."""
        if self.random_seed is not None:
            random.seed(self.random_seed + int(time.time()))

        policy = _Policy(rows, cols, score_cols)
        cur = copy.deepcopy(board)
        player = to_move

        for _ in range(depth_cap):
            if time.perf_counter() > deadline:
                break

            term = is_terminal_fast(cur, rows, cols, score_cols)
            if term is not None:
                return 10_000.0 if term == self.player else -10_000.0

            moves = generate_all_moves(cur, player, rows, cols, score_cols)
            if not moves:
                return basic_evaluate_board(cur, self.player, rows, cols, score_cols)

            move = policy.pick(cur, moves, player)
            if move is None:
                return basic_evaluate_board(cur, self.player, rows, cols, score_cols)

            ok, nxt = simulate_move(cur, move, player, rows, cols, score_cols)
            if not ok or nxt is None:
                # if engine fallback prevents progress, evaluate current
                return basic_evaluate_board(cur, self.player, rows, cols, score_cols)

            cur = nxt
            player = get_opponent(player)

        # non-terminal cutoff: blend eval with distance differential
        val = basic_evaluate_board(cur, self.player, rows, cols, score_cols)
        myd  = distance_to_goal_rowsum(cur, self.player,   rows, cols, score_cols)
        oppd = distance_to_goal_rowsum(cur, self.opponent, rows, cols, score_cols)
        return val + 0.5 * (oppd - myd)

    def _monte_carlo_choose(self, board, rows, cols, score_cols, current_player_time, opponent_time):
        """Rollout-based move selection under same time budget policy."""
        moves = generate_all_moves(board, self.player, rows, cols, score_cols)
        if not moves:
            return None

        # reuse adaptive time policy from your minimal agent
        base_ms = 200
        if opponent_time < 3.0 or current_player_time < 3.0:
            base_ms = 50
        if current_player_time <= 0:
            return moves[0]
        deadline = time.perf_counter() + base_ms / 1000.0

        # pre-rank moves by 1-ply static eval to focus rollouts
        ranked = []
        for m in moves:
            if time.perf_counter() > deadline: break
            ok, nb = simulate_move(board, m, self.player, rows, cols, score_cols)
            v = basic_evaluate_board(nb, self.player, rows, cols, score_cols) if ok and nb is not None else -1e9
            ranked.append((v, m))
        ranked.sort(reverse=True, key=lambda t: t[0])

        # stats per move
        stats = {id(m): {"sum": 0.0, "n": 0, "move": m} for _, m in ranked}
        best_move = ranked[0][1]
        best_mean = -float('inf')

        # first-pass samples
        for _, move in ranked:
            mid = id(move)
            while stats[mid]["n"] < self.min_samples_per_move:
                if time.perf_counter() > deadline: break
                ok, nb = simulate_move(board, move, self.player, rows, cols, score_cols)
                if not ok or nb is None: break
                val = self._rollout_once(
                    nb, to_move=self.opponent,
                    rows=rows, cols=cols, score_cols=score_cols,
                    depth_cap=self.rollout_depth_cap, deadline=deadline
                )
                stats[mid]["sum"] += val
                stats[mid]["n"] += 1

            if stats[mid]["n"] > 0:
                mean = stats[mid]["sum"] / stats[mid]["n"]
                if mean > best_mean:
                    best_mean, best_move = mean, move

            if time.perf_counter() > deadline:
                break

        # round-robin extra samples up to cap while time remains
        i = 0
        while time.perf_counter() <= deadline and ranked:
            _, move = ranked[i % len(ranked)]
            mid = id(move)
            if stats[mid]["n"] >= self.samples_per_move_cap:
                i += 1
                if all(stats[id(m)]["n"] >= self.samples_per_move_cap for _, m in ranked):
                    break
                continue

            ok, nb = simulate_move(board, move, self.player, rows, cols, score_cols)
            if not ok or nb is None:
                i += 1
                continue

            val = self._rollout_once(
                nb, to_move=self.opponent,
                rows=rows, cols=cols, score_cols=score_cols,
                depth_cap=self.rollout_depth_cap, deadline=deadline
            )
            s = stats[mid]
            s["sum"] += val; s["n"] += 1
            mean = s["sum"] / s["n"]
            if mean > best_mean:
                best_mean, best_move = mean, move
            i += 1

        return best_move

    def choose(self, board, rows, cols, score_cols, current_player_time, opponent_time):
        """Monte Carlo move selection with graceful static fallback."""
        # if engine is missing, fall back to 1-ply static
        try:
            from gameEngine import validate_and_apply_move  # noqa: F401
            engine_ok = True
        except Exception:
            engine_ok = False

        if not engine_ok:
            moves = generate_all_moves(board, self.player, rows, cols, score_cols)
            if not moves: return None
            best_v, best_m = -float('inf'), moves[0]
            for m in moves:
                ok, nb = simulate_move(board, m, self.player, rows, cols, score_cols)
                if ok and nb is not None:
                    v = basic_evaluate_board(nb, self.player, rows, cols, score_cols)
                    if v > best_v: best_v, best_m = v, m
            return best_m

        return self._monte_carlo_choose(board, rows, cols, score_cols, current_player_time, opponent_time)