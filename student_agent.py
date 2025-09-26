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
import time
import random
import copy
import math
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


# ==================== GAME UTILITIES ====================
# Essential utility functions for game state analysis
# utils.py
from typing import List

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    return 0 <= x < cols and 0 <= y < rows

def score_cols_for(cols: int) -> List[int]:
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))

def top_score_row() -> int:
    return 2

def bottom_score_row(rows: int) -> int:
    return rows - 3

def get_opponent(player: str) -> str:
    return "square" if player == "circle" else "circle"

def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)


# ==================== MOVE GENERATION HELPERS ====================

def get_valid_moves_for_piece(board, x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    Generate all valid moves for a specific piece.
    
    Args:
        board: Current board state
        x, y: Piece position
        player: Current player
        rows, cols: Board dimensions
        score_cols: Scoring column indices
    
    Returns:
        List of valid move dictionaries
    """
    moves = []
    piece = board[y][x]
    
    if piece is None or piece.owner != player:
        return moves
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    if piece.side == "stone":
        # Stone movement
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, rows, cols):
                continue
            
            if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                continue
            
            if board[ny][nx] is None:
                # Simple move
                moves.append({"action": "move", "from": [x, y], "to": [nx, ny]})
            elif board[ny][nx].owner != player:
                # Push move
                px, py = nx + dx, ny + dy
                if (in_bounds(px, py, rows, cols) and 
                    board[py][px] is None and 
                    not is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
                    moves.append({"action": "push", "from": [x, y], "to": [nx, ny], "pushed_to": [px, py]})
        
        # Stone to river flips
        for orientation in ["horizontal", "vertical"]:
            moves.append({"action": "flip", "from": [x, y], "orientation": orientation})
    
    else:  # River piece
        # River to stone flip
        moves.append({"action": "flip", "from": [x, y]})
        
        # River rotation
        moves.append({"action": "rotate", "from": [x, y]})
    
    return moves

def generate_all_moves(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    Generate all legal moves for the current player.
    
    Args:
        board: Current board state
        player: Current player ("circle" or "square")
        rows, cols: Board dimensions
        score_cols: Scoring column indices
    
    Returns:
        List of all valid move dictionaries
    """
    all_moves = []
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player:
                piece_moves = get_valid_moves_for_piece(board, x, y, player, rows, cols, score_cols)
                all_moves.extend(piece_moves)
    
    return all_moves

# ==================== BOARD EVALUATION ====================

def count_stones_in_scoring_area(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    """Count how many stones a player has in their scoring area."""
    count = 0
    
    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)
    
    for x in score_cols:
        if in_bounds(x, score_row, rows, cols):
            piece = board[score_row][x]
            if piece and piece.owner == player and piece.side == "stone":
                count += 1
    
    return count




# ---------- helpers (cheap, local, no imports) ----------
def goal_row_for(p, rows):
    return bottom_score_row(rows) if p == "square" else top_score_row()

def lane_dir(p):
    # +1 means moving downwards, -1 upwards
    return 1 if p == "square" else -1

def is_stone(piece):
    return getattr(piece, "side", "stone") == "stone"

def is_river(piece):
    return piece is not None and not is_stone(piece)

def river_is_vertical(piece):
    ori = getattr(piece, "orientation", None)
    return ori == "vertical"

def river_is_horizontal(piece):
    ori = getattr(piece, "orientation", None)
    return ori == "horizontal"

def river_push_potential(board, ox, oy, player, rows, cols):
    """
    Heuristic: how likely a FRIENDLY river can help push the obstacle at (ox, oy)
    toward our goal direction. We look in a small neighborhood:
      - vertical river directly 'behind' the obstacle (toward our pieces) = strong assist
      - vertical river two squares behind with empty in between = moderate
      - horizontal rivers adjacent that point into the obstacle = weak assist
    """
    step = lane_dir(player)
    pot = 0.0

    # one behind
    bx, by = ox, oy - step
    if in_bounds(bx, by, rows, cols):
        r1 = board[by][bx]
        if r1 and is_river(r1) and r1.owner == player and river_is_vertical(r1):
            pot += 1.0

    # two behind with space
    bx2, by2 = ox, oy - 2*step
    if in_bounds(bx2, by2, rows, cols) and in_bounds(bx, by, rows, cols):
        r2 = board[by2][bx2]
        mid = board[by][bx]
        if r2 and is_river(r2) and r2.owner == player and river_is_vertical(r2) and mid is None:
            pot += 0.5

    # horizontal nudge from sides (weaker, still useful if square beyond obstacle might open)
    for sx in (ox-1, ox+1):
        if in_bounds(sx, oy, rows, cols):
            rh = board[oy][sx]
            if rh and is_river(rh) and rh.owner == player and river_is_horizontal(rh):
                pot += 0.35

    return pot

def scan_lane(board, x, y, player, rows, cols, score_cols):
    """
    Scan straight toward player's goal row on column x.
    Count hard blocks (stones), soft blocks (rivers), and accumulate push assists.
    Returns a dict with:
      - enemy_stones, self_stones, rivers
      - my_rivers, opp_rivers
      - push_assist (friendly rivers near the FIRST hard block)
      - clear (no items at all)
    """
    grow  = goal_row_for(player, rows)
    step  = lane_dir(player)

    enemy_stones = 0
    self_stones  = 0
    rivers       = 0
    my_rivers    = 0
    opp_rivers   = 0
    push_assist  = 0.0
    first_hard_block_seen = False

    r = y + step
    # inclusive toward the goal row
    while (r <= grow if step == 1 else r >= grow):
        piece = board[r][x]
        if piece is not None:
            if is_stone(piece):
                # record who blocks
                if piece.owner == player:
                    self_stones += 1
                else:
                    enemy_stones += 1
                # first hard block: assess push potential
                if not first_hard_block_seen:
                    push_assist += river_push_potential(board, x, r, player, rows, cols)
                    first_hard_block_seen = True
            else:
                # rivers in lane
                rivers += 1
                if piece.owner == player:
                    my_rivers += 1
                else:
                    opp_rivers += 1
        r += step

    return {
        "enemy_stones": enemy_stones,
        "self_stones":  self_stones,
        "rivers":       rivers,
        "my_rivers":    my_rivers,
        "opp_rivers":   opp_rivers,
        "push_assist":  push_assist,
        "clear":        (enemy_stones == 0 and self_stones == 0 and rivers == 0),
        "in_score_col": (x in score_cols),
    }
def basic_evaluate_board(board, player, rows, cols, score_cols):
    """
    Perspective-correct evaluation: positive favors `player`, negative favors opponent.
    Incorporates:
      - scored stones
      - imminent scores (1 move away on scoring columns)
      - distance-to-goal
      - lane clarity per stone (hard vs soft blocks)
      - river push assist against first hard block in lane
      - lane presence on scoring columns
      - mobility (clipped)
      - basic push-safety proxy
    """
    opponent = get_opponent(player)
    score_set = set(score_cols)

    # --------- weights (don’t get heroic; this hierarchy works) ---------
    W_SCORED        = 200.0  # already in scoring row
    W_IMMINENT      = 100.0   # 1 step from scoring
    W_DIST          = 8.0    # vertical progress
    W_LANE_CLEAR    = 60.0   # pristine column to goal
    W_BLOCK_ENEMY   = 15.0   # enemy stone blocks your lane
    W_BLOCK_SELF    = 10.0   # your own stone blocks your lane
    W_RIVER_SOFT    = 6.0    # any river in your lane is friction
    W_MY_RIVER_HELP = 20.0    # friendly rivers in lane (soft help)
    W_OPP_RIVER_HURT= 20.0    # opponent rivers in lane (soft hurt)
    W_PUSH_ASSIST   = 12.0   # friendly river poised to push first hard block
    W_LANE_PRESENCE = 10.0    # being in scoring columns
    W_SAFETY_THREAT = 8.0    # adjacent push threat proxy
    W_MOBILITY      = 0.6    # clipped mobility diff

    # --------- quick helpers ----------
    def can_be_pushed(x, y, by_owner):
        # Very cheap proxy: adjacent enemy stone with empty “behind” square to push into
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        for dx, dy in dirs:
            ax, ay = x + dx, y + dy
            bx, by = x - dx, y - dy
            if not in_bounds(ax, ay, rows, cols) or not in_bounds(bx, by, rows, cols):
                continue
            a = board[ay][ax]
            b = board[by][bx]
            if a and is_stone(a) and a.owner == by_owner:
                if b is None and not is_opponent_score_cell(bx, by, by_owner, rows, cols, score_cols):
                    return True
        return False

    # --------- 1) base: scored stones dominate ----------
    my_scored  = count_stones_in_scoring_area(board, player,   rows, cols, score_cols)
    opp_scored = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    score = W_SCORED * (my_scored - opp_scored)

    my_row  = goal_row_for(player, rows)
    opp_row = goal_row_for(opponent, rows)

    my_lane_presence  = 0
    opp_lane_presence = 0

    # --------- 2) iterate all pieces once ----------
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece:
                continue

            if is_stone(piece):
                # -------- my stones --------
                if piece.owner == player:
                    dist = abs(my_row - y)
                    score += W_DIST * (rows - dist)

                    # imminent: one step away and in scoring col
                    if dist == 1 and x in score_set:
                        score += W_IMMINENT

                    # lane presence
                    if x in score_set:
                        my_lane_presence += 1

                    # lane scan
                    lane = scan_lane(board, x, y, player, rows, cols, score_cols)
                    if lane["clear"]:
                        score += W_LANE_CLEAR
                    else:
                        # river push assist reduces the pain of first hard block
                        assist = lane["push_assist"]
                        score -= W_BLOCK_ENEMY * max(0, lane["enemy_stones"] - assist)
                        score -= W_BLOCK_SELF  * lane["self_stones"]
                        score -= W_RIVER_SOFT  * lane["rivers"]
                        score += W_MY_RIVER_HELP  * lane["my_rivers"]
                        score -= W_OPP_RIVER_HURT * lane["opp_rivers"]

                    # push safety: if they can likely push me, penalize
                    if can_be_pushed(x, y, by_owner=opponent):
                        score -= W_SAFETY_THREAT

                # -------- opponent stones (mirror, subtract) --------
                elif piece.owner == opponent:
                    dist = abs(opp_row - y)
                    score -= W_DIST * (rows - dist)

                    if dist == 1 and x in score_set:
                        score -= W_IMMINENT

                    if x in score_set:
                        opp_lane_presence += 1

                    lane = scan_lane(board, x, y, opponent, rows, cols, score_cols)
                    if lane["clear"]:
                        score -= W_LANE_CLEAR
                    else:
                        assist = lane["push_assist"]
                        # From their POV, our stones are the “enemy”
                        score += W_BLOCK_ENEMY * max(0, lane["enemy_stones"] - assist)
                        score += W_BLOCK_SELF  * lane["self_stones"]
                        score += W_RIVER_SOFT  * lane["rivers"]
                        score -= W_MY_RIVER_HELP  * lane["my_rivers"]
                        score += W_OPP_RIVER_HURT * lane["opp_rivers"]

                    if can_be_pushed(x, y, by_owner=player):
                        score += W_SAFETY_THREAT

            else:
                # rivers are already accounted for via lane scans and push assist.
                # If you insist on micromanaging orientation, add tiny forward nudges here.
                pass

    # --------- 3) lane presence ----------
    score += W_LANE_PRESENCE * (my_lane_presence - opp_lane_presence)

    # --------- 4) mobility (clipped to keep it secondary) ----------
    # my_moves  = len(generate_all_moves(board, player,   rows, cols, score_cols))
    # opp_moves = len(generate_all_moves(board, opponent, rows, cols, score_cols))
    # mob_delta = max(-20, min(20, my_moves - opp_moves))
    # score += W_MOBILITY * mob_delta

    return score




def simulate_move(board: List[List[Any]], move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any]:
    """
    Simulate a move on a copy of the board.
    
    Args:
        board: Current board state
        move: Move to simulate
        player: Player making the move
        rows, cols: Board dimensions
        score_cols: Scoring column indices
    
    Returns:
        (success: bool, new_board_state or error_message)
    """
    # Import the game engine's move validation function
    try:
        from gameEngine import validate_and_apply_move
        board_copy = copy.deepcopy(board)
        success, message = validate_and_apply_move(board_copy, move, player, rows, cols, score_cols)
        return success, board_copy if success else message
    except ImportError:
        # Fallback to basic simulation if game engine not available
        return True, copy.deepcopy(board)

# ==================== BASE AGENT CLASS ====================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def __init__(self, player: str):
        """Initialize agent with player identifier."""
        self.player = player
        self.opponent = get_opponent(player)
    
    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.
        
        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions
            score_cols: List of column indices for scoring areas
        
        Returns:
            Dictionary representing the chosen move, or None if no moves available
        """
        pass

# ==================== STUDENT AGENT IMPLEMENTATION ====================

class StudentAgent(BaseAgent):
    """
    Student Agent Implementation
    
    TODO: Implement your AI agent for the River and Stones game.
    The goal is to get 4 of your stones into the opponent's scoring area.
    
    You have access to these utility functions:
    - generate_all_moves(): Get all legal moves for current player
    - basic_evaluate_board(): Basic position evaluation 
    - simulate_move(): Test moves on board copy
    - count_stones_in_scoring_area(): Count stones in scoring positions
    """
    
    def __init__(self, player: str):
        super().__init__(player)
        # TODO: Add any initialization you need
    
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.
        
        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions  
            score_cols: Column indices for scoring areas
            
        Returns:
            Dictionary representing your chosen move
        """
        moves = generate_all_moves(board, self.player, rows, cols, score_cols)
        
        if not moves:
            return None
        
        ##adaptive choice of depth
        base_ms = 500
        if opponent_time < 3.0:
            base_ms = 200  # opponent running out of time
        elif current_player_time < 3.0:
            base_ms = 200  # player running out of time
        deadline = time.perf_counter() + base_ms/1000.0
        depth = 25 if ( current_player_time < 3.0 or opponent_time < 3.0 ) else 30
        start_time = time.perf_counter()



        if current_player_time <=0: ## no time left
            return None

        # deadline = time.perf_counter() + 0.05 ## 50 ms


        best_score = float('-inf')
        best_move = None
        
        for move in moves:
            ok, result = simulate_move(board, move, self.player, rows, cols, score_cols)
            if not ok:
                continue
            new_board = result
            
            score = self.minimax(new_board,depth-1,alpha = float('-inf'),beta=float('inf'),maxim = False, to_move = self.opponent, rows= rows, cols = cols, score_cols = score_cols,deadline = deadline, start_time = start_time)
            if score>best_score:
                best_score=score
                best_move = move
        
            if time.perf_counter() > deadline:
                break
        if not best_move:
            best_move = moves[0]
        
        return best_move
    def minimax(self,board,depth,alpha,beta,maxim,to_move,rows,cols,score_cols,deadline, start_time):
        # print(f"depth : {depth} at {time.perf_counter()-start_time}\n")
        if time.perf_counter() > deadline or depth ==0: ## timeout or final depth reached
            return basic_evaluate_board(board,self.player,rows,cols,score_cols)
        moves = generate_all_moves(board, to_move, rows, cols, score_cols)
        if not moves:
            return basic_evaluate_board(board,self.player,rows,cols,score_cols) ##evaluate from the POV of the player

        if maxim:
            value = float('-inf')
            for move in moves:
                ok, result = simulate_move(board, move, to_move, rows, cols, score_cols)
                if not ok:
                    continue
                child_board = result
                child = self.minimax(child_board, depth-1, alpha, beta, False, get_opponent(to_move), rows, cols, score_cols, deadline, start_time)
                value= max(value,child)
                alpha = max(alpha,value)

                if beta <=alpha: ## pruning condition
                    break
                if deadline < time.perf_counter(): ## overtime
                    break
            return value
        else:
            value = float('inf')
            for move in moves:
                ok, result = simulate_move(board, move, to_move, rows, cols, score_cols)
                if not ok:
                    continue
                child_board = result
                child = self.minimax(child_board, depth-1, alpha, beta, False, get_opponent(to_move), rows, cols, score_cols, deadline, start_time)
                value= min(value,child)
                beta = min(beta,value)

                if beta <=alpha: ## pruning condition
                    break
                if deadline > time.perf_counter(): ## overtime
                    break
            return value

# ==================== TESTING HELPERS ====================

def test_student_agent():
    """
    Basic test to verify the student agent can be created and make moves.
    """
    print("Testing StudentAgent...")
    
    try:
        from gameEngine import default_start_board, DEFAULT_ROWS, DEFAULT_COLS
        
        rows, cols = DEFAULT_ROWS, DEFAULT_COLS
        score_cols = score_cols_for(cols)
        board = default_start_board(rows, cols)
        
        agent = StudentAgent("circle")
        move = agent.choose(board, rows, cols, score_cols,1.0,1.0)
        
        if move:
            print("✓ Agent successfully generated a move")
        else:
            print("✗ Agent returned no move")
    
    except ImportError:
        agent = StudentAgent("circle")
        print("✓ StudentAgent created successfully")

if __name__ == "__main__":
    # Run basic test when file is executed directly
    test_student_agent()
