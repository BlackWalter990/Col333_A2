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
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# ==================== GAME UTILITIES ====================
# Essential utility functions for game state analysis

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < cols and 0 <= y < rows

def score_cols_for(cols: int) -> List[int]:
    """Get the column indices for scoring areas."""
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))

def top_score_row() -> int:
    """Get the row index for Circle's scoring area."""
    return 2

def bottom_score_row(rows: int) -> int:
    """Get the row index for Square's scoring area."""
    return rows - 3

def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the opponent's scoring area."""
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the player's own scoring area."""
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)

def get_opponent(player: str) -> str:
    """Get the opponent player identifier."""
    return "square" if player == "circle" else "circle"

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

def basic_evaluate_board(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> float:
    """
    Basic board evaluation function.
    
    Returns a score where higher values are better for the given player.
    Students can use this as a starting point and improve it.
    """
    # score = 0.0
    # opponent = get_opponent(player)
    
    # # Count stones in scoring areas
    # player_scoring_stones = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    # opponent_scoring_stones = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    
    # score += player_scoring_stones * 100  
    # score -= opponent_scoring_stones * 100  
    
    # # Count total pieces and positional factors
    # for y in range(rows):
    #     for x in range(cols):
    #         piece = board[y][x]
    #         if piece and piece.owner == player and piece.side == "stone":
    #             # Basic positional scoring
    #             if player == "circle":
    #                 score += (rows - y) * 0.1
    #             else:
    #                 score += y * 0.1
    
    # return score

    opponent = get_opponent(player)
    score = 0.0

    # === 1. Stones already scored ===
    my_scored = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    opp_scored = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    score += my_scored * 100
    score -= opp_scored * 100

    # Target rows
    my_goal_row = bottom_score_row(rows) if player == "square" else top_score_row()
    opp_goal_row = bottom_score_row(rows) if opponent == "square" else top_score_row()

    # === 2. Distance-to-goal + 3. Threats-in-one ===
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece:
                continue

            # My stones
            if piece.owner == player and piece.side == "stone":
                dist = abs(my_goal_row - y)
                score += (rows - dist) * 2  # closer is better

                # If can step into goal in one move
                if dist == 1 and x in score_cols:
                    score += 50

            # Opponent stones
            elif piece.owner == opponent and piece.side == "stone":
                dist = abs(opp_goal_row - y)
                score -= (rows - dist) * 2

                if dist == 1 and x in score_cols:
                    score -= 50

            # === 4. Rivers ===
            elif piece.side.startswith("river"):
                # Reward rivers near my stones, penalize if near opponent
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny, rows, cols):
                        adj = board[ny][nx]
                        if adj and adj.side == "stone":
                            if adj.owner == player:
                                score += 5
                            else:
                                score -= 5

    # === 5. Mobility advantage ===
    my_moves = len(generate_all_moves(board, player, rows, cols, score_cols))
    opp_moves = len(generate_all_moves(board, opponent, rows, cols, score_cols))
    score += (my_moves - opp_moves) * 0.5

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
        base_ms = 1000
        if opponent_time < 3.0:
            base_ms = 500  # opponent running out of time
        elif current_player_time < 3.0:
            base_ms = 500  # player running out of time
        deadline = time.perf_counter() + base_ms/1000.0
        depth = 10 if ( current_player_time < 3.0 or opponent_time < 3.0 ) else 12


        if current_player_time <=0: ## no time left
            return None

        # deadline = time.perf_counter() + 0.05 ## 50 ms


        best_score = float('-inf')
        best_move = None
        
        for move in moves:
            ok, new_board = simulate_move(board,move,self.player,rows,cols,score_cols)
            if not (new_board and ok) :
                continue
            
            score = self.minimax(new_board,depth-1,alpha = float('-inf'),beta=float('inf'),maxim = False, to_move = self.opponent, rows= rows, cols = cols, score_cols = score_cols,deadline = deadline)
            if score>best_score:
                best_score=score
                best_move = move
        
            if time.perf_counter() > deadline:
                break
        if not best_move:
            best_move = moves[0]
        
        return best_move
    def minimax(self,board,depth,alpha,beta,maxim,to_move,rows,cols,score_cols,deadline):
        if time.perf_counter() > deadline or depth ==0: ## timeout or final depth reached
            return basic_evaluate_board(board,to_move,rows,cols,score_cols)
        moves = generate_all_moves(board, to_move, rows, cols, score_cols)
        if not moves:
            return basic_evaluate_board(board,self.player,rows,cols,score_cols) ##evaluate from the POV of the player

        if maxim:
            value = float('-inf')
            for move in moves:
                ok, new_board = simulate_move(board,move,to_move,rows,cols,score_cols)
                if not (new_board and ok):
                    continue
                child = self.minimax(new_board,depth-1,alpha,beta,False,get_opponent(to_move),rows,cols,score_cols,deadline)
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
                ok, new_board = simulate_move(board,move,to_move,rows,cols,score_cols)
                if not (new_board and ok):
                    continue
                child = self.minimax(new_board,depth-1,alpha,beta,True,get_opponent(to_move),rows,cols,score_cols,deadline)
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
