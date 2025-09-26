import argparse, json, copy, time
from typing import List, Optional, Dict, Any, Tuple

# Agent factory now expects only (side, strategy)
from agent import get_agent

# pygame optional
try:
    import pygame
except Exception:
    pygame = None

# ---------------- Config ----------------
DEFAULT_ROWS = 13
DEFAULT_COLS = 12
CELL = 48
MARGIN = 60
FPS = 30
TIME_PER_PLAYER = 1 * 60  # Default 1 minute per player
WIN_COUNT = 4

# Colors - Light yellow background color scheme
BG = (255, 253, 240)  # Light yellow background
BOARD_COLOR = (250, 248, 235)  # Light cream board color
GRID_COLOR = (100, 110, 120)  # Darker gray for grid visibility
HIGHLIGHT = (50, 150, 80)  # Green highlight
CIRCLE_COLOR = (200, 60, 80)  # Red for circles
SQUARE_COLOR = (60, 100, 200)  # Blue for squares
STONE_FILL = (255, 255, 255)  # Pure white for stone centers
RIVER_FILL_CIRCLE = (180, 70, 70)  # Red for circle rivers
RIVER_FILL_SQUARE = (50, 80, 180)  # Blue for square rivers
TEXT_COLOR = (40, 50, 60)  # Dark text for light background
SELECTED_COLOR = (220, 140, 40)  # Orange for selection
SCORE_AREA_COLOR = (150, 120, 180)  # Purple for score areas
SHADOW_COLOR = (0, 0, 0, 40)  # Slightly darker shadow for light background

# ---------------- Piece & Board Utilities ----------------
def opponent(p):
    return 'circle' if p == 'square' else 'square'
class Piece:
    def __init__(self, owner:str, side:str="stone", orientation:Optional[str]=None):
        self.owner = owner
        self.side = side
        self.orientation = orientation if orientation else "horizontal"
    def copy(self): return Piece(self.owner, self.side, self.orientation)
    def to_dict(self): return {"owner":self.owner,"side":self.side,"orientation":self.orientation}
    @staticmethod
    def from_dict(d:Optional[Dict[str,Any]]):
        if d is None: return None
        return Piece(d["owner"], d.get("side","stone"), d.get("orientation","horizontal"))

def empty_board(rows:int, cols:int) -> List[List[Optional[Piece]]]:
    return [[None for _ in range(cols)] for __ in range(rows)]

def default_start_board(rows:int, cols:int) -> List[List[Optional[Piece]]]:
    board = empty_board(rows, cols)
    width = min(6, max(2, cols - 6))
    start_cols = list(range((cols - width)//2, (cols - width)//2 + width))
    top_rows = [3,4]   # buffer at 0
    bot_rows = [rows-5, rows-4]  # buffer at rows-1
    for r in top_rows:
        for c in start_cols:
            board[r][c] = Piece("square","stone")
    for r in bot_rows:
        for c in start_cols:
            board[r][c] = Piece("circle","stone")
    return board

def load_board_from_file(path:str):
    with open(path,"r",encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data.get("board")
    rows = len(raw); cols = len(raw[0])
    board = [[Piece.from_dict(cell) if cell else None for cell in row] for row in raw]
    return board, rows, cols

def save_board_to_file(board, path:str):
    data = {"board":[[cell.to_dict() if cell else None for cell in row] for row in board]}
    with open(path,"w",encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

# ---------------- Score helpers ----------------
def score_cols_for(cols:int) -> List[int]:
    w=4
    start = max(0, (cols - w)//2)
    return list(range(start, start+w))

def top_score_row() -> int:
    return 2

def bottom_score_row(rows:int) -> int:
    return rows - 3

# ---------------- Scoring helpers ----------------
def is_own_score_cell(x:int, y:int, player:str, rows:int, cols:int, score_cols:List[int]) -> bool:
    """Return True if (x,y) is player's own scoring cell."""
    # A cell is player's own scoring area if it would be considered opponent score
    # when the 'player' argument is swapped (i.e. is_opponent_score_cell for opponent)
    return is_opponent_score_cell(x, y, opponent(player), rows, cols, score_cols)

def count_scoring_pieces(board:List[List[Optional[Piece]]],
                         player:str, rows:int, cols:int, score_cols:List[int]) -> int:
    """n_self: number of player's pieces (stone side up) already in player's scoring area."""
    n = 0
    for y,row in enumerate(board):
        for x,p in enumerate(row):
            if p and p.owner == player and p.side == "stone" and is_own_score_cell(x, y, player, rows, cols, score_cols):
                n += 1
    return n

def count_reachable_in_one(board:List[List[Optional[Piece]]],
                           player:str, rows:int, cols:int, score_cols:List[int]) -> int:
    """
    m_self: number of player's pieces (stone side up) that can reach the player's scoring
    area in one legal move (including moves produced by river flow and pushes returned
    by compute_valid_targets).
    """
    m = 0
    for y,row in enumerate(board):
        for x,p in enumerate(row):
            if p and p.owner == player and p.side == "stone":
                if is_own_score_cell(x, y, player, rows, cols, score_cols):
                    continue
                info = compute_valid_targets(board, x, y, player, rows, cols, score_cols)
                # moves is a set of (tx,ty)
                for (tx,ty) in info.get('moves', set()):
                    if is_own_score_cell(tx, ty, player, rows, cols, score_cols):
                        m += 1
                        break
                else:
                    # check pushes: pushes is list of ((tx,ty),(ptx,pty))
                    for of,pushed in info.get('pushes', []):
                        ptx, pty = pushed
                        if is_own_score_cell(ptx, pty, player, rows, cols, score_cols):
                            m += 1
                            break
            if p and p.owner == player and p.side == "river":
                # if the river piece is in the scoring area already then can be flipped to get a stone in the scoring area
                if is_own_score_cell(x, y, player, rows, cols, score_cols):
                    m += 1
    return m

def compute_final_scores(board:List[List[Optional[Piece]]],
                         winner:Optional[str],
                         rows:int, cols:int, score_cols:List[int],
                         remaining_times:Optional[Dict[str,float]] = None) -> Dict[str,float]:
    """
    Return dict {'circle':score, 'square':score}.
    winner may be 'circle', 'square', or None (draw).
    If remaining_times provided (dict with keys 'circle' and 'square' containing remaining time),
    then if one player's clock is <= 0 and the other player's clock > 0 the latter is declared winner.
    Implements the scoring rules from the spec:
      - Victory: winner gets 100 - (n_lose + m_lose/10), loser gets (n_lose + m_lose/10)
      - Draw: each player gets DrawScore (30) + MarginScore/4
        where MarginScore = 39 + ((n_self + m_self/10) - (n_opp + m_opp/10))
    """
    # If a remaining_times dict is passed and no winner was set, derive winner from clocks
    if remaining_times is not None and winner is None:
        c_time = remaining_times.get('circle', None)
        s_time = remaining_times.get('square', None)
        if c_time is not None and s_time is not None:
            if c_time <= 0 and s_time > 0:
                winner = 'square'
            elif s_time <= 0 and c_time > 0:
                winner = 'circle'
            # if both <= 0 or both > 0 -> leave winner as-is (None or previously set)

    # helper to obtain n and m for a player or opponent
    def nm_for(player):
        n = count_scoring_pieces(board, player, rows, cols, score_cols)
        m = count_reachable_in_one(board, player, rows, cols, score_cols)
        return float(n), float(m)

    scores = {'circle': 0.0, 'square': 0.0}
    if winner in ("circle", "square"):
        loser = opponent(winner)
        n_loser, m_loser = nm_for(loser)
        loser_score = float(n_loser) + float(m_loser) / 10.0
        winner_score = 100.0 - loser_score
        scores[winner] = winner_score
        scores[loser] = loser_score
    else:
        # draw
        DRAW_SCORE = 30.0
        for player in ("circle", "square"):
            n_self, m_self = nm_for(player)
            n_opp, m_opp = nm_for(opponent(player))
            margin = 39.0 + ((float(n_self) + float(m_self)/10.0) - (float(n_opp) + float(m_opp)/10.0))
            total = DRAW_SCORE + margin / 4.0
            scores[player] = total
    return scores


def in_bounds(x:int,y:int,rows:int,cols:int) -> bool:
    return 0 <= x < cols and 0 <= y < rows

def is_opponent_score_cell(x:int,y:int,player:str,rows:int,cols:int,score_cols:List[int]) -> bool:
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

# ---------------- River flow & validation (authoritative) ----------------
def get_river_flow_destinations(board:List[List[Optional[Piece]]],
                                rx:int, ry:int, sx:int, sy:int, player:str,
                                rows:int, cols:int, score_cols:List[int],
                                river_push:bool=False) -> List[Tuple[int,int]]:
    destinations=[]; visited=set(); queue=[(rx,ry)]
    while queue:
        x,y = queue.pop(0)
        if (x,y) in visited or not in_bounds(x,y,rows,cols): continue
        visited.add((x,y))
        cell = board[y][x]
        if river_push and x==rx and y==ry:
            cell = board[sy][sx]
        if cell is None:
            if is_opponent_score_cell(x,y,player,rows,cols,score_cols):
                # block entering opponent score
                pass
            else:
                destinations.append((x,y))
            continue
        if cell.side != "river":
            continue
        dirs = [(1,0),(-1,0)] if cell.orientation == "horizontal" else [(0,1),(0,-1)]
        for dx,dy in dirs:
            nx, ny = x+dx, y+dy
            while in_bounds(nx,ny,rows,cols):
                if is_opponent_score_cell(nx,ny,player,rows,cols,score_cols):
                    break
                next_cell = board[ny][nx]
                if next_cell is None:
                    destinations.append((nx,ny)); nx += dx; ny += dy; continue
                if nx==sx and ny==sy:
                    nx += dx; ny += dy; continue
                if next_cell.side == "river":
                    queue.append((nx,ny)); break
                break
    out=[]; seen=set()
    for d in destinations:
        if d not in seen:
            seen.add(d); out.append(d)
    return out

# ---------------- Compute valid targets (authoritative) ----------------
def compute_valid_targets(board:List[List[Optional[Piece]]],
                          sx:int, sy:int, player:str,
                          rows:int, cols:int, score_cols:List[int]) -> Dict[str,Any]:
    if not in_bounds(sx,sy,rows,cols):
        return {'moves': set(), 'pushes': []}
    p = board[sy][sx]
    if p is None or p.owner != player:
        return {'moves': set(), 'pushes': []}
    moves=set(); pushes=[]
    dirs=[(1,0),(-1,0),(0,1),(0,-1)]
    for dx,dy in dirs:
        tx,ty = sx+dx, sy+dy
        if not in_bounds(tx,ty,rows,cols): continue
        # block entering opponent score cell
        if is_opponent_score_cell(tx,ty,player,rows,cols,score_cols):
            continue
        target = board[ty][tx]
        if target is None:
            moves.add((tx,ty))
        elif target.side == "river":
            flow = get_river_flow_destinations(board, tx, ty, sx, sy, player, rows, cols, score_cols)
            for d in flow: moves.add(d)
        else:
            # stone occupied
            if p.side == "stone":
                px,py = tx+dx, ty+dy
                if in_bounds(px,py,rows,cols) and board[py][px] is None and not is_opponent_score_cell(px,py,p.owner,rows,cols,score_cols):
                    pushes.append(((tx,ty),(px,py)))
            else:
                pushed_player = target.owner
                flow = get_river_flow_destinations(board, tx, ty, sx, sy, pushed_player, rows, cols, score_cols, river_push=True)
                for d in flow:
                    if not is_opponent_score_cell(d[0],d[1],pushed_player,rows,cols,score_cols):
                        pushes.append(((tx,ty),(d[0],d[1])))
    return {'moves': moves, 'pushes': pushes}

# ---------------- Validate & apply move (authoritative) ----------------
def validate_and_apply_move(board:List[List[Optional[Piece]]],
                            move:Dict[str,Any],
                            player:str,
                            rows:int, cols:int, score_cols:List[int]) -> Tuple[bool,str]:
    if not isinstance(move, dict):
        return False, "move must be dict"
    action = move.get("action")
    if action == "move":
        fr = move.get("from"); to = move.get("to")
        if not fr or not to: return False, "move needs from & to"
        fx,fy = int(fr[0]), int(fr[1]); tx,ty = int(to[0]), int(to[1])
        if not in_bounds(fx,fy,rows,cols) or not in_bounds(tx,ty,rows,cols): return False, "oob"
        if is_opponent_score_cell(tx,ty,player,rows,cols,score_cols): return False, "can't go into opponent score"
        piece = board[fy][fx]
        if piece is None or piece.owner != player: return False, "invalid piece"
        if board[ty][tx] is None:
            board[ty][tx]=piece; board[fy][fx]=None; return True, "moved"
        pushed = move.get("pushed_to")
        if not pushed: return False, "destination occupied; pushed_to required"
        ptx,pty = int(pushed[0]), int(pushed[1])
        dx = tx - fx; dy = ty - fy
        if (ptx,pty) != (tx+dx, ty+dy): return False, "invalid pushed_to"
        if not in_bounds(ptx,pty,rows,cols): return False, "oob"
        if is_opponent_score_cell(ptx,pty,player,rows,cols,score_cols): return False, "can't push into opponent score"
        if board[pty][ptx] is not None: return False, "pushed_to not empty"
        board[pty][ptx] = board[ty][tx]; board[ty][tx] = piece; board[fy][fx] = None
        return True, "move+push applied"

    elif action == "push":
        fr = move.get("from"); to = move.get("to"); pushed = move.get("pushed_to")
        if not fr or not to or not pushed:
            return False, "push needs from,to,pushed_to"

        fx, fy = int(fr[0]), int(fr[1])
        tx, ty = int(to[0]), int(to[1])
        px, py = int(pushed[0]), int(pushed[1])

        if not (in_bounds(fx,fy,rows,cols) and in_bounds(tx,ty,rows,cols) and in_bounds(px,py,rows,cols)):
            return False, "oob"
        pushed_player = board[ty][tx].owner if board[ty][tx] else None
        if (is_opponent_score_cell(tx,ty,player,rows,cols,score_cols) or
            is_opponent_score_cell(px,py,pushed_player,rows,cols,score_cols)):
            return False, "push would enter opponent score cell"

        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"

        if board[ty][tx] is None:
            return False, "to must be occupied"
        if board[py][px] is not None:
            return False, "pushed_to not empty"

        if piece.side == "river" and board[ty][tx].side == "river":
            return False, "rivers cannot push rivers"

        info = compute_valid_targets(board, fx, fy, player, rows, cols, score_cols)
        valid_pairs = info['pushes']
        if ((tx,ty), (px,py)) not in valid_pairs:
            return False, "push pair invalid"

        board[py][px] = board[ty][tx]  # enemy goes to pushed_to
        board[ty][tx] = board[fy][fx]  # mover goes into enemy's cell
        board[fy][fx] = None           # origin cleared

        mover = board[ty][tx]
        if mover.side == "river":
            mover.side = "stone"
            mover.orientation = None

        return True, "push applied"
    
        

        # if (tx,ty),(px,py) not in valid_pairs and not any(of==(tx,ty) and pf==(px,py) for of,pf in valid_pairs):
        #     return False, "push pair invalid"
        board[py][px] = board[ty][tx]; board[ty][tx] = board[fy][fx]; board[fy][fx] = None
        return True, "push applied"

    elif action == "flip":
        fr = move.get("from")
        if not fr: return False, "flip needs from"
        fx,fy = int(fr[0]), int(fr[1])
        if not in_bounds(fx,fy,rows,cols): return False, "oob"
        piece = board[fy][fx]
        if piece is None or piece.owner != player: return False, "invalid piece"
        if piece.side == "stone":
            ori = move.get("orientation")
            if ori not in ("horizontal","vertical"): return False, "stone->river needs orientation"
            # check resulting river flow doesn't reach opponent score
            piece.side="river"; piece.orientation=ori
            flow = get_river_flow_destinations(board, fx, fy, fx, fy, player, rows, cols, score_cols)
            # revert for now; we will finalize only if safe
            piece.side="stone"; piece.orientation=None
            for (dx,dy) in flow:
                if is_opponent_score_cell(dx,dy,player,rows,cols,score_cols):
                    return False, "flip would allow flow into opponent score"
            # commit flip
            piece.side="river"; piece.orientation=ori
            return True, "flipped to river"
        else:
            piece.side="stone"; piece.orientation=None
            return True, "flipped to stone"

    elif action == "rotate":
        fr = move.get("from")
        if not fr: return False, "rotate needs from"
        fx,fy = int(fr[0]), int(fr[1])
        if not in_bounds(fx,fy,rows,cols): return False, "oob"
        piece = board[fy][fx]
        if piece is None or piece.owner != player: return False, "invalid"
        if piece.side != "river": return False, "rotate only on river"
        piece.orientation = "horizontal" if piece.orientation=="vertical" else "vertical"
        flow = get_river_flow_destinations(board, fx, fy, fx, fy, player, rows, cols, score_cols)
        for (dx,dy) in flow:
            if is_opponent_score_cell(dx,dy,player,rows,cols,score_cols):
                piece.orientation = "horizontal" if piece.orientation=="vertical" else "vertical"
                return False, "rotation allows flow into opponent score"
        return True, "rotated"

    return False, "unknown action"

# ---------------- Generate moves for agents (compatibility) ----------------
def generate_all_moves(board:List[List[Optional[Piece]]],
                       player:str, rows:int, cols:int, score_cols:List[int]) -> List[Dict[str,Any]]:
    # This is a convenience implementation; agents have their own generators,
    # but main provides this as well for reference or alternative usage.
    moves=[]
    dirs=[(1,0),(-1,0),(0,1),(0,-1)]
    for y in range(rows):
        for x in range(cols):
            p = board[y][x]
            if not p or p.owner != player: continue
            if p.side == "stone":
                for dx,dy in dirs:
                    nx,ny = x+dx,y+dy
                    if not in_bounds(nx,ny,rows,cols): continue
                    if is_opponent_score_cell(nx,ny,player,rows,cols,score_cols): continue
                    if board[ny][nx] is None:
                        moves.append({"action":"move","from":[x,y],"to":[nx,ny]})
                    else:
                        target = board[ny][nx]
                        if target.side == "river":
                            # moves that flow through the river
                            flow = get_river_flow_destinations(board, nx, ny, x, y, player, rows, cols, score_cols)
                            for d in flow:
                                moves.append({"action":"move","from":[x,y],"to":d})
                        else:
                            # moves to push the stone pieces (can push self and opponent pieces both)
                            px,py = nx+dx, ny+dy
                            if in_bounds(px,py,rows,cols) and board[py][px] is None and not is_opponent_score_cell(px,py,target.owner,rows,cols,score_cols):
                                moves.append({"action":"push","from":[x,y],"to":[nx,ny],"pushed_to":[px,py]})
                # flips
                for ori in ("horizontal","vertical"):
                    p.side="river"; p.orientation=ori
                    moves.append({"action":"flip","from":[x,y],"orientation":ori})
            else:
                for dx,dy in dirs:
                    nx,ny = x+dx,y+dy
                    if not in_bounds(nx,ny,rows,cols): continue
                    if is_opponent_score_cell(nx,ny,player,rows,cols,score_cols): continue
                    if board[ny][nx] is None:
                        moves.append({"action":"move","from":[x,y],"to":[nx,ny]})
                    else:
                        target = board[ny][nx]
                        if target.side == "river":
                            # moves that flow through the river
                            flow = get_river_flow_destinations(board, nx, ny, x, y, player, rows, cols, score_cols)
                            for d in flow:
                                moves.append({"action":"move","from":[x,y],"to":d})
                        else:
                            # moves to push the stone pieces (can push self and opponent pieces both)
                            px,py = nx+dx, ny+dy
                            if in_bounds(px,py,rows,cols) and board[py][px] is None and not is_opponent_score_cell(px,py,target.owner,rows,cols,score_cols):
                                moves.append({"action":"push","from":[x,y],"to":[nx,ny],"pushed_to":[px,py]})
                # flip to stone side
                moves.append({"action":"flip","from":[x,y]})
                # rotate
                new_ori = "vertical" if p.orientation=="horizontal" else "horizontal"
                p.orientation = new_ori
                moves.append({"action":"rotate","from":[x,y]})
    return moves

# ---------------- Win check ----------------
def check_win(board:List[List[Optional[Piece]]], rows:int, cols:int, score_cols:List[int]) -> Optional[str]:
    top = top_score_row(); bot = bottom_score_row(rows)
    ccount=0; scount=0
    for x in score_cols:
        if in_bounds(x, top, rows, cols):
            p = board[top][x]; 
            if p and p.owner=="circle" and p.side=="stone": ccount+=1
        if in_bounds(x, bot, rows, cols):
            q = board[bot][x]
            if q and q.owner=="square" and q.side=="stone": scount+=1
    if ccount >= WIN_COUNT: return "circle"
    if scount >= WIN_COUNT: return "square"
    return None

# ---------------- ASCII for CLI ----------------
def board_to_ascii(board:List[List[Optional[Piece]]], rows:int, cols:int, score_cols:List[int]) -> str:
    """Enhanced ASCII representation with better visualization."""
    result = "\n" + "="*50 + "\n"
    result += "🎮 RIVER AND STONES GAME 🎮\n"
    result += "="*50 + "\n"
    
    # Print column numbers
    col_header = "   "
    for x in range(cols):
        col_header += f"{x:2d} "
    result += col_header + "\n"
    
    rows_out = []
    top = top_score_row()
    bot = bottom_score_row(rows)
    
    for y in range(rows):
        row_str = f"{y:2d} "
        
        for x in range(cols):
            p = board[y][x]
            
            # Determine if this is a scoring cell
            is_circle_score = (y == top) and (x in score_cols)
            is_square_score = (y == bot) and (x in score_cols)
            
            if not p:
                if is_circle_score:
                    cell = "🔴"  # Circle scoring area
                elif is_square_score:
                    cell = "🔵"  # Square scoring area
                else:
                    cell = " ·"  # Empty cell
            else:
                if p.owner == "circle":
                    if p.side == "stone":
                        cell = "⭕" if is_circle_score else "🔴"
                    else:  # river
                        if p.orientation == "horizontal":
                            cell = "🔴↔"  # Red horizontal two-way arrow for circle
                        else:
                            cell = "🔴↕"  # Red vertical two-way arrow for circle
                else:  # square
                    if p.side == "stone":
                        cell = "⬜" if is_square_score else "🔵"
                    else:  # river
                        if p.orientation == "horizontal":
                            cell = "🔵↔"  # Blue horizontal two-way arrow for square
                        else:
                            cell = "🔵↕"  # Blue vertical two-way arrow for square
            
            row_str += f"{cell} "
        
        # Add row indicator for scoring rows
        if y == top:
            row_str += " ← Circle scores here"
        elif y == bot:
            row_str += " ← Square scores here"
            
        rows_out.append(row_str)
    
    result += "\n".join(rows_out)
    
    # Add legend
    legend = "\n\nLEGEND:\n"
    legend += "🔴 Circle stone  ⭕ Circle in score\n"
    legend += "🔵 Square stone  ⬜ Square in score\n"
    legend += "🔴↔ Circle horizontal rivers  🔴↕ Circle vertical rivers\n"
    legend += "🔵↔ Square horizontal rivers  🔵↕ Square vertical rivers\n"
    legend += "🔴🔵 Empty scoring areas\n"
    
    return result + legend

# ---------------- GUI rendering & loop ----------------
if pygame:
    pygame.init()
    FONT = pygame.font.SysFont("arial", 14)
    BIGFONT = pygame.font.SysFont("arial", 18)

def draw_board(screen, board, rows, cols, score_cols, selected, highlights, msg, timers, current):
    screen.fill(BG)
    
    # Draw background with gradient effect
    board_rect = pygame.Rect(MARGIN-30, MARGIN-30, cols*CELL+60, rows*CELL+60)
    pygame.draw.rect(screen, BOARD_COLOR, board_rect, border_radius=15)
    
    # Add subtle border
    pygame.draw.rect(screen, GRID_COLOR, board_rect, 3, border_radius=15)
    
    # Draw scoring areas with subtle design
    top = top_score_row(); bot = bottom_score_row(rows)
    
    # Circle's scoring area (top) - subtle design
    for x in score_cols:
        r = pygame.Rect(MARGIN + x*CELL - CELL//2, MARGIN + top*CELL - CELL//2, CELL, CELL)
        # Subtle background
        s = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        s.fill((*CIRCLE_COLOR, 20))
        screen.blit(s, r.topleft)
        # Subtle border
        pygame.draw.rect(screen, CIRCLE_COLOR, r, 2, border_radius=8)
    
    # Square's scoring area (bottom) - subtle design
    for x in score_cols:
        r = pygame.Rect(MARGIN + x*CELL - CELL//2, MARGIN + bot*CELL - CELL//2, CELL, CELL)
        # Subtle background
        s = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        s.fill((*SQUARE_COLOR, 20))
        screen.blit(s, r.topleft)
        # Subtle border
        pygame.draw.rect(screen, SQUARE_COLOR, r, 2, border_radius=8)

    # Draw subtle grid points
    for y in range(rows):
        for x in range(cols):
            cx = MARGIN + x*CELL; cy = MARGIN + y*CELL
            # Smaller, subtle grid points
            pygame.draw.circle(screen, GRID_COLOR, (cx,cy), 3)
            pygame.draw.circle(screen, BG, (cx,cy), 1)

    # Draw subtle highlights
    for hx,hy in highlights:
        center = (MARGIN + hx*CELL, MARGIN + hy*CELL)
        # Subtle highlight ring
        pygame.draw.circle(screen, HIGHLIGHT, center, 24, 2)

    # Draw subtle selection indicator
    if selected:
        sx, sy = selected
        center = (MARGIN + sx*CELL, MARGIN + sy*CELL)
        # Subtle selection ring
        pygame.draw.circle(screen, SELECTED_COLOR, center, 26, 3)

    # Draw pieces with larger white centers and subtle design
    for y in range(rows):
        for x in range(cols):
            p = board[y][x]
            if not p: continue
            cx = MARGIN + x*CELL; cy = MARGIN + y*CELL
            color = CIRCLE_COLOR if p.owner=="circle" else SQUARE_COLOR
            
            # Subtle shadow effect
            shadow_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surf, SHADOW_COLOR, (CELL//2 + 1, CELL//2 + 1), CELL//2 - 4)
            screen.blit(shadow_surf, (cx - CELL//2, cy - CELL//2))
            
            # Outer colored ring (thinner)
            pygame.draw.circle(screen, color, (cx, cy), CELL//2 - 4, 3)
            
            # Larger white inner circle for all pieces
            pygame.draw.circle(screen, STONE_FILL, (cx, cy), CELL//2 - 8)
            
            if p.side == "stone":
                # Stone appearance with subtle colored border around larger white center
                pygame.draw.circle(screen, color, (cx, cy), CELL//2 - 8, 2)
                # Small center dot for stone identification
                pygame.draw.circle(screen, color, (cx, cy), 3)
            else:
                # River appearance - thin rivers inside the larger white circle
                river_color = RIVER_FILL_CIRCLE if p.owner == "circle" else RIVER_FILL_SQUARE
                
                if p.orientation == "horizontal":
                    # Thin horizontal river inside larger white circle
                    river_width = 3  # Thinner river
                    river_length = CELL - 20  # Fits inside larger white circle
                    river_rect = pygame.Rect(cx - river_length//2, cy - river_width//2, river_length, river_width)
                    pygame.draw.rect(screen, river_color, river_rect, border_radius=1)
                    
                    # Subtle flow direction indicators
                    arrow_size = 2
                    for i in range(2):
                        arrow_x = cx - river_length//2 + 8 + i * (river_length - 16) // 1
                        # Small arrow pointing right
                        pygame.draw.polygon(screen, river_color, [
                            (arrow_x, cy - arrow_size),
                            (arrow_x + arrow_size, cy),
                            (arrow_x, cy + arrow_size)
                        ])
                else:
                    # Thin vertical river inside larger white circle
                    river_width = 3  # Thinner river
                    river_length = CELL - 20  # Fits inside larger white circle
                    river_rect = pygame.Rect(cx - river_width//2, cy - river_length//2, river_width, river_length)
                    pygame.draw.rect(screen, river_color, river_rect, border_radius=1)
                    
                    # Subtle flow direction indicators
                    arrow_size = 2
                    for i in range(2):
                        arrow_y = cy - river_length//2 + 8 + i * (river_length - 16) // 1
                        # Small arrow pointing down
                        pygame.draw.polygon(screen, river_color, [
                            (cx - arrow_size, arrow_y),
                            (cx, arrow_y + arrow_size),
                            (cx + arrow_size, arrow_y)
                        ])

    # Improved message display with background
    msg_bg_rect = pygame.Rect(10, rows*CELL + MARGIN + 10, screen.get_width()-20, 35)
    pygame.draw.rect(screen, (0,0,0,150), msg_bg_rect, border_radius=5)
    msg_surf = BIGFONT.render(msg, True, (255, 255, 255))  # White text
    screen.blit(msg_surf, (20, rows*CELL + MARGIN + 18))
    
    # Improved timer display
    timer_bg = pygame.Rect(10, 5, 250, 70)
    pygame.draw.rect(screen, (0,0,0,150), timer_bg, border_radius=5)
    
    t1 = FONT.render(f"Circle: {format_time(timers['circle'])}", True, CIRCLE_COLOR)
    t2 = FONT.render(f"Square: {format_time(timers['square'])}", True, SQUARE_COLOR)
    turn_color = CIRCLE_COLOR if current == "circle" else SQUARE_COLOR
    turn = BIGFONT.render(f"Turn: {current.title()}", True, turn_color)
    
    screen.blit(t1, (20, 15))
    screen.blit(t2, (20, 35))
    screen.blit(turn, (20, 55))
    
    # Add game instructions in corner
    instructions = [
        "Controls:",
        "M - Move mode",
        "P - Push mode", 
        "F - Flip (stone ↔ river)",
        "R - Rotate river",
        "ESC - Cancel",
        "S - Save game"
    ]
    
    inst_bg = pygame.Rect(screen.get_width()-180, 5, 170, len(instructions)*20 + 10)
    pygame.draw.rect(screen, (0,0,0,120), inst_bg, border_radius=5)
    
    for i, instruction in enumerate(instructions):
        color = (255, 255, 255)  # White text for all instructions
        font = BIGFONT if i == 0 else FONT
        inst_surf = font.render(instruction, True, color)
        screen.blit(inst_surf, (screen.get_width()-175, 15 + i*20))
    
    pygame.display.flip()

def format_time(sec:float) -> str:
    if sec < 0: sec = 0
    m = int(sec//60); s = int(sec%60)
    return f"{m:02d}:{s:02d}"

def run_gui(mode:str, circle_strategy:str, square_strategy:str, load_file:Optional[str],
            rows:int, cols:int, time_per_player:float):
    if not pygame:
        print("pygame not available; use --nogui")
        return
    score_cols = score_cols_for(cols)
    board = default_start_board(rows, cols)
    turn=0

    window_width = max(800, cols*CELL + MARGIN*2 + 200)
    window_height = max(600, rows*CELL + MARGIN*2 + 100)
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"🎮 River and Stones - {mode.upper()} Mode")  # ADDED: set window caption

    clock = pygame.time.Clock()
    players = {"circle":"human","square":"human"}
    # if mode == "hvai": players["square"]="ai"
    # elif mode == "aivai": players = {"circle":"ai","square":"ai"}
    if mode=="aivai": players={"circle":"ai","square":"ai"}
    elif mode=="hvh": players={"circle":"human","square":"human"}
    else:
        if circle_strategy=="random": players={"circle":"ai","square":"human"}
        else: players={"circle":"human","square":"ai"}
    
    # instantiate agents (they only receive board)
    agent_circle = get_agent("circle", circle_strategy)
    agent_square = get_agent("square", square_strategy)
    agents = {}
    if players["circle"]=="ai": agents["circle"] = agent_circle
    if players["square"]=="ai": agents["square"] = agent_square

    timers = {"circle": time_per_player, "square": time_per_player}

    current = "circle"
    selected = None
    highlights = set()
    msg = "Select a piece and choose an action (M/P/F/R). Welcome to River and Stones!"
    action_mode = None
    winner = None
    game_scores = None
    push_stage = None
    push_candidate = None
    game_over = False

    turn_start = time.time()

    while True:
        clock.tick(FPS)

        # Check timeouts
        if not game_over:
            if timers["circle"] <= 0 and timers["square"] > 0:
                winner = "square"; game_over = True
                print("Circle timed out.")
            elif timers["square"] <= 0 and timers["circle"] > 0:
                winner = "circle"; game_over = True
                print("Square timed out.")
            elif timers["circle"] <= 0 and timers["square"] <= 0:
                winner = None; game_over = True
                print("Both players timed out. Draw.")

        # if not winner:
        #     timers[current] -= (now - last); last = now
        #     if timers[current] <= 0:
        #         winner = opponent(current); msg = f"{current.title()} timed out. {winner.title()} wins!"

        # AI turn (single call)
        if players[current] == "ai" and not winner and not game_over:
            ai_start = time.time()
            agent = agents[current]
            move = agent.choose(board, rows, cols, score_cols, timers[current], timers[opponent(current)])
            ai_end = time.time()
            ai_elapsed = ai_end - ai_start
            timers[current] -= ai_elapsed
            if timers[current] <= 0:
                winner = opponent(current); msg = f"{current.title()} timed out. {winner.title()} wins!"; game_over = True
            else:
                if move:
                    ok, info = validate_and_apply_move(board, move, current, rows, cols, score_cols)
                    msg = f"AI {current}: {info}"
                    if ok:
                        w = check_win(board, rows, cols, score_cols)
                        if w: winner = w; msg = f"{w.title()} wins!"; game_over = True
                        current = opponent(current)
                        selected=None; highlights=set(); action_mode=None; push_stage=None; push_candidate=None
                        turn_start = time.time()  # NEW: reset timer when switching to next (human) turn
                    else:
                        current = opponent(current)
                        turn_start = time.time()
                else:
                    current = opponent(current)
                    turn_start = time.time()
            draw_board(screen, board, rows, cols, score_cols, selected, highlights, msg, timers, current)
            turn += 1
            if turn > 1000:
                print("Turn limit reached -> draw"); break
            continue

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if game_over:  # block further moves
                continue

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_s:
                    save_board_to_file(board, "saved_board.json"); msg = "Saved board"
                if ev.key == pygame.K_ESCAPE:
                    selected=None; highlights=set(); action_mode=None; push_stage=None; push_candidate=None; msg="Cleared"
                if selected and ev.key == pygame.K_m:
                    action_mode = "move"
                    info = compute_valid_targets(board, selected[0], selected[1], current, rows, cols, score_cols)
                    highlights = set(info['moves'])
                    msg = "Move mode: click a highlighted dest"
                if selected and ev.key == pygame.K_p:
                    action_mode="push"; push_stage=0; push_candidate=None
                    info = compute_valid_targets(board, selected[0], selected[1], current, rows, cols, score_cols)
                    own_finals = set([of for of,pf in info['pushes']])
                    highlights = set(own_finals); msg = "Push mode: click own_final"
                if selected and ev.key == pygame.K_f:
                    action_mode="flip"; msg = "Flip: press H/V for stone->river or F to flip river->stone"
                if selected and ev.key == pygame.K_r:
                    sx,sy = selected; p = board[sy][sx]
                    if p and p.owner==current and p.side=="river":
                        m = {"action":"rotate","from":[sx,sy]}
                        ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols)
                        msg = info
                        if ok:
                            w = check_win(board, rows, cols, score_cols)
                            if w: winner=w; msg = f"{w.title()} wins!"; game_over = True
                            current = opponent(current); selected=None; highlights=set(); action_mode=None
                    else:
                        msg = "Rotate needs selected river piece"
                if action_mode=="flip" and selected:
                    sx,sy = selected
                    if ev.key == pygame.K_h or ev.key == pygame.K_v:
                        ori = "horizontal" if ev.key==pygame.K_h else "vertical"
                        m={"action":"flip","from":[sx,sy],"orientation":ori}
                        ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols)
                        msg = info
                        if ok:
                            w = check_win(board, rows, cols, score_cols)
                            if w: winner=w; msg = f"{w.title()} wins!"; game_over = True
                            current = opponent(current); selected=None; highlights=set(); action_mode=None
                    elif ev.key == pygame.K_f:
                        m={"action":"flip","from":[sx,sy]}
                        ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols)
                        msg = info
                        if ok:
                            w = check_win(board, rows, cols, score_cols)
                            if w: winner=w; msg = f"{w.title()} wins!"; game_over = True
                            current = opponent(current); selected=None; highlights=set(); action_mode=None

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button==1:
                # measure elapsed thinking time before move
                elapsed = time.time() - turn_start
                timers[current] -= elapsed
                if timers[current] <= 0:
                    winner = opponent(current)
                    msg = f"{current.title()} timed out. {winner.title()} wins!"
                    game_over = True
                    continue
                turn_start = time.time()

                mx,my = ev.pos
                rx = round((mx - MARGIN)/CELL); ry = round((my - MARGIN)/CELL)
                if not in_bounds(rx,ry,rows,cols): continue

                if selected is None:
                    p = board[ry][rx]
                    if p and p.owner==current:
                        selected=(rx,ry); highlights=set(); action_mode=None
                        push_stage=None; push_candidate=None
                        msg=f"Selected {selected}"
                    else:
                        msg = "Select one of your pieces"

                else:
                    sx,sy = selected
                    if action_mode=="move":
                        info = compute_valid_targets(board,sx,sy,current,rows,cols,score_cols)
                        if (rx,ry) not in info['moves']:
                            msg = "Invalid target"
                        else:
                            if board[ry][rx] is None:
                                m={"action":"move","from":[sx,sy],"to":[rx,ry]}
                            else:
                                dx,dy = rx-sx, ry-sy
                                m={"action":"move","from":[sx,sy],"to":[rx,ry],"pushed_to":[rx+dx,ry+dy]}
                            ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols)
                            msg = info
                            if ok:
                                w = check_win(board,rows,cols,score_cols)
                                if w: winner=w; msg=f"{w.title()} wins!"; game_over=True
                                current = opponent(current)
                                selected=None; highlights=set(); action_mode=None
                                push_stage=None; push_candidate=None
                                turn_start = time.time()  # NEW: reset for next turn

                    elif action_mode=="push":
                        info = compute_valid_targets(board,sx,sy,current,rows,cols,score_cols)
                        push_pairs = info['pushes']
                        own_finals = set([of for of,pf in push_pairs])
                        if push_stage==0 or push_stage is None:
                            if (rx,ry) not in own_finals:
                                msg = "Click a valid own_final"
                            else:
                                push_candidate=(rx,ry)
                                pushed_options=[pf for of,pf in push_pairs if of==push_candidate]
                                highlights=set(pushed_options); push_stage=1
                                msg=f"Selected own_final {push_candidate}. Click pushed_to"
                        else:
                            if push_candidate is None:
                                msg = "Push error; reselect"
                                push_stage=None; push_candidate=None; highlights=set(); action_mode=None
                            else:
                                candidate_pair=(push_candidate,(rx,ry))
                                if candidate_pair not in push_pairs:
                                    msg="Invalid pushed_to"
                                else:
                                    m={"action":"push","from":[sx,sy],
                                       "to":[push_candidate[0],push_candidate[1]],
                                       "pushed_to":[rx,ry]}
                                    ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols)
                                    msg=info
                                    push_stage=None; push_candidate=None; highlights=set(); action_mode=None
                                    if ok:
                                        w = check_win(board,rows,cols,score_cols)
                                        if w: winner=w; msg=f"{w.title()} wins!"; game_over=True
                                        current = opponent(current)
                                        selected=None
                                        turn_start = time.time()  # NEW

                    elif action_mode=="flip":
                        p = board[sy][sx]
                        if p.side=="river":
                            m={"action":"flip","from":[sx,sy]}
                            ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols)
                            msg=info
                            if ok:
                                w = check_win(board,rows,cols,score_cols)
                                if w: winner=w; msg=f"{w.title()} wins!"; game_over=True
                                current = opponent(current)
                                selected=None; action_mode=None
                                turn_start = time.time()  # NEW
                        else:
                            msg = "Press H/V for stone->river in flip mode"

                    else:
                        info = compute_valid_targets(board,sx,sy,current,rows,cols,score_cols)
                        if (rx,ry) in info['moves']:
                            if board[ry][rx] is None:
                                m={"action":"move","from":[sx,sy],"to":[rx,ry]}
                            else:
                                dx,dy = rx-sx, ry-sy
                                m={"action":"move","from":[sx,sy],"to":[rx,ry],"pushed_to":[rx+dx,ry+dy]}
                            ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols)
                            msg=info
                            if ok:
                                w = check_win(board,rows,cols,score_cols)
                                if w: winner=w; msg=f"{w.title()} wins!"; game_over=True
                                current = opponent(current)
                                selected=None; highlights=set(); action_mode=None
                                turn_start = time.time()  # NEW
                        else:
                            newp = board[ry][rx]
                            if newp and newp.owner==current:
                                selected=(rx,ry); highlights=set(); action_mode=None
                                msg=f"Selected {selected}"
                            else:
                                msg="Invalid click"
        # turn += 1
        # print(turn)
        # if turn > 1000:
        #     print("Turn limit reached -> draw"); break

        # --- DRAW ---
        draw_board(screen, board, rows, cols, score_cols, selected, highlights, msg, timers, current)

        if game_over and game_scores is None:
            game_scores = compute_final_scores(
                board, winner, rows, cols, score_cols,
                remaining_times={'circle': timers['circle'], 'square': timers['square']}
            )
            if winner in ("circle","square"):
                msg = f"{winner.title()} wins! Scores — Circle: {game_scores['circle']:.1f}, Square: {game_scores['square']:.1f}"
            else:
                msg = f"Draw. Scores — Circle: {game_scores['circle']:.1f}, Square: {game_scores['square']:.1f}"


# ---------------- CLI interactive runner ----------------
def run_cli(mode:str, circle_strategy:str, square_strategy:str, load_file:Optional[str], rows:int, cols:int, time_per_player:float):
    score_cols = score_cols_for(cols)
    board = default_start_board(rows, cols)
    agent_circle = get_agent("circle", circle_strategy)
    agent_square = get_agent("square", square_strategy)
    players = {"circle":"human","square":"human"}
    if mode=="aivai": players={"circle":"ai","square":"ai"}
    elif mode=="hvh": players={"circle":"human","square":"human"}
    else:
        if circle_strategy=="random": players={"circle":"ai","square":"human"}
        else: players={"circle":"human","square":"ai"}
    
    current="circle"; winner=None; turn=0

    print("🎮 Welcome to River and Stones! 🎮")
    print(f"Mode: {mode.upper()}")
    print(f"Circle: {circle_strategy}, Square: {square_strategy}")
    
    # Timers (seconds) - ADDED
    timers = {"circle": time_per_player, "square": time_per_player}  # ADDED

    while True:
        print(board_to_ascii(board, rows, cols, score_cols))
        
        w = check_win(board, rows, cols, score_cols)
        if w:
            winner = w
            print(f"\n🎉 WINNER: {w.upper()} 🎉")
            break
            
        # immediate timeout check
        if timers["circle"] <= 0 and timers["square"] > 0:
            winner = "square"; print("\nCircle timed out."); break
        if timers["square"] <= 0 and timers["circle"] > 0:
            winner = "circle"; print("\nSquare timed out."); break
        if timers["circle"] <= 0 and timers["square"] <= 0:
            # both timed out -> draw
            print("\nBoth players timed out. Game ends as a draw.")
            break

        print(f"\n{'='*30}")
        print(f"Turn {turn + 1}: {current.upper()}'s move")
        print(f"Remaining time — Circle: {format_time(timers['circle'])} | Square: {format_time(timers['square'])}")
        print(f"{'='*30}")
        
        if players[current]=="ai":
            print(f"🤖 AI {current} is thinking...")
            agent = agent_circle if current=="circle" else agent_square
            ai_start = time.time()  # ADDED: measure AI thinking time
            move = agent.choose(board, rows, cols, score_cols, timers[current], timers[opponent(current)])
            ai_end = time.time()  # ADDED
            elapsed = ai_end - ai_start  # ADDED
            timers[current] -= elapsed  # ADDED: subtract AI think time from clock

            # check timeout after thinking
            if timers[current] <= 0:
                winner = opponent(current)
                print(f"{current.title()} timed out while thinking. {winner.title()} wins!")
                break

            if move is None:
                print(f"AI {current} has no moves; pass")
                current = opponent(current)
                turn += 1
                if turn > 1000:
                    print("Turn limit reached -> draw"); break
                # do not count the "press enter to continue" as clock time; skip it
                input("\nPress Enter to continue...")  # keep for readability
                continue
            ok,msg = validate_and_apply_move(board, move, current, rows, cols, score_cols)
            print(f"AI {current} -> {move}")
            print(f"Result: {msg}")
            if not ok:
                current = opponent(current)
                turn += 1
                if turn > 1000:
                    print("Turn limit reached -> draw"); break
                input("\nPress Enter to continue...")
                continue
        else:
            # Human: measure time spent entering the move so the timer decreases
            print("Commands:")
            print("  Move: {'action':'move','from':[x,y],'to':[x,y]}")
            print("  Push: {'action':'push','from':[x,y],'to':[x,y],'pushed_to':[x,y]}")
            print("  Flip: {'action':'flip','from':[x,y],'orientation':'horizontal/vertical'}")
            print("  Rotate: {'action':'rotate','from':[x,y]}")
            print("  'q' to quit")
            start = time.time()  # ADDED: start timing user input
            try:
                s = input(f"\n{current} move JSON: ").strip()
            except Exception:
                print("\nInput interrupted. Exiting.")
                break
            end = time.time()  # ADDED
            elapsed = end - start  # ADDED
            timers[current] -= elapsed  # ADDED: subtract user input time from timer

            # check timeout after user input
            if timers[current] <= 0:
                winner = opponent(current)
                print(f"{current.title()} timed out. {winner.title()} wins!")
                break

            if s.lower()=="q":
                break
            try:
                move = json.loads(s)
            except Exception as e:
                print(f"Bad JSON: {e}"); 
                # do not change player on bad input, let them try again
                continue
            ok,msg = validate_and_apply_move(board, move, current, rows, cols, score_cols)
            print(f"Result: {msg}")
            if not ok:
                continue
            
        # after a successful move / AI move attempt, check board win
        w = check_win(board, rows, cols, score_cols)
        if w:
            winner = w
            print(f"\n🎉 WINNER: {w.upper()} 🎉")
            break

        # next player's turn
        current = opponent(current)
        turn += 1
        if turn > 1000:
            print("Turn limit reached -> draw"); break

        # Press Enter pause for readability — DO NOT count this time as player's clock (unchanged behavior)
        try:
            _ = input("\nPress Enter to continue...")
        except KeyboardInterrupt:
            pass
        
    # compute and print final scores with remaining times accounted for (CHANGED: pass timers)
    final_scores = compute_final_scores(board, winner, rows, cols, score_cols,
                                        remaining_times={'circle': timers['circle'], 'square': timers['square']})
    if winner:
        print(f"\n{winner.title()} wins!")
    else:
        print("\nGame ended in a draw.")
    print(f"Final Scores -> Circle: {final_scores['circle']:.1f} | Square: {final_scores['square']:.1f}\n")

# ---------------- Entrypoint ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hvh","hvai","aivai"], default="hvai")
    ap.add_argument("--circle", choices=["random","student","student_cpp"], default="random")
    ap.add_argument("--square", choices=["random","student","student_cpp"], default="random")
    ap.add_argument("--load", default=None)
    ap.add_argument("--nogui", action="store_true")
    ap.add_argument("--time", type=float, default=1.0, help="Time per player in minutes (default: 1.0)")
    args = ap.parse_args()

    rows = DEFAULT_ROWS; cols = DEFAULT_COLS
    time_per_player = args.time * 60  # Convert minutes to seconds

    if args.nogui:
        run_cli(args.mode, args.circle, args.square, args.load, rows, cols, time_per_player)
    else:
        run_gui(args.mode, args.circle, args.square, args.load, rows, cols, time_per_player)

if __name__=="__main__":
    main()
