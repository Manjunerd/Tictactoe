"""
Tic Tac Toe backend built with FastAPI.

Game rules implemented server-side:
  - Human is always 'X', AI is always 'O'.
  - Human moves first.
  - After every human move, the server validates it, checks for a
    win/draw, and if the game continues, immediately computes and
    applies the AI's move using the minimax algorithm (with alpha-beta
    pruning). Difficulty controls how "smart" the AI's move choice is:
      * easy      -> random legal move
      * medium    -> 50% best (minimax) move, 50% random move
      * difficult -> always the best (minimax) move -> unbeatable

All game state lives server-side, keyed by a game_id (uuid4), so the
client cannot cheat by mutating the board directly. The client only
ever sends which cell it wants to play.
"""

import math
import random
import uuid
from enum import Enum
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from database import Base, engine, get_db
import models
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.responses import RedirectResponse
import secrets
import os
app = FastAPI(title="Tic Tac Toe API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET"),
    same_site="lax",
    https_only=False,   # Local development only
)

# 2. Configure Authlib with your Google Credentials  # Loads variables from your .env file
oauth = OAuth(config)

oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)
Base.metadata.create_all(bind=engine)

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------


# Allow the front end to be served from anywhere during development.
# If you serve the HTML from this same FastAPI app (recommended, see
# bottom of file) CORS isn't even needed, but it's harmless to leave on.
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]

HUMAN = "X"
AI = "O"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    difficult = "difficult"


class Status(str, Enum):
    ongoing = "ongoing"
    x_win = "x_win"
    o_win = "o_win"
    draw = "draw"


# --------------------------------------------------------------------------
# In-memory game storage
# --------------------------------------------------------------------------

class GameState:
    __slots__ = ("board", "difficulty", "status")

    def __init__(self, difficulty: Difficulty):
        self.board: List[Optional[str]] = [None] * 9
        self.difficulty = difficulty
        self.status: Status = Status.ongoing


GAMES: Dict[str, GameState] = {}


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------

class NewGameRequest(BaseModel):
    difficulty: Difficulty = Difficulty.medium


class MoveRequest(BaseModel):
    cell: int = Field(..., ge=0, le=8, description="Board index 0-8")


class GameResponse(BaseModel):
    game_id: str
    board: List[Optional[str]]
    status: Status
    difficulty: Difficulty
    message: str


class RecordStatsRequest(BaseModel):
    result: str  # "win", "loss", "draw"



# --------------------------------------------------------------------------
# Core game logic
# --------------------------------------------------------------------------

def check_status(board: List[Optional[str]]) -> Status:
    for a, b, c in WIN_LINES:
        if board[a] is not None and board[a] == board[b] == board[c]:
            return Status.x_win if board[a] == HUMAN else Status.o_win
    if all(cell is not None for cell in board):
        return Status.draw
    return Status.ongoing


def empty_cells(board: List[Optional[str]]) -> List[int]:
    return [i for i, cell in enumerate(board) if cell is None]


def minimax(board: List[Optional[str]], depth: int, is_maximizing: bool,
            alpha: float, beta: float) -> int:
    """
    Returns the minimax score of `board` from AI's (O's) perspective.
    Positive scores favor the AI (O), negative favor the human (X).
    `depth` is used to prefer faster wins / slower losses.
    """
    status = check_status(board)
    if status == Status.o_win:
        return 10 - depth
    if status == Status.x_win:
        return depth - 10
    if status == Status.draw:
        return 0

    if is_maximizing:
        best = -math.inf
        for i in empty_cells(board):
            board[i] = AI
            score = minimax(board, depth + 1, False, alpha, beta)
            board[i] = None
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return int(best)
    else:
        best = math.inf
        for i in empty_cells(board):
            board[i] = HUMAN
            score = minimax(board, depth + 1, True, alpha, beta)
            board[i] = None
            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return int(best)


def best_ai_move(board: List[Optional[str]]) -> int:
    """Full-strength minimax move for the AI (unbeatable)."""
    best_score = -math.inf
    move = empty_cells(board)[0]
    for i in empty_cells(board):
        board[i] = AI
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[i] = None
        if score > best_score:
            best_score = score
            move = i
    return move


def choose_ai_move(board: List[Optional[str]], difficulty: Difficulty) -> int:
    cells = empty_cells(board)
    if difficulty == Difficulty.easy:
        return random.choice(cells)
    if difficulty == Difficulty.medium:
        if random.random() < 0.5:
            return best_ai_move(board)
        return random.choice(cells)
    # difficult -> always optimal
    return best_ai_move(board)


def status_message(status: Status) -> str:
    return {
        Status.ongoing: "Your move.",
        Status.x_win: "You win!",
        Status.o_win: "AI wins!",
        Status.draw: "It's a draw!",
    }[status]


def get_game(game_id: str) -> GameState:
    game = GAMES.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------

@app.post("/api/game/new", response_model=GameResponse)
def new_game(req: NewGameRequest):
    game_id = str(uuid.uuid4())
    game = GameState(req.difficulty)
    GAMES[game_id] = game
    return GameResponse(
        game_id=game_id,
        board=game.board,
        status=game.status,
        difficulty=game.difficulty,
        message="New game started. Your move.",
    )


@app.get("/api/game/{game_id}", response_model=GameResponse)
def get_game_state(game_id: str):
    game = get_game(game_id)
    return GameResponse(
        game_id=game_id,
        board=game.board,
        status=game.status,
        difficulty=game.difficulty,
        message=status_message(game.status),
    )


@app.post("/api/game/{game_id}/move", response_model=GameResponse)
def play_move(game_id: str, move: MoveRequest):
    game = get_game(game_id)

    if game.status != Status.ongoing:
        raise HTTPException(status_code=400, detail="Game already finished")

    if game.board[move.cell] is not None:
        raise HTTPException(status_code=400, detail="Cell already occupied")

    # Apply human move
    game.board[move.cell] = HUMAN
    game.status = check_status(game.board)

    # If the game continues, let the AI respond immediately
    if game.status == Status.ongoing:
        ai_index = choose_ai_move(game.board, game.difficulty)
        game.board[ai_index] = AI
        game.status = check_status(game.board)

    return GameResponse(
        game_id=game_id,
        board=game.board,
        status=game.status,
        difficulty=game.difficulty,
        message=status_message(game.status),
    )


@app.delete("/api/game/{game_id}")
def delete_game(game_id: str):
    GAMES.pop(game_id, None)
    return {"deleted": True}


# --------------------------------------------------------------------------
# Serve the front end
# --------------------------------------------------------------------------
# The tictactoe.html file lives in ./static. Mounting it means you can run
# this single FastAPI app and open http://localhost:8000/ to play, with no
# separate web server and no CORS issues.

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("login.html")


@app.get("/login.html")
def serve_login():
    return FileResponse("login.html")


@app.get("/gameplay.html")
def serve_gameplay():
    return FileResponse("gameplay.html")

@app.get("/api/auth/login")
async def login(request: Request):
    redirect_uri = str(request.url_for('auth_callback'))
    if "127.0.0.1" in redirect_uri:
        redirect_uri = redirect_uri.replace("127.0.0.1", "localhost")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/api/auth/callback", name="auth_callback")
async def auth_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")
    
    user_info = token.get('userinfo')
    if not user_info:
        resp = await oauth.google.get('https://www.googleapis.com/oauth2/v3/userinfo', token=token)
        user_info = resp.json()
        
    if not user_info:
        raise HTTPException(status_code=400, detail="Failed to retrieve user profile from Google")
        
    google_id = user_info.get('sub')
    email = user_info.get('email')
    name = user_info.get('name')
    picture = user_info.get('picture')
    
    if not google_id or not email:
        raise HTTPException(status_code=400, detail="Required user profile fields missing from Google")
        
    # Check if user already exists
    user = db.query(models.User).filter(models.User.google_id == google_id).first()
    if not user:
        user = models.User(
            id=str(uuid.uuid4()),
            google_id=google_id,
            email=email,
            name=name or email.split('@')[0],
            picture=picture,
            wins=0,
            losses=0,
            draws=0,
            games=0,
            highest_streak=0,
            current_streak=0
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Update name and picture if changed
        user.name = name or user.name
        user.picture = picture or user.picture
        db.commit()
        
    request.session['user_id'] = user.id
    
    return RedirectResponse(url="/gameplay.html")


@app.get("/api/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login.html")


@app.get("/api/user/me")
def get_current_user(request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get('user_id')
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
        
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "picture": user.picture,
        "wins": user.wins,
        "losses": user.losses,
        "draws": user.draws,
        "games": user.games,
        "highest_streak": user.highest_streak,
        "current_streak": user.current_streak
    }


@app.post("/api/user/stats/record")
def record_stats(req: RecordStatsRequest, request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get('user_id')
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
        
    user.games += 1
    if req.result == "win":
        user.wins += 1
        user.current_streak += 1
        if user.current_streak > user.highest_streak:
            user.highest_streak = user.current_streak
    elif req.result == "loss":
        user.losses += 1
        user.current_streak = 0
    elif req.result == "draw":
        user.draws += 1
        user.current_streak = 0
        
    db.commit()
    db.refresh(user)
    return {
        "wins": user.wins,
        "losses": user.losses,
        "draws": user.draws,
        "games": user.games,
        "highest_streak": user.highest_streak,
        "current_streak": user.current_streak
    }
@app.post("/api/user/stats/reset")
def reset_stats(request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get('user_id')
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
        
    user.wins = 0
    user.losses = 0
    user.draws = 0
    user.games = 0
    user.highest_streak = 0
    user.current_streak = 0
    db.commit()
    return {"message": "Stats reset successfully"}



@app.get("/db-test")
def db_test(db: Session = Depends(get_db)):
    return {"message": "Database connected successfully"}