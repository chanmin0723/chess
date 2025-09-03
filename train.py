# train.py (with eval+checkpoint manager)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Play 기반 ValueNet 학습 스크립트 (평가/체크포인트 관리 포함).

업그레이드 요약
 - TD(0) 부트스트랩 타깃: y_t = γ * (-V(s_{t+1})) (종국이면 ±1/0)
 - 스파링 주입: 주기적으로 greedy1/random과 대국하여 버퍼 혼합
 - 드로우 가중치 낮춤: 무승부 샘플 손실 가중치
 - 오프닝 다양화/수제한 완화
 - [NEW] 주기 평가(Eval) 및 저장 정책:
     * latest: save_name (항상 최신으로 덮어씀)
     * best  : save_name_base_best.pth (평가 점수 최고 갱신 시)
     * numbered: save_name_base_gXXXX.pth (주기 저장, --keep-last개 유지)
 - [NEW] train_log.csv에 진행 로그/평가 기록

권장 예시
  python train.py \
    --num-games 800 --depth 2 --device auto \
    --max-moves 768 --opening-moves 4 \
    --epsilon-start 0.40 --epsilon-final 0.05 --epsilon-decay-games 500 \
    --td-gamma 0.99 --td-weight 0.7 --draw-weight 0.3 \
    --spar-every 10 --spar-opponent greedy1 --spar-games 3 \
    --eval-every 50 --eval-games 10 --eval-opponent greedy1 \
    --save-dir models --save-name value_net.pth --keep-last 5
"""
import os
import csv
import time
import random
import argparse
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import chess

# ------------------------------
# 인코딩: UI와 1:1 동일
# ------------------------------
PIECE_TO_PLANE = {
    (chess.PAWN, True): 0,
    (chess.KNIGHT, True): 1,
    (chess.BISHOP, True): 2,
    (chess.ROOK, True): 3,
    (chess.QUEEN, True): 4,
    (chess.KING, True): 5,
    (chess.PAWN, False): 6,
    (chess.KNIGHT, False): 7,
    (chess.BISHOP, False): 8,
    (chess.ROOK, False): 9,
    (chess.QUEEN, False): 10,
    (chess.KING, False): 11,
}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        r = chess.square_rank(square); c = chess.square_file(square)
        planes[plane_idx, r, c] = 1.0
    extras = np.array([
        1.0 if board.turn == chess.WHITE else -1.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
    ], dtype=np.float32)
    feat = np.concatenate([planes.reshape(-1), extras], axis=0)  # 768 + 5 = 773
    return torch.from_numpy(feat)

# ------------------------------
# 모델: UI와 동일
# ------------------------------
class ValueNet(nn.Module):
    def __init__(self, in_dim: int = 773):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))

@torch.no_grad()
def evaluate_value(board: chess.Board, model: 'ValueNet', device: torch.device) -> float:
    x = board_to_tensor(board).to(device).unsqueeze(0)
    return float(model(x).squeeze(0).item())

def negamax(board: chess.Board, model: 'ValueNet', device: torch.device,
            depth: int, alpha: float, beta: float) -> float:
    outcome = board.outcome()
    if outcome is not None:
        if outcome.winner is None: return 0.0
        return 1.0 if (outcome.winner == board.turn) else -1.0
    if depth == 0:
        return evaluate_value(board, model, device)
    max_eval = -1e9
    for move in board.legal_moves:
        board.push(move)
        val = -negamax(board, model, device, depth - 1, -beta, -alpha)
        board.pop()
        if val > max_eval: max_eval = val
        if max_eval > alpha: alpha = max_eval
        if alpha >= beta: break
    return max_eval

@torch.no_grad()
def select_move(board: chess.Board, model: 'ValueNet', device: torch.device,
                depth: int = 2) -> Optional[chess.Move]:
    legal = list(board.legal_moves)
    if not legal: return None
    best_move, best_val = None, -1e9
    for mv in legal:
        board.push(mv)
        val = -negamax(board, model, device, max(depth - 1, 0), -1e9, 1e9)
        board.pop()
        if val > best_val:
            best_val, best_move = val, mv
    return best_move

# ------------------------------
# 학습/스파링/평가 보조
# ------------------------------
PIECE_VALUE = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0,
}
def random_move(board: chess.Board) -> Optional[chess.Move]:
    legal = list(board.legal_moves)
    return random.choice(legal) if legal else None

def material_score(board: chess.Board, color: bool) -> float:
    s_me = sum(PIECE_VALUE[p.piece_type] for p in board.piece_map().values() if p.color == color)
    s_opp = sum(PIECE_VALUE[p.piece_type] for p in board.piece_map().values() if p.color != color)
    return s_me - s_opp

def greedy1_move(board: chess.Board) -> Optional[chess.Move]:
    color_to_move = board.turn
    legal = list(board.legal_moves)
    if not legal: return None
    best, best_val = None, -1e18
    for mv in legal:
        board.push(mv)
        moved_color = not board.turn
        oc = board.outcome()
        if oc is not None and oc.winner == moved_color:
            board.pop(); return mv  # 즉시 메이트
        val = material_score(board, moved_color)
        board.pop()
        if val > best_val:
            best_val, best = val, mv
    return best

def choose_spar_move(board: chess.Board, opponent: str) -> Optional[chess.Move]:
    if opponent == "greedy1": return greedy1_move(board)
    if opponent == "random":  return random_move(board)
    return None

# ------------------------------
# Config
# ------------------------------
@dataclass
class Config:
    num_games: int = 200
    max_moves: int = 768
    opening_moves: int = 4
    depth: int = 2
    epsilon_start: float = 0.40
    epsilon_final: float = 0.05
    epsilon_decay_games: int = 500
    buffer_size: int = 250_000
    batch_size: int = 1024
    updates_per_game: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    seed: int = 42
    save_dir: str = "models"
    save_name: str = "value_net.pth"
    save_every: int = 20
    keep_last: int = 5        # numbered checkpoint 보관 개수
    device: str = "auto"
    use_amp: bool = True
    # TD/가중치/스파링
    td_gamma: float = 0.99
    td_weight: float = 0.7
    draw_weight: float = 0.3
    spar_every: int = 10
    spar_games: int = 3
    spar_opponent: str = "greedy1"  # greedy1|random|none
    # 평가(주기적으로 빠른 매치)
    eval_every: int = 0        # 0이면 비활성
    eval_games: int = 10
    eval_opponent: str = "greedy1"
    eval_depth: int = 2
    eval_opening_moves: int = 0
    eval_max_moves: int = 512

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device(arg: str) -> torch.device:
    if arg == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)

def epsilon_by_game(cfg: Config, game_idx: int) -> float:
    t = min(1.0, game_idx / max(1, cfg.epsilon_decay_games))
    return cfg.epsilon_start + t * (cfg.epsilon_final - cfg.epsilon_start)

def make_optimizer(model: nn.Module, cfg: Config):
    return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

# ------------------------------
# 체크포인트/로그 유틸
# ------------------------------
def _split_name(path: str):
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    return stem, ext

def save_latest(cfg: Config, model: nn.Module):
    os.makedirs(cfg.save_dir, exist_ok=True)
    latest_path = os.path.join(cfg.save_dir, cfg.save_name)
    torch.save(model.state_dict(), latest_path)
    print(f"[INFO] Saved latest: {latest_path}")
    return latest_path

def save_numbered(cfg: Config, model: nn.Module, game_idx: int):
    stem, ext = _split_name(cfg.save_name)
    path = os.path.join(cfg.save_dir, f"{stem}_g{game_idx:04d}{ext}")
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved numbered: {path}")
    return path

def save_best(cfg: Config, model: nn.Module):
    stem, ext = _split_name(cfg.save_name)
    best_path = os.path.join(cfg.save_dir, f"{stem}_best{ext}")
    torch.save(model.state_dict(), best_path)
    print(f"[INFO] Saved BEST: {best_path}")
    return best_path

def prune_numbered(cfg: Config):
    stem, ext = _split_name(cfg.save_name)
    prefix = os.path.join(cfg.save_dir, f"{stem}_g")
    files = [f for f in os.listdir(cfg.save_dir) if f.startswith(f"{stem}_g") and f.endswith(ext)]
    if len(files) <= cfg.keep_last: return
    files_full = [os.path.join(cfg.save_dir, f) for f in files]
    files_full.sort()  # g0001 < g0002 ...
    to_del = files_full[:-cfg.keep_last]
    for p in to_del:
        try:
            os.remove(p)
            print(f"[INFO] Pruned: {p}")
        except Exception as e:
            print(f"[WARN] Failed to prune {p}: {e}")

def append_log(save_dir: str, row: dict):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "train_log.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header: w.writeheader()
        w.writerow(row)

# ------------------------------
# 데이터 수집 (자기대국/스파링) + TD/MC targets
# ------------------------------
def play_game_collect(model: ValueNet, device: torch.device, depth: int,
                      epsilon: float, max_moves: int, opening_moves: int,
                      spar_opponent: Optional[str] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    board = chess.Board()
    states: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    side_to_move_list: List[bool] = []
    moves_played = 0

    for _ in range(opening_moves):
        if board.is_game_over(): break
        mv = random_move(board); 
        if mv is None: break
        board.push(mv)

    while not board.is_game_over() and moves_played < max_moves:
        states.append(board_to_tensor(board).numpy())
        side_to_move_list.append(board.turn == chess.WHITE)

        if spar_opponent is not None and ((board.turn == chess.WHITE) == True):
            mv = choose_spar_move(board, spar_opponent)
        else:
            use_random = (random.random() < epsilon)
            mv = random_move(board) if use_random else select_move(board, model, device, depth=depth)
        if mv is None: break

        board.push(mv); moves_played += 1
        next_states.append(board_to_tensor(board).numpy())

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        mc_targets = [0.0 for _ in states]
    else:
        mc_targets = []
        for wtm in side_to_move_list:
            mc_targets.append(1.0 if (outcome.winner == (chess.WHITE if wtm else chess.BLACK)) else -1.0)

    if len(next_states) < len(states):
        while len(next_states) < len(states):
            next_states.append(states[-1])

    return states, next_states, mc_targets

# ------------------------------
# 샘플링/업데이트
# ------------------------------
def sample_minibatch(buffer: deque, batch_size: int):
    n = min(batch_size, len(buffer))
    batch = random.sample(buffer, k=n)
    xs       = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    ys_mc    = torch.from_numpy(np.stack([b[1] for b in batch])).float().unsqueeze(1)
    xs_next  = torch.from_numpy(np.stack([b[2] for b in batch])).float()
    weights  = torch.from_numpy(np.stack([b[3] for b in batch])).float().unsqueeze(1)
    return xs, ys_mc, xs_next, weights

def train_step(model: ValueNet, optimizer, scaler, device,
               xs, ys_mc, xs_next, w, td_gamma: float, td_weight: float,
               grad_clip: float, use_amp: bool):
    xs = xs.to(device, non_blocking=True)
    ys_mc = ys_mc.to(device, non_blocking=True)
    xs_next = xs_next.to(device, non_blocking=True)
    w = w.to(device, non_blocking=True)

    loss_fn = nn.SmoothL1Loss(reduction='none')
    optimizer.zero_grad(set_to_none=True)

    if use_amp:
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            v = model(xs)
            loss_mc = loss_fn(v, ys_mc)
            if td_weight > 0.0:
                with torch.no_grad():
                    v_next = model(xs_next)
                td_target = td_gamma * (-v_next)
                loss_td = loss_fn(v, td_target)
                loss = (w * (loss_mc + td_weight * loss_td)).mean()
            else:
                loss = (w * loss_mc).mean()
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        v = model(xs)
        loss_mc = loss_fn(v, ys_mc)
        if td_weight > 0.0:
            with torch.no_grad():
                v_next = model(xs_next)
            td_target = td_gamma * (-v_next)
            loss_td = loss_fn(v, td_target)
            loss = (w * (loss_mc + td_weight * loss_td)).mean()
        else:
            loss = (w * loss_mc).mean()
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return float(loss.detach().cpu().item())

# ------------------------------
# 빠른 평가 (주기적으로 호출)
# ------------------------------
@torch.no_grad()
def quick_eval(model: ValueNet, device: torch.device, cfg: Config) -> dict:
    def play_one(model_is_white: bool) -> int:
        board = chess.Board()
        for _ in range(cfg.eval_opening_moves):
            if board.is_game_over(): break
            mv = random_move(board)
            if mv is None: break
            board.push(mv)
        plies = 0
        while not board.is_game_over() and plies < cfg.eval_max_moves:
            if (board.turn == chess.WHITE) == model_is_white:
                mv = select_move(board, model, device, cfg.eval_depth)
            else:
                mv = greedy1_move(board) if cfg.eval_opponent == "greedy1" else random_move(board)
            if mv is None: break
            board.push(mv); plies += 1
        outcome = board.outcome()
        if outcome is None or outcome.winner is None: return 0
        return 1 if ((outcome.winner == chess.WHITE) == model_is_white) else -1

    wins = losses = draws = 0
    for g in range(cfg.eval_games):
        res = play_one(model_is_white=(g % 2 == 0))
        if res > 0: wins += 1
        elif res < 0: losses += 1
        else: draws += 1
    total = max(1, wins + losses + draws)
    score = wins + 0.5 * draws
    return {
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": 100.0 * wins / total, "score_rate": 100.0 * score / total
    }

# ------------------------------
# 메인
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games", type=int, default=200)
    ap.add_argument("--max-moves", type=int, default=768)
    ap.add_argument("--opening-moves", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epsilon-start", type=float, default=0.40)
    ap.add_argument("--epsilon-final", type=float, default=0.05)
    ap.add_argument("--epsilon-decay-games", type=int, default=500)
    ap.add_argument("--buffer-size", type=int, default=250000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--updates-per-game", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--save-dir", type=str, default="models")
    ap.add_argument("--save-name", type=str, default="value_net.pth")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--keep-last", type=int, default=5)
    ap.add_argument("--resume", type=str, default="")
    # TD/가중치/스파링
    ap.add_argument("--td-gamma", type=float, default=0.99)
    ap.add_argument("--td-weight", type=float, default=0.7)
    ap.add_argument("--draw-weight", type=float, default=0.3)
    ap.add_argument("--spar-every", type=int, default=10)
    ap.add_argument("--spar-games", type=int, default=3)
    ap.add_argument("--spar-opponent", type=str, default="greedy1", choices=["greedy1", "random", "none"])
    # 평가 옵션
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-games", type=int, default=10)
    ap.add_argument("--eval-opponent", type=str, default="greedy1", choices=["greedy1", "random"])
    ap.add_argument("--eval-depth", type=int, default=2)
    ap.add_argument("--eval-opening-moves", type=int, default=0)
    ap.add_argument("--eval-max-moves", type=int, default=512)
    args = ap.parse_args()

    cfg = Config(
        num_games=args.num_games, max_moves=args.max_moves, opening_moves=args.opening_moves, depth=args.depth,
        epsilon_start=args.epsilon_start, epsilon_final=args.epsilon_final, epsilon_decay_games=args.epsilon_decay_games,
        buffer_size=args.buffer_size, batch_size=args.batch_size, updates_per_game=args.updates_per_game,
        lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip, seed=args.seed, device=args.device,
        use_amp=args.use_amp, save_dir=args.save_dir, save_name=args.save_name, save_every=args.save_every,
        keep_last=args.keep_last, td_gamma=args.td_gamma, td_weight=args.td_weight, draw_weight=args.draw_weight,
        spar_every=args.spar_every, spar_games=args.spar_games, spar_opponent=args.spar_opponent,
        eval_every=args.eval_every, eval_games=args.eval_games, eval_opponent=args.eval_opponent,
        eval_depth=args.eval_depth, eval_opening_moves=args.eval_opening_moves, eval_max_moves=args.eval_max_moves
    )

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"[INFO] Device: {device.type}")

    model = ValueNet().to(device)
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state); print(f"[INFO] Resumed: {args.resume}")

    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.use_amp and device.type == "cuda"))

    replay: deque = deque(maxlen=cfg.buffer_size)
    best_score_rate = -1.0  # 평가 최고치 추적

    pbar = trange(cfg.num_games, desc="Self-Play+Train", ncols=100)
    moving_loss = None
    seen_games = 0

    for g in pbar:
        eps = epsilon_by_game(cfg, g)

        # 자기대국
        states, next_states, mc_targets = play_game_collect(
            model, device, depth=cfg.depth, epsilon=eps,
            max_moves=cfg.max_moves, opening_moves=cfg.opening_moves,
            spar_opponent=None
        )
        for s, s_next, t in zip(states, next_states, mc_targets):
            w = cfg.draw_weight if abs(t) < 1e-6 else 1.0
            replay.append((s.astype(np.float32), np.float32(t), s_next.astype(np.float32), np.float32(w)))

        seen_games += 1

        # 스파링
        if cfg.spar_opponent != "none" and (seen_games % max(1, cfg.spar_every) == 0):
            for _ in range(cfg.spar_games):
                S, Snext, T = play_game_collect(
                    model, device, depth=cfg.depth, epsilon=0.0,
                    max_moves=cfg.max_moves, opening_moves=cfg.opening_moves,
                    spar_opponent=cfg.spar_opponent
                )
                for s, s_next, t in zip(S, Snext, T):
                    w = cfg.draw_weight if abs(t) < 1e-6 else 1.0
                    replay.append((s.astype(np.float32), np.float32(t), s_next.astype(np.float32), np.float32(w)))

        # 업데이트
        total_loss = 0.0; steps = 0
        if len(replay) >= max(256, cfg.batch_size):
            for _ in range(cfg.updates_per_game):
                xs, ys_mc, xs_next, w = sample_minibatch(replay, cfg.batch_size)
                l = train_step(model, optimizer, scaler, device,
                               xs, ys_mc, xs_next, w,
                               td_gamma=cfg.td_gamma, td_weight=cfg.td_weight,
                               grad_clip=cfg.grad_clip, use_amp=cfg.use_amp)
                total_loss += l; steps += 1
        if steps > 0:
            avg_loss = total_loss / steps
            moving_loss = avg_loss if moving_loss is None else (0.9 * moving_loss + 0.1 * avg_loss)

        # 진행바
        pbar.set_postfix({
            "eps": f"{eps:.2f}",
            "buf": len(replay),
            "loss": f"{(moving_loss if moving_loss is not None else 0.0):.4f}",
            "TDw": cfg.td_weight,
            "DW": cfg.draw_weight,
        })

        # 주기 저장
        if (g + 1) % cfg.save_every == 0 or (g + 1) == cfg.num_games:
            save_latest(cfg, model)
            num_path = save_numbered(cfg, model, g + 1)
            prune_numbered(cfg)

            # 주기 평가(옵션)
            eval_score = None
            if cfg.eval_every > 0 and ((g + 1) % cfg.eval_every == 0):
                model.eval()
                res = quick_eval(model, device, cfg)
                model.train()
                eval_score = res["score_rate"]
                print(f"[EVAL] g={g+1} vs {cfg.eval_opponent}: "
                      f"W/L/D={res['wins']}/{res['losses']}/{res['draws']} "
                      f"ScoreRate={res['score_rate']:.1f}% WinRate={res['win_rate']:.1f}%")
                # best 갱신
                if eval_score > best_score_rate:
                    best_score_rate = eval_score
                    save_best(cfg, model)

            # 로그 남기기
            append_log(cfg.save_dir, {
                "game": g + 1,
                "buffer": len(replay),
                "epsilon": round(eps, 4),
                "loss_ma": round(moving_loss if moving_loss is not None else 0.0, 6),
                "eval_score_rate": round(eval_score, 2) if eval_score is not None else "",
                "saved_numbered": os.path.basename(num_path),
            })

    # 마지막 저장
    save_latest(cfg, model)
    print("[INFO] Training finished.")

if __name__ == "__main__":
    main()
