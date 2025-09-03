#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Play로 체스 가치함수(Value Net)를 학습하는 간단한 강화학습 스크립트.
- python-chess로 규칙 처리
- 12x8x8 원-핫 피스 플레인 + 부가 피처로 상태 인코딩
- 출력: [-1, 1] 범위의 스칼라 가치(사이드 투 무브 관점)
- 얕은 negamax(+alpha-beta) 탐색으로 행동 선택
- 에피소드 종료 후 모든 스테이트에 최종 보상(z) 할당하여 MSE 학습

업데이트:
- 주기적으로 "현재 모델 vs 처음(초기) 모델" 평가전 진행
- 승/무/패, 점수율(승+0.5*무) 로그 출력 및 CSV 저장
- 학습 종료 시 손실/점수율 PNG 그래프 저장
"""
import argparse
import os
import random
import copy
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
from tqdm import trange

# 그래프 저장용
import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서도 저장 가능
import matplotlib.pyplot as plt

# ------------------------------
# 상태 인코딩
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
    """12x8x8 one-hot + 5 extras (side_to_move, castling rights 4) -> (773,)"""
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        r = chess.square_rank(square)
        c = chess.square_file(square)
        planes[plane_idx, r, c] = 1.0

    extras = np.array([
        1.0 if board.turn == chess.WHITE else -1.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
    ], dtype=np.float32)

    feat = np.concatenate([planes.reshape(-1), extras], axis=0)  # (12*64)+5=773
    return torch.from_numpy(feat)

# ------------------------------
# 모델
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
        return torch.tanh(self.net(x))  # [-1, 1]

# ------------------------------
# 탐색: 얕은 negamax + alpha-beta
# ------------------------------
@torch.no_grad()
def evaluate(board: chess.Board, model: ValueNet, device: torch.device) -> float:
    x = board_to_tensor(board).to(device).unsqueeze(0)
    v = model(x).squeeze(0).item()
    return float(v)

def negamax(board: chess.Board, model: ValueNet, device: torch.device,
            depth: int, alpha: float, beta: float) -> float:
    outcome = board.outcome()
    if outcome is not None:
        if outcome.winner is None:
            return 0.0
        return 1.0 if (outcome.winner == board.turn) else -1.0

    if depth == 0:
        return evaluate(board, model, device)

    max_eval = -1e9
    for move in board.legal_moves:
        board.push(move)
        val = -negamax(board, model, device, depth - 1, -beta, -alpha)
        board.pop()
        if val > max_eval:
            max_eval = val
        if max_eval > alpha:
            alpha = max_eval
        if alpha >= beta:
            break
    return max_eval

def select_move(board: chess.Board, model: ValueNet, device: torch.device,
                depth: int = 1, epsilon: float = 0.1) -> Optional[chess.Move]:
    legal = list(board.legal_moves)
    if not legal:
        return None
    if random.random() < epsilon:
        return random.choice(legal)

    best_move = None
    best_val = -1e9
    for mv in legal:
        board.push(mv)
        val = -negamax(board, model, device, max(depth - 1, 0), -1e9, 1e9)
        board.pop()
        if val > best_val:
            best_val = val
            best_move = mv
    return best_move or random.choice(legal)

# ------------------------------
# 데이터셋
# ------------------------------
class PositionDataset(Dataset):
    def __init__(self, feats: List[torch.Tensor], targets: List[float]):
        self.X = torch.stack(feats)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------
# 자기대국 생성
# ------------------------------
@dataclass
class GameSample:
    feat: torch.Tensor
    side: bool  # True: white to move, False: black to move

def play_self_play_game(model: ValueNet, device: torch.device,
                        search_depth: int, epsilon: float, max_moves: int
                        ) -> Tuple[List[GameSample], float]:
    board = chess.Board()
    samples: List[GameSample] = []

    for _ply in range(max_moves):
        if board.is_game_over():
            break
        samples.append(GameSample(board_to_tensor(board), board.turn))
        mv = select_move(board, model, device, depth=search_depth, epsilon=epsilon)
        if mv is None:
            break
        board.push(mv)

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        result_val_white = 0.0  # draw or unfinished
    else:
        result_val_white = 1.0 if outcome.winner == chess.WHITE else -1.0

    return samples, result_val_white

def assign_targets(samples: List[GameSample], result_val_white: float) -> List[float]:
    targets: List[float] = []
    for s in samples:
        if result_val_white == 0.0:
            targets.append(0.0)
        else:
            val = result_val_white if s.side is True else -result_val_white
            targets.append(val)
    return targets

# ------------------------------
# 평가전: 현재 모델 vs 초기 모델
# ------------------------------
@torch.no_grad()
def play_model_vs_model(model_a: ValueNet, model_b: ValueNet, device: torch.device,
                        depth_a: int, depth_b: int, max_moves: int, a_is_white: bool) -> int:
    """
    model_a vs model_b 한 판 대국.
    반환: model_a 관점 결과 (1=승, 0=무, -1=패)
    """
    board = chess.Board()
    turn_is_a = a_is_white  # 백이 a면 True부터 시작

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if turn_is_a:
            mv = select_move(board, model_a, device, depth=depth_a, epsilon=0.0)
        else:
            mv = select_move(board, model_b, device, depth=depth_b, epsilon=0.0)
        if mv is None:
            break
        board.push(mv)
        turn_is_a = not turn_is_a

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0
    winner_is_white = (outcome.winner == chess.WHITE)
    if (winner_is_white and a_is_white) or ((not winner_is_white) and (not a_is_white)):
        return 1
    else:
        return -1

@torch.no_grad()
def evaluate_against_initial(current: ValueNet, initial: ValueNet, device: torch.device,
                             eval_games: int, eval_depth: int, eval_max_moves: int) -> Tuple[int, int, int, float]:
    """
    현재 모델 vs 초기 모델 N판.
    반환: (wins, draws, losses, score_rate)  # score_rate = (승 + 0.5*무)/N
    """
    wins = draws = losses = 0
    for i in range(eval_games):
        a_white = (i % 2 == 0)  # 색 번갈아가며 공정성 확보
        res = play_model_vs_model(current, initial, device, eval_depth, eval_depth, eval_max_moves, a_white)
        if res == 1:
            wins += 1
        elif res == 0:
            draws += 1
        else:
            losses += 1
    score = (wins + 0.5 * draws) / max(1, eval_games)
    return wins, draws, losses, float(score)

# ------------------------------
# 학습 루프
# ------------------------------
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    model = ValueNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 초기(처음) 모델 스냅샷 고정본
    initial_model = ValueNet().to(device)
    initial_model.load_state_dict(copy.deepcopy(model.state_dict()))
    initial_model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    replay_feats: List[torch.Tensor] = []
    replay_tgts: List[float] = []

    # 로깅용 버퍼
    loss_history_steps: List[int] = []
    loss_history_vals: List[float] = []
    eval_points_games: List[int] = []
    eval_score_rates: List[float] = []
    eval_win_rates: List[float] = []  # wins / games

    # metrics.csv 헤더
    metrics_path = os.path.join(args.save_dir, args.metrics_file)
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["games_played", "wins", "draws", "losses", "win_rate", "score_rate"])

    # 학습
    for g in trange(args.games, desc="Self-Play Games"):
        eps = max(args.epsilon_final,
                  args.epsilon - (args.epsilon - args.epsilon_final) * (g / max(1, args.games - 1)))
        samples, result_val_white = play_self_play_game(
            model, device, args.depth, eps, args.max_moves
        )
        tgts = assign_targets(samples, result_val_white)
        replay_feats.extend([s.feat for s in samples])
        replay_tgts.extend(tgts)

        # 버퍼 크기 제한
        if len(replay_tgts) > args.replay_max:
            overflow = len(replay_tgts) - args.replay_max
            replay_feats = replay_feats[overflow:]
            replay_tgts = replay_tgts[overflow:]

        # 주기적으로 미니배치 학습
        if (g + 1) % args.train_every == 0 and len(replay_tgts) >= args.batch_size:
            ds = PositionDataset(replay_feats, replay_tgts)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
            model.train()
            for _ in range(args.epochs):
                for X, y in dl:
                    X = X.to(device)
                    y = y.to(device)
                    optimizer.zero_grad()
                    pred = model(X)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
                    # 손실 로그(스텝: 지금까지 본 게임 수 g 기준으로 기록)
                    loss_history_steps.append(g + 1)
                    loss_history_vals.append(loss.item())

        # 평가전: 현재 vs 초기
        if (g + 1) % args.eval_every == 0:
            model.eval()
            wins, draws, losses, score = evaluate_against_initial(
                current=model, initial=initial_model, device=device,
                eval_games=args.eval_games, eval_depth=args.eval_depth, eval_max_moves=args.eval_max_moves
            )
            win_rate = wins / max(1, args.eval_games)
            eval_points_games.append(g + 1)
            eval_score_rates.append(score)
            eval_win_rates.append(win_rate)

            # 콘솔 로그
            print(f"[Eval @ {g+1} games] W/D/L = {wins}/{draws}/{losses} | "
                  f"Win%={win_rate:.3f}, Score%={score:.3f}")

            # CSV 추가
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([g + 1, wins, draws, losses, f"{win_rate:.6f}", f"{score:.6f}"])

    # 최종 저장
    save_path = os.path.join(args.save_dir, args.save_name)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "in_dim": 773,
                "desc": "ValueNet tanh output in [-1,1], side-to-move perspective",
            },
        },
        save_path,
    )
    print(f"Saved model to {save_path}")

    # 그래프 저장
    plot_path = os.path.join(args.save_dir, args.plot_file)
    fig = plt.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Training Loss (per mini-batch)")
    ax1.set_xlabel("Games seen (approx.)")
    ax1.set_ylabel("MSE Loss")
    if len(loss_history_steps) > 0:
        ax1.plot(loss_history_steps, loss_history_vals)
    else:
        ax1.text(0.5, 0.5, "No loss logged yet", ha="center", va="center", transform=ax1.transAxes)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Score Rate vs Initial Model")
    ax2.set_xlabel("Games played")
    ax2.set_ylabel("Score rate (win + 0.5*draw)")
    if len(eval_points_games) > 0:
        ax2.plot(eval_points_games, eval_score_rates, marker="o")
        # 보조로 순수 승률도 점선으로
        ax2.plot(eval_points_games, eval_win_rates, linestyle="--")
        ax2.legend(["Score rate", "Win rate"], loc="best")
        ax2.set_ylim(0.0, 1.0)
    else:
        ax2.text(0.5, 0.5, "No evaluation yet", ha="center", va="center", transform=ax2.transAxes)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    print(f"Saved metrics CSV to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=300, help="Self-play 게임 수")
    parser.add_argument("--max-moves", type=int, default=200, help="한 게임 최대 수(플라이)")
    parser.add_argument("--depth", type=int, default=1, help="탐색 깊이 (1~2 권장)")
    parser.add_argument("--epsilon", type=float, default=0.20, help="초기 무작위 탐색 비율")
    parser.add_argument("--epsilon-final", dest="epsilon_final", type=float, default=0.05, help="최종 epsilon")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1, help="리플레이 학습시 epoch 수")
    parser.add_argument("--train-every", type=int, default=25, help="N게임마다 리플레이 학습")
    parser.add_argument("--replay-max", type=int, default=150_000, help="리플레이 버퍼 최대 샘플 수")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--save-name", type=str, default="value_net.pth")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    # 평가/로그 옵션
    parser.add_argument("--eval-every", type=int, default=50, help="N게임마다 초기 모델과 평가전")
    parser.add_argument("--eval-games", type=int, default=10, help="평가전 판수")
    parser.add_argument("--eval-depth", type=int, default=1, help="평가전 탐색 깊이")
    parser.add_argument("--eval-max-moves", type=int, default=150, help="평가전 한 판 최대 수(플라이)")
    parser.add_argument("--plot-file", type=str, default="training_progress.png", help="저장할 PNG 파일명")
    parser.add_argument("--metrics-file", type=str, default="metrics.csv", help="지표 저장 CSV 파일명")

    args = parser.parse_args()
    train(args)
