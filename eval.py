# eval.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
저장된 ValueNet을 기본 봇과 맞붙여 성능 평가 + 드로우 원인 집계 + (옵션) 드로우 게임 PGN 저장.

예)
  python eval.py --model models/value_net.pth --opponent greedy1 --games 50 \
                 --depth 2 --opening-moves 0 --max-moves 768 --device auto \
                 --save-draw-pgns draws --save-first-n-draws 10
"""
import argparse
import os
import random
from collections import Counter
from typing import Optional, Tuple, List

import numpy as np
import chess
import chess.pgn
import torch
import torch.nn as nn
from tqdm import trange

# ------------------------------
# 인코딩 / 모델 / 탐색 (UI와 동일)
# ------------------------------
PIECE_TO_PLANE = {
    (chess.PAWN, True): 0,  (chess.KNIGHT, True): 1, (chess.BISHOP, True): 2,
    (chess.ROOK, True): 3,  (chess.QUEEN, True): 4,  (chess.KING, True): 5,
    (chess.PAWN, False): 6, (chess.KNIGHT, False): 7, (chess.BISHOP, False): 8,
    (chess.ROOK, False): 9, (chess.QUEEN, False): 10, (chess.KING, False): 11,
}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        r = chess.square_rank(sq); c = chess.square_file(sq)
        planes[plane_idx, r, c] = 1.0
    extras = np.array([
        1.0 if board.turn == chess.WHITE else -1.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
    ], dtype=np.float32)
    feat = np.concatenate([planes.reshape(-1), extras], axis=0)
    return torch.from_numpy(feat)

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
def evaluate(board: chess.Board, model: ValueNet, device: torch.device) -> float:
    x = board_to_tensor(board).to(device).unsqueeze(0)
    return float(model(x).squeeze(0).item())

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

@torch.no_grad()
def select_move(board: chess.Board, model: ValueNet, device: torch.device,
                depth: int = 2,
                rep_penalty: float = 0.03,      # 3회/5회 반복 청구 가능 상태 패널티
                fifty_coeff: float = 0.001,     # 50수 규칙 임박 패널티 계수(halfmove_clock 기준)
                reset_bonus: float = 0.01       # 50수 카운터 리셋(잡거나 폰 이동) 보너스
                ) -> Optional[chess.Move]:
    legal = list(board.legal_moves)
    if not legal:
        return None

    best_move = None
    best_val = -1e9

    for mv in legal:
        # 50수 카운터 리셋 여부(미리 계산)
        resets_50 = board.is_capture(mv) or (board.piece_type_at(mv.from_square) == chess.PAWN)

        board.push(mv)
        val = -negamax(board, model, device, max(depth - 1, 0), -1e9, 1e9)

        # 반복/50수 규칙 위험 패널티
        pen = 0.0
        # 반복 청구 가능 시(3회/5회) 약한 패널티
        if board.can_claim_threefold_repetition():
            pen -= rep_penalty
        # 50수 규칙 임박 시(예: 90 이상부터 선형 패널티)
        if board.halfmove_clock >= 90:
            pen -= (board.halfmove_clock - 90) * fifty_coeff
        # 스테일메이트 직전 같은 건 큰 패널티
        if board.is_stalemate():
            pen -= 0.2

        # 50수 카운터 리셋이면 소폭 보너스(무승부 루프 탈출 유도)
        if resets_50:
            pen += reset_bonus

        score = val + pen
        board.pop()

        if score > best_val:
            best_val = score
            best_move = mv

    return best_move


# ------------------------------
# 기본 봇들 (랜덤 / 그리디 1-ply)
# ------------------------------
PIECE_VALUE = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0,
}
def material_score(board: chess.Board, color: bool) -> float:
    s_me = sum(PIECE_VALUE[p.piece_type] for p in board.piece_map().values() if p.color == color)
    s_opp = sum(PIECE_VALUE[p.piece_type] for p in board.piece_map().values() if p.color != color)
    return s_me - s_opp

def greedy1_move(board: chess.Board) -> Optional[chess.Move]:
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

def random_move(board: chess.Board) -> Optional[chess.Move]:
    legal = list(board.legal_moves)
    return random.choice(legal) if legal else None

# ------------------------------
# 드로우 원인 판별
# ------------------------------
def draw_reason(board: chess.Board, outcome: Optional[chess.Outcome], reached_max_moves: bool) -> str:
    """
    python-chess 기준:
      - 자동 드로우: STALEMATE, INSUFFICIENT_MATERIAL, SEVENTYFIVE_MOVES, FIVEFOLD_REPETITION
      - claimable(청구 가능): can_claim_threefold_repetition(), can_claim_fifty_moves()
      - 우리가 max_moves 제한으로 멈춘 경우는 별도 표시
    """
    if reached_max_moves and (outcome is None or outcome.winner is None):
        # 수제한으로 멈춤
        # 끝 시점에 '청구 가능' 상태였다면 그 이유도 덧붙여 주자
        claim_50 = board.can_claim_fifty_moves()
        claim_3f = board.can_claim_threefold_repetition()
        if claim_50 and claim_3f:
            return "MAX_MOVES_LIMIT (CLAIMABLE_50 & CLAIMABLE_3FOLD)"
        if claim_50:
            return "MAX_MOVES_LIMIT (CLAIMABLE_50)"
        if claim_3f:
            return "MAX_MOVES_LIMIT (CLAIMABLE_3FOLD)"
        return "MAX_MOVES_LIMIT"

    if outcome is None:
        return "UNKNOWN_DRAW"  # 정상적으로는 잘 안옴

    if outcome.winner is not None:
        return "NOT_A_DRAW"

    term = outcome.termination
    # Enum 이름을 문자열로
    try:
        return str(term.name)
    except Exception:
        return str(term)

# ------------------------------
# 한 판 플레이 (드로우 원인/PGN용으로 수 기록)
# ------------------------------
def play_one(model: ValueNet, device: torch.device, depth: int, opponent: str,
             model_is_white: bool, opening_moves: int, max_moves: int
            ) -> Tuple[int, Optional[chess.Outcome], str, List[chess.Move]]:
    board = chess.Board()
    moves: List[chess.Move] = []

    # 오프닝 다양화
    for _ in range(opening_moves):
        if board.is_game_over(): break
        mv = random_move(board)
        if mv is None: break
        board.push(mv); moves.append(mv)

    # 본게임
    plies = 0
    while not board.is_game_over() and plies < max_moves:
        if (board.turn == chess.WHITE) == model_is_white:
            mv = select_move(board, model, device, depth)
        else:
            mv = greedy1_move(board) if opponent == "greedy1" else random_move(board)
        if mv is None:
            break
        board.push(mv); moves.append(mv)
        plies += 1

    outcome = board.outcome()
    reached_max = (plies >= max_moves) and (outcome is None or outcome.winner is None)
    reason = draw_reason(board, outcome, reached_max)

    # 결과를 "모델 관점"으로 매핑
    if outcome is None or outcome.winner is None:
        return 0, outcome, reason, moves
    win_color = outcome.winner
    res = 1 if (win_color == chess.WHITE) == model_is_white else -1
    return res, outcome, reason, moves

def moves_to_pgn(moves: List[chess.Move], result_string: str, event_note: str) -> str:
    game = chess.pgn.Game()
    game.headers["Event"] = event_note
    game.headers["Result"] = result_string
    node = game
    board = chess.Board()
    for mv in moves:
        node = node.add_variation(mv)
        board.push(mv)
    return str(game)

# ------------------------------
# Elo 추정
# ------------------------------
def elo_from_score(score_per_game: float) -> float:
    import math
    p = max(1e-3, min(1 - 1e-3, score_per_game))
    return -400.0 * math.log10(1.0 / p - 1.0)

# ------------------------------
# 메인
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/value_net.pth")
    ap.add_argument("--opponent", type=str, default="greedy1", choices=["random", "greedy1"])
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--opening-moves", type=int, default=0)
    ap.add_argument("--max-moves", type=int, default=768)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=123)

    # NEW: 드로우 디버깅/저장
    ap.add_argument("--save-draw-pgns", type=str, default="", help="드로우 난 게임을 이 폴더에 PGN으로 저장")
    ap.add_argument("--save-first-n-draws", type=int, default=0, help="드로우 게임 중 앞의 N개만 저장(0=모두 저장 안 함)")

    args = ap.parse_args()

    # 시드/디바이스
    random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    print(f"[INFO] Device: {device.type}")

    # 모델 로드
    model = ValueNet().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state); model.eval()
    print(f"[INFO] Loaded model: {args.model}")

    # 저장 폴더 준비
    save_draws = (args.save_draw_pgns != "" and args.save_first_n_draws != 0)
    if save_draws:
        os.makedirs(args.save_draw_pgns, exist_ok=True)
    saved_draw_count = 0

    # 매치
    wins = losses = draws = 0
    draw_reasons = Counter()

    pbar = trange(args.games, ncols=100, desc=f"Eval vs {args.opponent}")
    for g in pbar:
        model_is_white = (g % 2 == 0)  # 교대
        res, outcome, reason, moves = play_one(
            model, device, depth=args.depth, opponent=args.opponent,
            model_is_white=model_is_white,
            opening_moves=args.opening_moves, max_moves=args.max_moves
        )

        if res > 0: wins += 1
        elif res < 0: losses += 1
        else:
            draws += 1
            draw_reasons[reason] += 1

            if save_draws and (saved_draw_count < (args.save_first_n_draws if args.save_first_n_draws > 0 else 999999)):
                # 결과 문자열
                result_string = "1/2-1/2"
                event_note = f"draw_reason={reason}"
                pgn = moves_to_pgn(moves, result_string, event_note)
                out_path = os.path.join(args.save_draw_pgns, f"draw_g{g+1:03d}.pgn")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(pgn)
                saved_draw_count += 1

        total = wins + losses + draws
        score = wins + 0.5 * draws
        wr = 100.0 * wins / max(1, total)
        sr = 100.0 * score / max(1, total)
        pbar.set_postfix(W=wins, L=losses, D=draws, WinRate=f"{wr:.1f}%", ScoreRate=f"{sr:.1f}%")

    # 요약
    total = max(1, wins + losses + draws)
    score = wins + 0.5 * draws
    p = score / total
    dE = elo_from_score(p)
    print("\n========== Evaluation Summary ==========")
    print(f"Opponent     : {args.opponent}")
    print(f"Games        : {total}  (opening-moves={args.opening_moves}, depth={args.depth}, max-moves={args.max_moves})")
    print(f"W/L/D        : {wins}/{losses}/{draws}")
    print(f"Score Rate   : {100.0*p:.2f}% (win=1, draw=0.5, loss=0)")
    print(f"Elo diff(~)  : {dE:+.1f} vs {args.opponent} (로지스틱 근사)")
    print(f"Draw breakdown: {dict(draw_reasons)}")
    if save_draws:
        print(f"Saved {saved_draw_count} draw games to: {os.path.abspath(args.save_draw_pgns)}")

if __name__ == "__main__":
    main()
