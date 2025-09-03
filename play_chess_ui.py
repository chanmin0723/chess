"""
python play_chess_ui.py
봇과 직접 플레이가 가능한 UI 실행 파이썬 파일
- 진행 방법: 말 클릭 --> 목적지 클릭
- 하이라이트: 선택 칸/합법적 목적지, 마지막 수
- 단축키: R(리셋), U(한 수 되돌리기), ESC(종료)
- 인자: --model, --bot-color, --depth, --flip, --device, --size
- 추가: --piece-style (unicode|images), --piece-dir (이모지 폴더)
"""
import argparse
import os
from typing import Optional, List
from functools import lru_cache

import pygame
import pygame.freetype
import chess
import torch
import torch.nn as nn
import numpy as np

# ------------------------------
# 모델 & 인코딩 (훈련 동일)
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
# 렌더링 자원
# ------------------------------
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HL = (246, 246, 105)
LAST = (186, 202, 68)
BG = (30, 30, 35)
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)

UNICODE_PIECES = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
}

def try_load_unicode_font(size: int) -> pygame.freetype.Font:
    """체스 유니코드 글리프(♔♕♖♗♘♙/♚♛♜♝♞♟)를 지원하는 폰트를 우선적으로 로드."""
    pygame.freetype.init()
    candidates = ["DejaVu Sans", "FreeSerif", "Noto Sans Symbols2", "Symbola", "Arial Unicode MS"]
    for name in candidates:
        path = pygame.font.match_font(name)
        if path:
            f = pygame.freetype.Font(path, size)
            try:
                if f.get_rect("♔").width > 0 and f.get_rect("♟").width > 0:
                    return f
            except Exception:
                pass
    return pygame.freetype.SysFont(None, size)

@lru_cache(maxsize=1)
def load_piece_images(piece_dir: str, square_size: int) -> dict:
    """PNG 말 아이콘 로드 후 정사각형으로 스케일링."""
    names = {
        'P':'wP','N':'wN','B':'wB','R':'wR','Q':'wQ','K':'wK',
        'p':'bP','n':'bN','b':'bB','r':'bR','q':'bQ','k':'bK',
    }
    imgs = {}
    for sym, key in names.items():
        path = os.path.join(piece_dir, f"{key}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"말 아이콘 파일을 찾을 수 없습니다: {path}")
        img = pygame.image.load(path).convert_alpha()
        imgs[sym] = pygame.transform.smoothscale(img, (square_size, square_size))
    return imgs

def draw_board(screen, board: chess.Board, square_size: int,
               selected: int, legal_sqs: List[int],
               last_move: Optional[chess.Move], font_or_none, piece_images, flip: bool):
    screen.fill(BG)
    # 바둑판
    for r in range(8):
        for c in range(8):
            file = c
            rank = 7 - r
            sq = chess.square(file, rank)
            if flip:
                sq = chess.square(7 - file, 7 - rank)
            color = LIGHT if (r + c) % 2 == 0 else DARK
            pygame.draw.rect(screen, color, (c * square_size, r * square_size, square_size, square_size))
            if selected == sq:
                pygame.draw.rect(screen, HL, (c * square_size, r * square_size, square_size, square_size), 0)
            if sq in legal_sqs:
                # 목적지 점 표시(중앙 작은 점)
                pygame.draw.circle(screen, BLACK,
                                   (c * square_size + square_size // 2, r * square_size + square_size // 2), 7)

    # 마지막 수 하이라이트(테두리)
    if last_move is not None:
        for sq in [last_move.from_square, last_move.to_square]:
            rf = chess.square_file(sq)
            rr = 7 - chess.square_rank(sq)
            if flip:
                rf = 7 - rf
                rr = 7 - rr
            pygame.draw.rect(screen, LAST, (rf * square_size, rr * square_size, square_size, square_size), 4)

    # 말
    for sq, piece in board.piece_map().items():
        rf = chess.square_file(sq)
        rr = 7 - chess.square_rank(sq)
        if flip:
            rf = 7 - rf
            rr = 7 - rr
        rect = pygame.Rect(rf * square_size, rr * square_size, square_size, square_size)
        sym = piece.symbol()

        if piece_images is not None:
            # PNG 아이콘
            img = piece_images.get(sym)
            if img is not None:
                screen.blit(img, rect.topleft)
        else:
            # 유니코드 글리프
            glyph = UNICODE_PIECES.get(sym, sym)
            text_surf, _ = font_or_none.render(glyph, WHITE)
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)

def square_at_mouse(mx, my, square_size: int, flip: bool) -> Optional[int]:
    c = mx // square_size
    r = my // square_size
    if not (0 <= c < 8 and 0 <= r < 8):
        return None
    file = c
    rank = 7 - r
    sq = chess.square(file, rank)
    if flip:
        sq = chess.square(7 - file, 7 - rank)
    return sq

def legal_dests_for(board: chess.Board, src: int) -> List[int]:
    return [m.to_square for m in board.legal_moves if m.from_square == src]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/value_net.pth")
    parser.add_argument("--bot-color", type=str, default="black", choices=["white", "black"])  # 사람은 반대색
    parser.add_argument("--depth", type=int, default=2, help="봇 탐색 깊이")
    parser.add_argument("--flip", action="store_true", help="흑 시점으로 보드 뒤집기")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--size", type=int, default=640, help="보드 픽셀 크기")

    # 추가 인자: 말 렌더링 방식
    parser.add_argument("--piece-style", type=str, default="images",
                        choices=["unicode", "images"], help="말 렌더링 방식 선택")
    parser.add_argument("--piece-dir", type=str, default="images",
                        help="PNG 말 아이콘 폴더 (wK.png, bQ.png 등)")

    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # 모델 로드
    model = ValueNet().to(device)
    if os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded model: {args.model}")
    else:
        print(f"[경고] 모델 파일을 찾을 수 없습니다: {args.model} — 무학습 가중치로 플레이합니다(매우 약함).")
    # model.eval()

    human_is_white = (args.bot_color.lower() == "black")

    pygame.init()
    pygame.display.set_caption("RL Chess — Human vs Bot")
    size = args.size
    square = size // 8
    screen = pygame.display.set_mode((size, size))

    # 말 렌더링 리소스 준비
    piece_images = None
    font = None
    if args.piece_style == "images":
        try:
            piece_images = load_piece_images(args.piece_dir, square)
        except Exception as e:
            print(f"[경고] 이미지 로드 실패: {e}\n→ 유니코드 렌더링으로 대체합니다.")
            font = try_load_unicode_font(int(square * 0.7))
    else:
        font = try_load_unicode_font(int(square * 0.7))

    board = chess.Board()
    selected_sq: Optional[int] = None
    legal_sqs: List[int] = []
    last_move: Optional[chess.Move] = None

    running = True
    clock = pygame.time.Clock()

    @torch.no_grad()
    def bot_move_if_needed():
        nonlocal board, last_move
        if board.is_game_over():
            return
        bot_to_move = (board.turn == chess.WHITE) if not human_is_white else (board.turn == chess.BLACK)
        if bot_to_move:
            mv = select_move(board, model, device, depth=args.depth)
            if mv is not None:
                board.push(mv)
                last_move = mv

    # 봇이 백이면 선수로 두게 함
    if not human_is_white:
        bot_move_if_needed()

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    board.reset()
                    selected_sq = None
                    legal_sqs = []
                    last_move = None
                    if not human_is_white:
                        bot_move_if_needed()
                elif event.key == pygame.K_u:
                    # 사람-봇 한 턴(2수) 되돌리기
                    if board.move_stack:
                        board.pop()
                        if board.move_stack:
                            board.pop()
                        selected_sq = None
                        legal_sqs = []
                        last_move = board.move_stack[-1] if board.move_stack else None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if board.is_game_over():
                    continue
                human_to_move = (board.turn == chess.WHITE) if human_is_white else (board.turn == chess.BLACK)
                if not human_to_move:
                    continue

                mx, my = pygame.mouse.get_pos()
                sq = square_at_mouse(mx, my, square, args.flip)
                if sq is None:
                    continue

                if selected_sq is None:
                    # 말 선택: 내 말만 선택
                    piece = board.piece_at(sq)
                    if piece is None:
                        continue
                    if (piece.color and not human_is_white) or ((not piece.color) and human_is_white):
                        continue
                    selected_sq = sq
                    legal_sqs = legal_dests_for(board, sq)
                else:
                    # 목적지 선택
                    move = chess.Move(from_square=selected_sq, to_square=sq)
                    # 프로모션(자동 퀸)
                    piece = board.piece_at(selected_sq)
                    if piece and piece.piece_type == chess.PAWN:
                        to_rank = chess.square_rank(sq)
                        if (board.turn == chess.WHITE and to_rank == 7) or (board.turn == chess.BLACK and to_rank == 0):
                            move = chess.Move(selected_sq, sq, promotion=chess.QUEEN)

                    if move in board.legal_moves:
                        board.push(move)
                        last_move = move
                        selected_sq = None
                        legal_sqs = []
                        # 봇 응수
                        bot_move_if_needed()
                    else:
                        # 다시 선택(같은 색 말 클릭 시)
                        piece = board.piece_at(sq)
                        if piece is not None and ((piece.color and human_is_white) or ((not piece.color) and (not human_is_white))):
                            selected_sq = sq
                            legal_sqs = legal_dests_for(board, sq)
                        else:
                            selected_sq = None
                            legal_sqs = []

        # 렌더링
        draw_board(screen, board, square,
                   selected_sq if selected_sq is not None else -1,
                   legal_sqs, last_move,
                   font if piece_images is None else None,
                   piece_images, args.flip)

        # 상태 텍스트
        if board.is_game_over():
            outcome = board.outcome()
            msg = "Draw" if (outcome is None or outcome.winner is None) else ("White wins" if outcome.winner == chess.WHITE else "Black wins")
        else:
            msg = "Your turn" if ((board.turn == chess.WHITE) == human_is_white) else "Bot thinking..."

        pygame.draw.rect(screen, BG, (0, 0, max(160, square * 4), 24))
        try:
            label_font = pygame.freetype.SysFont(None, 18)
        except Exception:
            label_font = font if font is not None else pygame.freetype.SysFont(None, 18)
        surf, _ = label_font.render(msg, WHITE)
        screen.blit(surf, (8, 4))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
