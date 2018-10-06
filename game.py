#!/usr/bin/env python

import numpy as np
from typing import Union, Tuple


class Othello(object):
    around = (
        (-1, -1), (-1, 0), (-1, 1),
        (0,  -1),          (0,  1),
        (1,  -1), (1,  0), (1,  1)
    )

    # Black: 1
    # White: 2
    # board[y, x]
    def __init__(self, n=8):
        assert n % 2 == 0

        self.n = n
        self.board = np.full((n, n), 0, dtype=np.uint8)

        self.board[n // 2][n // 2] = 1
        self.board[n // 2][n // 2 + 1] = 2
        self.board[n // 2 + 1][n // 2] = 2
        self.board[n // 2 + 1][n // 2 + 1] = 1

    @property
    def data(self) -> np.ndarray:
        black = self.board == 1
        white = self.board == 2
        return np.stack((black, white))

    def play(self, point: Tuple[int, int], player: int=1) -> bool:
        assert player == 1 or player == 2
        assert isinstance(point, tuple) or isinstance(point, list) and len(point) == 2

        next, is_success = self.next(self.board, point, player)
        if is_success:
            self.board = next
        return is_success

    def next(self, board, point, player) -> Tuple[np.ndarray, bool]:
        is_success = False
        if board[point[0]][point[1]] != 0:
            return board, False
        for dy, dx in self.around:
            y, x = point
            num = 0
            while all(0 < i < self.n for i in (y + dy, x + dx)):
                y += dy
                x += dx
                num += 1
                if board[y][x] == 0:
                    break
                elif board[y][x] == player:
                    for i, j in [(point[0] + dy * k, point[1] + dx * k) for k in range(1, num)]:
                        is_success = True
                        board[i][j] = player
                    break
        if is_success:
            board[point[0]][point[1]] = player
        return board, is_success

    def step(self, action: np.ndarray) -> Tuple[Union[None, np.ndarray], float, bool]:
        action = [i.item() for i in np.where(action)]
        if not self.play(action):
            return None, -1, True
        for point in [(i, j) for i in range(self.n) for j in range(self.n)]:
            if self.play(point, 2):
                break
        else:
            if action is None:
                return None, self.result, True
        return self.data, 0, False

    @property
    def result(self):
        black = np.sum(self.board == 1)
        white = np.sum(self.board == 2)
        if black > white:
            return 1
        else:
            return 0


def main():
    import pygame
    game = Othello()


if __name__ == '__main__':
    main()
