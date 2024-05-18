from typing import List, Tuple, Dict
import numpy as np
import copy

# 0th entry is turn --- -1 is min, 1 is max
# 1-9th entries are the board, in row major form
initial_board: np.ndarray = np.array([1, 
                                      -1, -1, -1, 
                                      0, 0, 0, 
                                      1, 1, 1])

class HexapawnGame:
    def __init__(self, board=initial_board):
        self.board = board 
        self.actions = self.compute_actions() 

    def compute_actions(self):
        actions = []
        for idx, value in enumerate(self.board[1:], start=1):
            if value != self.board[0] or value == 0:
                continue

            if self.board[0] == 1:  
                # Piece in back row --- Shouldn't happen, because this is terminal
                if idx < 4:
                    continue

                # Check forward move
                if idx - 3 >= 1 and self.board[idx - 3] == 0:
                    actions.append((0, idx, idx - 3))

                # Check diagonal moves
                if idx % 3 != 0:  # Not rightmost column
                    if self.board[idx - 2] == -1:
                        actions.append((1, idx, idx - 2))
                if idx % 3 != 1:  # Not leftmost column
                    if self.board[idx - 4] == -1:
                        actions.append((1, idx, idx - 4))

            else:
                if idx > 6:
                    continue

                if idx + 3 <= 9 and self.board[idx + 3] == 0:
                    actions.append((0, idx, idx + 3))

                if idx % 3 != 1:
                    if self.board[idx + 2] == 1:
                        actions.append((1, idx, idx + 2))
                if idx % 3 != 0: 
                    if self.board[idx + 4] == 1:
                        actions.append((1, idx, idx + 4))
        return actions 
            
    def result(self, action: Tuple[int, int, int]):
        new_board = copy.deepcopy(self.board)
        new_board[0] = 1 if self.board[0] == -1 else -1
        # Action type isn't really needed, but is nice for debugging
        _, initial, final = action
        new_board[final] = new_board[initial]
        new_board[initial] = 0
        return new_board

    def is_terminal(self):
        # Game ends if pawn is promoted, or if a player has no available actions 
        terminal = False
        if 1 in self.board[1:4]:
            terminal = True
        if -1 in self.board[7:10]:
            terminal = True
        if len(self.actions) == 0:
            terminal = True
        return terminal

    def utility(self):
        if 1 in self.board[1:4]:
            return 1
        elif -1 in self.board[7:10]:
            return -1
        else: 
            return 1 if self.board[0] == -1 else -1

def minimax_search(game: HexapawnGame):
    return max_value(game)

def max_value(game: HexapawnGame) -> Tuple[
    int,
    Dict[int, List[Tuple[int,int,int]]]
]:
    if game.is_terminal():
        return (game.utility(), None)
    policy_table = {}
    v = -1e9
    for a in game.actions:
        v_2, _ = min_value(HexapawnGame(game.result(a)))
        if v_2 > v:
            v = v_2
        policy_table.setdefault(v_2, []).append(a)
    return v, policy_table

def min_value(game: HexapawnGame) -> Tuple[
    int,
    Dict[int, List[Tuple[int,int,int]]]
]:
    if game.is_terminal():
        return (game.utility(), None)
    policy_table = {}
    v = 1e9
    for a in game.actions:
        v_2, _ = max_value(HexapawnGame(game.result(a)))
        if v_2 < v:
            v = v_2
        policy_table.setdefault(v_2, []).append(a)
    return v, policy_table

