from typing import List, Tuple, Dict
from hexapawn import HexapawnGame

# Modified copy of minimax search, which will store all possible game states
games_seen = {}
non_terminal_nodes = 0

def minimax_search(game: HexapawnGame):
    return max_value(game)

def max_value(game: HexapawnGame) -> Tuple[
    int,
    Dict[int, List[Tuple[int,int,int]]]
]:
    if game.is_terminal():
        return (game.utility(), None)
    global non_terminal_nodes
    non_terminal_nodes += 1
    policy_table = {}
    v = -1e9
    for a in game.actions:
        v_2, _ = min_value(HexapawnGame(game.result(a)))
        if v_2 > v:
            v = v_2
        policy_table.setdefault(v_2, []).append(a)
    games_seen[str(game.board)] = policy_table
    return v, policy_table

def min_value(game: HexapawnGame) -> Tuple[
    int,
    Dict[int, List[Tuple[int,int,int]]]
]:
    if game.is_terminal():
        return (game.utility(), None)
    global non_terminal_nodes
    non_terminal_nodes += 1
    policy_table = {}
    v = 1e9
    for a in game.actions:
        v_2, _ = max_value(HexapawnGame(game.result(a)))
        if v_2 < v:
            v = v_2
        policy_table.setdefault(v_2, []).append(a)
    games_seen[str(game.board)] = policy_table
    return v, policy_table

def unload_games():
    print(len(games_seen))
    print(non_terminal_nodes)
    with open("hexapawn_game_states.py", "w") as file:
        file.write("states = {\n")
        for game, policy in games_seen.items():
            file.write("\t")
            file.write('"' + game + '"' + " : ") 
            file.write(str(policy) + ",\n") 
        file.write("}")
    
minimax_search(HexapawnGame())
unload_games()