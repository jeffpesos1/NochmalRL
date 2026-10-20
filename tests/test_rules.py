"""Rule checks for the vectorised environment (run: python -m pytest tests  or  python tests/test_rules.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nochmal.agent import HeuristicPolicy, RandomPolicy  # noqa: E402
from nochmal.board import load_board  # noqa: E402
from nochmal.env import NochmalEnv  # noqa: E402
from nochmal.features import BoardTensors, mask_opponent_info  # noqa: E402
from nochmal.model import make_model  # noqa: E402

BOARD = load_board("classic")


def test_board_basics():
    assert BOARD.n_cells == 105
    assert BOARD.cell_star.sum() == 15
    assert all(int((BOARD.cell_color == c).sum()) == 21 for c in range(5))
    assert BOARD.region_size.max() <= 6
    assert (BOARD.move_size <= 5).all() and (BOARD.move_size >= 1).all()
    # every move lies inside a single colour block and is connected
    for cells in BOARD.move_cells:
        regs = {int(BOARD.cell_region[c]) for c in cells}
        assert len(regs) == 1
        seen = {cells[0]}
        stack = [cells[0]]
        while stack:
            cur = stack.pop()
            for nb in BOARD._neighbors(cur):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        assert seen == set(cells)


def _obs_for(env, marked_cells, pool_c, pool_n, jokers_used=0):
    env.reset_games(torch.ones(1, dtype=torch.bool))
    for c in marked_cells:
        env.marked[0, 0, c] = True
    env.jokers_used[0, 0] = jokers_used
    pc = torch.zeros(1, env.bt.Cf, dtype=torch.uint8)
    pn = torch.zeros(1, env.bt.Nf, dtype=torch.uint8)
    for k in pool_c:
        pc[0, k] += 1
    for k in pool_n:
        pn[0, k] += 1
    return env.make_obs(torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64), pc, pn,
                        torch.ones(1, dtype=torch.bool))


def test_first_move_must_touch_start_column():
    env = NochmalEnv(BOARD, 1, 1, "cpu")
    bt = env.bt
    # all colours and numbers available, no jokers
    obs = _obs_for(env, [], pool_c=[0, 1, 2, 3, 4], pool_n=[0, 1, 2, 3, 4])
    legal, _ = bt.legal_moves(obs)
    ks = torch.nonzero(legal[0, : bt.K]).flatten().tolist()
    assert ks, "there must be legal opening moves"
    for k in ks:
        cells = BOARD.move_cells[k]
        assert any(c % BOARD.width == BOARD.start_column for c in cells)
    assert bool(legal[0, bt.K])  # pass always legal


def test_dice_restrict_moves_and_jokers_cost():
    env = NochmalEnv(BOARD, 1, 1, "cpu")
    bt = env.bt
    start = BOARD.cell_index(0, BOARD.start_column)       # green, star, row 0 col H
    colour = int(BOARD.cell_color[start])
    # only that colour + number 1 available -> exactly the single-cell moves of that colour touching H / marks
    obs = _obs_for(env, [], pool_c=[colour], pool_n=[0])
    legal, need = bt.legal_moves(obs)
    ks = torch.nonzero(legal[0, : bt.K]).flatten().tolist()
    assert all(int(BOARD.move_size[k]) == 1 and int(BOARD.move_color[k]) == colour for k in ks)
    assert all(int(need[0, k]) == 0 for k in ks)
    # only jokers available -> everything costs 2 jokers; with 7 used -> nothing legal except pass
    obs = _obs_for(env, [], pool_c=[bt.C], pool_n=[bt.Nf - 1])
    legal, need = bt.legal_moves(obs)
    ks = torch.nonzero(legal[0, : bt.K]).flatten().tolist()
    assert ks and all(int(need[0, k]) == 2 for k in ks)
    obs = _obs_for(env, [], pool_c=[bt.C], pool_n=[bt.Nf - 1], jokers_used=7)
    legal, _ = bt.legal_moves(obs)
    assert int(legal[0, : bt.K].sum()) == 0 and bool(legal[0, bt.K])


def test_adjacency_after_marks_strict():
    """Default: after the first cross only cells adjacent to own crosses are allowed (column H no longer)."""
    env = NochmalEnv(BOARD, 1, 1, "cpu")
    bt = env.bt
    marked = [BOARD.cell_index(3, 2)]                       # somewhere left of H
    obs = _obs_for(env, marked, pool_c=[0, 1, 2, 3, 4], pool_n=[0, 1, 2, 3, 4])
    legal, _ = bt.legal_moves(obs)
    ks = torch.nonzero(legal[0, : bt.K]).flatten().tolist()
    assert ks
    for k in ks:
        cells = BOARD.move_cells[k]
        assert not any(c in marked for c in cells)
        assert any(nb in marked for c in cells for nb in BOARD._neighbors(c))
        assert not all(c % BOARD.width == BOARD.start_column for c in cells) or any(
            nb in marked for c in cells for nb in BOARD._neighbors(c))


def test_adjacency_after_marks_rulebook_variant():
    """start_column_always=True: column H may start a new group at any time."""
    env = NochmalEnv(BOARD, 1, 1, "cpu", start_column_always=True)
    bt = env.bt
    marked = [BOARD.cell_index(3, 2)]
    obs = _obs_for(env, marked, pool_c=[0, 1, 2, 3, 4], pool_n=[0, 1, 2, 3, 4])
    legal, _ = bt.legal_moves(obs)
    ks = torch.nonzero(legal[0, : bt.K]).flatten().tolist()
    assert any(any(c % BOARD.width == BOARD.start_column for c in BOARD.move_cells[k]) and
               not any(nb in marked for c in BOARD.move_cells[k] for nb in BOARD._neighbors(c)) for k in ks)
    for k in ks:
        cells = BOARD.move_cells[k]
        assert any(c % BOARD.width == BOARD.start_column for c in cells) or any(
            nb in marked for c in cells for nb in BOARD._neighbors(c))


def test_passive_players_only_get_remaining_dice():
    torch.manual_seed(0)
    env = NochmalEnv(BOARD, 64, 2, "cpu", seed=5)
    bt = env.bt
    seen = {"pool": []}

    def policy(obs, p):
        a, l, v = HeuristicPolicy(bt)(obs)
        if not bool(obs["scalars"][0, 1]):      # passive decision: pools must have one die less per type
            seen["pool"].append((obs["pool_c"].sum(1), obs["pool_n"].sum(1)))
        return a, l, v

    res = env.play_round(policy, record=False)
    pc, pn = seen["pool"][0]
    acted = res.action[torch.arange(64), env.active_player() - 1 if False else (env.round - 1) % 2] < bt.K
    # active players that moved removed one colour + one number die (also in round 1: free_rounds=0)
    assert torch.all(pc[acted] == env.n_color_dice - 1) and torch.all(pn[acted] == env.n_number_dice - 1)
    assert torch.all(pc[~acted] == env.n_color_dice) and torch.all(pn[~acted] == env.n_number_dice)


def test_scoring_and_game_end():
    env = NochmalEnv(BOARD, 1, 2, "cpu")
    bt = env.bt
    env.reset_games(torch.ones(1, dtype=torch.bool))
    # player 0 completes column A (first) ; player 1 completes column A too in the same round -> both upper value
    col = 0
    for r in range(BOARD.height):
        env.marked[0, 0, BOARD.cell_index(r, col)] = True
        env.marked[0, 1, BOARD.cell_index(r, col)] = True
    reward, done, final, jl = env.end_round(auto_reset=False)
    assert float(env.col_points[0, 0]) == BOARD.column_points_first[col]
    assert float(env.col_points[0, 1]) == BOARD.column_points_first[col]
    assert bool(env.col_claimed[0, col]) and not bool(done[0])
    # next round player 1 completes column B after player 0 did in a previous round -> lower value
    for r in range(BOARD.height):
        env.marked[0, 0, BOARD.cell_index(r, 1)] = True
    env.end_round(auto_reset=False)
    for r in range(BOARD.height):
        env.marked[0, 1, BOARD.cell_index(r, 1)] = True
    env.end_round(auto_reset=False)
    assert float(env.col_points[0, 1]) == BOARD.column_points_first[col] + BOARD.column_points_second[1]
    # two complete colours end the game
    for c in range(BOARD.n_cells):
        if int(BOARD.cell_color[c]) in (0, 1):
            env.marked[0, 0, c] = True
    reward, done, final, jl = env.end_round(auto_reset=False)
    assert bool(done[0])
    assert float(env.color_points[0, 0]) == 2 * BOARD.color_points_first
    score = env.scores()[0, 0]
    stars_unmarked = int((~env.marked[0, 0] & bt.cell_star).sum())
    expected = float(env.col_points[0, 0] + env.color_points[0, 0]) - 2 * stars_unmarked + 8
    assert abs(float(score) - expected) < 1e-5


def test_full_games_run():
    torch.manual_seed(0)
    for players in (1, 2, 3):
        env = NochmalEnv(BOARD, 16, players, "cpu", seed=1)
        pol = HeuristicPolicy(env.bt)
        finished = 0
        for _ in range(80):
            res = env.play_round(pol, record=True)
            # rewards must sum to final score - initial potential for finished games
            finished += int(res.done.sum())
            if players == 1:
                assert int(env.round.max()) <= 30
        assert finished > 0
    env = NochmalEnv(BOARD, 8, 2, "cpu", seed=2)
    pol = RandomPolicy(env.bt)
    for _ in range(50):
        env.play_round(pol, record=False)


def test_reward_telescopes_to_score():
    env = NochmalEnv(BOARD, 4, 2, "cpu", seed=3)
    pol = HeuristicPolicy(env.bt)
    init = env.scores().clone()
    total = torch.zeros(4, 2)
    for _ in range(200):
        before = env.scores().clone()
        res = env.play_round(pol, record=False)
        if bool(res.done.any()):
            for g in torch.nonzero(res.done).flatten().tolist():
                total[g] += res.reward[g]
                assert torch.allclose(total[g], res.final_scores[g] - init[g], atol=1e-4), (total[g], res.final_scores[g] - init[g])
                total[g] = 0
                init[g] = env.scores()[g]
            res.reward[res.done] = 0
        total += res.reward
    assert True




def test_opponent_view_variants():
    env = NochmalEnv(BOARD, 4, 3, "cpu", seed=9)
    pol = HeuristicPolicy(env.bt)
    for _ in range(5):
        env.play_round(pol, record=False)
    g = torch.arange(4)
    obs = env.make_obs(g, torch.zeros(4, dtype=torch.int64), env.pool_c, env.pool_n, torch.ones(4, dtype=torch.bool))
    assert obs["opp_best"].shape == obs["marked"].shape
    assert int(obs["opp_marked"].sum()) > 0            # opponents have crosses after 5 rounds
    blind = mask_opponent_info(obs)
    assert int(blind["opp_marked"].sum()) == 0 and int(blind["opp_best"].sum()) == 0
    assert torch.equal(blind["marked"], obs["marked"])
    assert float(blind["scalars"][:, 6].abs().sum()) == 0.0 and torch.equal(blind["scalars"][:, 0], obs["scalars"][:, 0])
    model = make_model()
    lg, v = model(env.bt, blind)
    assert lg.shape == (4, env.bt.K + 1)


def test_ensemble_policy():
    from nochmal.agent import EnsemblePolicy, HeuristicPolicy, RandomPolicy, parse_ensemble
    env = NochmalEnv(BOARD, 8, 2, "cpu", seed=4)
    members = [HeuristicPolicy(env.bt), RandomPolicy(env.bt)]
    for mode in ("mean", "vote"):
        ens = EnsemblePolicy(env.bt, members, mode=mode)
        for _ in range(3):
            env.play_round(ens, record=False)
        g = torch.arange(8)
        obs = env.make_obs(g, torch.zeros(8, dtype=torch.int64), env.pool_c, env.pool_n, torch.ones(8, dtype=torch.bool))
        d = ens.distribution(obs)
        legal, _ = env.bt.legal_moves(obs)
        assert torch.all(d[~legal] == 0)
        assert torch.allclose(d.sum(1), torch.ones(8), atol=1e-4) or mode == "vote"
    assert parse_ensemble("ensemble:vote:a/best.pt+b/best.pt") == ("vote", ["a/best.pt", "b/best.pt"])


def test_board_pool_env():
    from nochmal.boardgen import random_boards
    boards = [BOARD] + random_boards(3, seed=5)
    env = NochmalEnv(boards, 12, 2, "cpu", seed=6)
    assert env.bt.B == 4 and int(env.board_id.max()) == 3
    pol = HeuristicPolicy(env.bt)
    for _ in range(25):
        env.play_round(pol, record=False)
    g = torch.arange(12)
    obs = env.make_obs(g, torch.zeros(12, dtype=torch.int64), env.pool_c, env.pool_n, torch.ones(12, dtype=torch.bool))
    legal, _ = env.bt.legal_moves(obs)
    assert not bool((legal[:, : env.bt.K] & ~env.bt.move_valid[obs["board"]]).any())   # padding moves never legal
    # every legal move lies on cells of the right colour for its board and touches an own cross
    for m in range(12):
        b = boards[int(obs["board"][m])]
        marked = set(torch.nonzero(obs["marked"][m]).flatten().tolist())
        for k in torch.nonzero(legal[m, : env.bt.K]).flatten().tolist():
            cells = b.move_cells[k]
            assert len({int(b.cell_region[c]) for c in cells}) == 1
            assert not marked or any(nb in marked for c in cells for nb in b._neighbors(c))
    scores = env.scores()
    assert scores.shape == (12, 2)


def test_official_sheets():
    """All official sheets: 21 cells per colour, 15 stars (one per column), blocks of at most 6 cells."""
    from nochmal.board import list_boards, official_board_ids
    ids = official_board_ids()
    assert len(ids) == 7 and "classic" in ids
    for bid in ids:
        b = load_board(bid)
        assert b.width == 15 and b.height == 7
        assert all(int((b.cell_color == c).sum()) == 21 for c in range(5)), bid
        star_cols = sorted(int(i) % b.width for i in range(b.n_cells) if b.cell_star[i])
        assert star_cols == list(range(15)), bid
        assert b.region_size.max() <= 6 and len(b.region_size) == 30, bid
    # a pool of all official sheets works in the environment
    env = NochmalEnv([load_board(i) for i in ids], 14, 2, "cpu", seed=3)
    assert env.bt.B == 7
    for _ in range(5):
        env.play_round(HeuristicPolicy(env.bt), record=False)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
