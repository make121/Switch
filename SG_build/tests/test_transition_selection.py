import numpy as np

from SG_build.skill_graph_V2 import GraphEdge, SkillGraph, SkillGraphBuilder


def _builder(**kwargs):
    return SkillGraphBuilder(
        motion_files=[],
        transition_selection="phase",
        phase_bins=4,
        edges_per_phase_bin=1,
        **kwargs,
    )


def test_phase_selection_covers_each_occupied_source_bin():
    builder = _builder()
    # Two target candidates per source. The lower-weight one must win, while
    # the four selected source states cover all four temporal bins.
    edges = []
    for src in (5, 30, 55, 80):
        edges.append(GraphEdge(src, 100 + src, 2.0, is_cross_skill=True))
        edges.append(GraphEdge(src, 200 + src, 1.0, is_cross_skill=True))

    selected = builder._select_pair_edges(edges, src_start=0, src_length=100)

    assert [edge.src for edge in selected] == [5, 30, 55, 80]
    assert all(edge.weight == 1.0 for edge in selected)


def test_phase_selection_uses_uniform_anchors_not_only_easy_cluster():
    builder = SkillGraphBuilder(
        motion_files=[], transition_selection="phase",
        phase_bins=1, edges_per_phase_bin=2,
    )
    edges = [
        GraphEdge(1, 101, 0.1, is_cross_skill=True),
        GraphEdge(2, 102, 0.2, is_cross_skill=True),
        GraphEdge(33, 133, 5.0, is_cross_skill=True),
        GraphEdge(67, 167, 6.0, is_cross_skill=True),
    ]

    selected = builder._select_pair_edges(edges, src_start=0, src_length=100)

    assert [edge.src for edge in selected] == [33, 67]


def test_pruning_keeps_temporal_and_only_selected_cross_edges():
    temporal = GraphEdge(0, 1, 1.0)
    kept = GraphEdge(1, 10, 2.0, is_cross_skill=True)
    removed = GraphEdge(2, 11, 3.0, is_cross_skill=True)
    graph = SkillGraph(edges=[temporal, kept, removed])
    graph._adj = {0: [(1, temporal)], 1: [(10, kept)], 2: [(11, removed)]}

    SkillGraphBuilder._retain_selected_cross_edges(graph, [kept])

    assert graph.edges == [temporal, kept]
    assert 2 not in graph._adj
    assert graph._adj[1] == [(10, kept)]


def test_recovery_roles_only_allow_directed_exits_to_tasks():
    builder = SkillGraphBuilder(
        motion_files=["task_a", "task_b", "rec_a", "rec_b"],
        skill_roles=["task", "task", "recovery", "recovery"],
    )

    assert builder._pair_allowed(0, 1)
    assert builder._pair_allowed(2, 0)
    assert not builder._pair_allowed(0, 2)
    assert not builder._pair_allowed(2, 3)


def test_recovery_exit_and_task_entry_windows_respect_boundaries():
    builder = SkillGraphBuilder(
        motion_files=["task", "recovery"],
        skill_roles=["task", "recovery"],
        exclude_boundary_frames=10,
        recovery_exit_fraction=0.2,
        task_entry_fraction=0.2,
    )
    builder._features = [np.empty(100, dtype=object),
                         np.empty(200, dtype=object)]

    src_mask, dst_mask = builder._pair_endpoint_masks(1, 0)

    # Recovery tail: [160, 190); final ten boundary frames stay excluded.
    assert src_mask[:160].all()
    assert not src_mask[160:190].any()
    assert src_mask[190:].all()
    # Task prefix: [10, 20); first ten boundary frames stay excluded.
    assert dst_mask[:10].all()
    assert not dst_mask[10:20].any()
    assert dst_mask[20:].all()
