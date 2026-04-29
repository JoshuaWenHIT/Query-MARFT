# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Flex-MG  — Flexible Markov Game modelling for Multi-Object Tracking.

Key differences from a standard Markov Game:
  * Asynchronous sequential execution (DAG topology).
  * Heterogeneous agents with independent action spaces.
  * Scene-adaptive dependency graph.

This module is *not* a nn.Module; it is a stateless game-step executor used
by the MARFT training engine.
"""

import copy
from typing import Any, Dict, List, Optional

from .scene_analyzer import SceneAnalyzer, SceneInfo


class FlexMGGame:
    """
    Models each frame of a MOT clip as one step in a Flex Markov Game.
    """

    BASE_DEPENDENCY: Dict[str, List[str]] = {
        'det': [],
        'assoc': ['det'],
        'update': ['assoc'],
        'corr': ['update'],
    }

    def __init__(
        self,
        scene_adaptive: bool = True,
        img_w: int = 1920,
        img_h: int = 1080,
    ):
        self.scene_adaptive = scene_adaptive
        self.analyzer = SceneAnalyzer(img_w, img_h)
        self._current_dep_graph = copy.deepcopy(self.BASE_DEPENDENCY)

    # ------------------------------------------------------------------
    def get_execution_order(
        self,
        scene_info: Optional[SceneInfo] = None,
    ) -> List[str]:
        """
        Topological sort of the (possibly adapted) dependency graph.
        With the current four agents the order is always linear, but
        this API allows future extensions to more complex DAGs.
        """
        if self.scene_adaptive and scene_info is not None:
            graph = self.adapt_dependency_graph(scene_info)
        else:
            graph = self._current_dep_graph
        return self._topo_sort(graph)

    # ------------------------------------------------------------------
    def adapt_dependency_graph(self, scene_info: SceneInfo) -> Dict[str, List[str]]:
        """
        Dynamically reshape the dependency graph based on scene statistics.
        """
        graph = copy.deepcopy(self.BASE_DEPENDENCY)

        if scene_info.occlusion_ratio > 0.4:
            if 'corr' not in graph.get('det', []):
                graph.setdefault('det', []).append('corr')

        if scene_info.target_density < 0.1:
            graph['corr'] = ['update']

        if scene_info.avg_speed > 0.05:
            if 'corr' not in graph.get('update', []):
                graph.setdefault('update', []).append('corr')

        self._current_dep_graph = graph
        return graph

    # ------------------------------------------------------------------
    @staticmethod
    def _topo_sort(graph: Dict[str, List[str]]) -> List[str]:
        """Kahn's algorithm.  Falls back to the canonical order on cycles."""
        in_deg = {n: 0 for n in graph}
        for deps in graph.values():
            for d in deps:
                if d in in_deg:
                    in_deg[d] += 1
        # NOTE: in_deg counts *reverse* edges (n depends on d ⇒ d has in_deg+1
        # in the *forward* execution graph). We actually want to topo-sort the
        # *dependency* DAG, so predecessors execute first.
        # Re-compute with forward edges: if 'assoc' depends on 'det', then
        # in the execution graph det → assoc, so in_deg[assoc] += 1.
        in_deg = {n: 0 for n in graph}
        adj: Dict[str, List[str]] = {n: [] for n in graph}
        for node, deps in graph.items():
            for d in deps:
                if d in adj:
                    adj[d].append(node)
                    in_deg[node] += 1

        queue = sorted([n for n, d in in_deg.items() if d == 0])
        order: List[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in sorted(adj.get(n, [])):
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)
        if len(order) != len(graph):
            return ['det', 'assoc', 'update', 'corr']
        return order
