import numpy as np

from typing import List, Tuple
from collections import deque
import networkx as nx


def DetermineLeadingLines(
    startGroupings: List[Tuple[int, int]], endGroupings: List[Tuple[int, int]]
) -> List[str]:
    """
    For each grouping in startGroupings, determine the leading line characters needed to connect to the corresponding grouping in endGroupings.

    Requirements:
    1) All lines must be connected, non-crossing, and must only use the following characters:
        ' ' (space) - no line
        '│' - vertical line
        '─' - horizontal line
        '┌' - corner from horizontal to vertical (top-left)
        '┐' - corner from horizontal to vertical (top-right)
        '└' - corner from vertical to horizontal (bottom-left)
        '┘' - corner from vertical to horizontal (bottom-right)
        '├' - T-junction (vertical with right horizontal)
        '┤' - T-junction (vertical with left horizontal)
        '┬' - T-junction (horizontal with down vertical)
        '┴' - T-junction (horizontal with up vertical)

    2) The first outputted line should contain a "cup" under each entire grouping in startGroupings. Extending out of the bottom of this cup should be the line that starts the leading line. If the grouping is too narrow for a cup, use ├, or ┤ as appropriate. If it is just a single character wide, use '│'

    3) Each leading line must start within the boundaries of its corresponding startGrouping and end within the boundaries of its corresponding endGrouping.

    4) Multiple lines can be outputted if necessary to avoid crossing lines. Lines should never cross or touch, except at the start and end points of their segments.

    ALGORITHM:
    Each line will start and end at the center of its grouping. Each line will therefore consist of three segments: a vertical drop from the start grouping to some row, a horizontal segment to the column of the end grouping, and a vertical drop to the end grouping. The only thing to determine is how many rows are needed to avoid crossing lines, and which lines go on which rows. This will be done by iteratively assigning lines to rows, starting new rows as needed.
    """

    # Treat spans as half-open intervals [start, end) to match AST positions
    # (start inclusive, end exclusive). This prevents cups from overshooting
    # and keeps leading lines aligned to the AST display string.
    def _stem(start: int, end: int) -> int:
        width = end - start
        center = width // 2
        return start + center

    num_lines = len(startGroupings)
    s_stems = [_stem(s[0], s[1]) for s in startGroupings]
    e_stems = [_stem(e[0], e[1]) for e in endGroupings]

    max_s_coord = max((s[1] for s in startGroupings), default=0)
    max_e_coord = max((e[1] for e in endGroupings), default=0)
    width = max(max_s_coord, max_e_coord)

    # Determine which lines have the possibility of crossing
    crossing_pairs = set()
    for i in range(num_lines):
        for j in range(i):
            if e_stems[i] <= s_stems[j]:
                crossing_pairs.add((i, j))  # i must be on a lower track than j
        for j in range(i + 1, num_lines):
            if e_stems[i] >= s_stems[j]:
                crossing_pairs.add((i, j))

    # Generate a graph of crossings (a -> b indicates that line a must be on a lower track than line b)
    cross_graph = nx.DiGraph()
    for i in range(num_lines):
        cross_graph.add_node(i)

    for i, j in crossing_pairs:
        cross_graph.add_edge(i, j)

    # Perform a topological sort to determine track assignments
    try:
        topo_order = list(nx.topological_sort(cross_graph))
    except nx.NetworkXUnfeasible:
        raise ValueError(
            "Impossible to route leading lines without crossing. This indicates a bug in the code somewhere upstream from here. This should never happen."
        )

    # Assign tracks
    track_assignments = {}
    for i in topo_order:
        neighbor_tracks = [
            track_assignments[j]
            for j in cross_graph.predecessors(i)
            if j in track_assignments
        ]

        if not neighbor_tracks:
            track_assignments[i] = 0  # This must be on the bottom track
        else:
            track_assignments[i] = (
                max(neighbor_tracks) + 1
            )  # One above the highest neighbor

    num_tracks = max(track_assignments.values()) + 1 if track_assignments else 0

    # Initialize the output lines
    output = np.full((num_tracks + 1, width), " ", dtype="<U1")
    for i, (s_start, s_end) in enumerate(startGroupings):
        w = s_end - s_start
        if w <= 0:
            raise ValueError("Invalid grouping span: end must be greater than start")
        if w == 1:
            cup = ["│"]
        elif w == 2:
            cup = list("├┘")
        else:
            cup = list("└" + "─" * (w - 2) + "┘")
            center = w // 2
            cup[center] = "┬"

        output[0, s_start:s_end] = cup

    for i in range(num_lines):
        track = track_assignments[i]
        row = num_tracks - track
        s_col = s_stems[i]
        e_col = e_stems[i]
        # Vertical drop from start grouping
        output[1:row, s_col] = "│"
        # Horizontal segment
        if s_col < e_col:
            output[row, s_col : e_col + 1] = "─"
            output[row, s_col] = "└"
            output[row, e_col] = "┐"
        elif s_col > e_col:
            output[row, e_col : s_col + 1] = "─"
            output[row, s_col] = "┘"
            output[row, e_col] = "┌"
        else:
            output[row, s_col] = "│"
        # Vertical drop to end grouping
        output[row + 1 :, e_col] = "│"

    return ["".join(output[r, :]) for r in range(output.shape[0])]
