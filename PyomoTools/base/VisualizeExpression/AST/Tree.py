from .Node import (
    ASTNode,
    NumberNode,
    OperatorNode,
    FunctionNode,
    ParenthesesNode,
    RelationalExprNode,
)
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import networkx as nx
from collections import deque


class AST:
    def __init__(self, root: ASTNode):
        self.root = root

    @property
    def display_str(self) -> str:
        """Generate a string representation of the AST."""
        return self.root.display_str

    def _hierarchy_pos(
        self,
        G: nx.DiGraph,
        root: int,
        width: float = 1.0,
        vert_gap: float = 0.2,
        vert_loc: float = 0,
        xcenter: float = 0.5,
    ) -> Dict[int, Tuple[float, float]]:
        """
        Position nodes in a tree layout.

        This is a helper function that creates a hierarchical layout for the tree,
        positioning parent nodes above their children.

        Args:
            G: The directed graph (tree)
            root: The root node
            width: Horizontal space allocated for this branch
            vert_gap: Gap between levels
            vert_loc: Vertical location of root
            xcenter: Horizontal location of root

        Returns:
            Dictionary mapping node to (x, y) position
        """
        pos = {root: (xcenter, vert_loc)}
        children = list(G.successors(root))

        if children:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos.update(
                    self._hierarchy_pos(
                        G,
                        child,
                        width=dx,
                        vert_gap=vert_gap,
                        vert_loc=vert_loc - vert_gap,
                        xcenter=nextx,
                    )
                )
        return pos

    def plot(
        self,
        figsize: Tuple[int, int] = (12, 8),
        node_size: int = 2000,
        font_size: int = 10,
        title: str = "AST Tree",
    ):
        """
        Plot the AST tree using matplotlib and networkx.

        Args:
            figsize: Figure size as (width, height) tuple
            node_size: Size of the nodes in the plot
            font_size: Font size for node labels
            title: Title of the plot
        """
        G = nx.DiGraph()
        labels = {}
        node_colors = []

        # Color scheme for different node types
        color_map = {
            "NumberNode": "#90EE90",  # Light green
            "BinaryOpNode": "#FFB6C1",  # Light pink
            "UnaryOpNode": "#DDA0DD",  # Plum
            "FunctionNode": "#87CEEB",  # Sky blue
            "RelationalExprNode": "#FFD700",  # Gold
        }

        def get_node_label(node: "ASTNode") -> str:
            """Get a display label for a node."""
            if isinstance(node, NumberNode):
                return node.original_str if node.original_str else str(node.value)
            elif isinstance(node, OperatorNode):
                return node.op
            elif isinstance(node, FunctionNode):
                return f"{node.name}()"
            elif isinstance(node, RelationalExprNode):
                return node.op
            return "?"

        def add_nodes_edges(
            node: "ASTNode", parent_id: Optional[int] = None, counter: List[int] = None
        ) -> int:
            """Recursively add nodes and edges to the graph."""
            if counter is None:
                counter = [0]

            node_id = counter[0]
            counter[0] += 1

            # Add node
            G.add_node(node_id)
            labels[node_id] = get_node_label(node)
            node_colors.append(color_map.get(type(node).__name__, "#FFFFFF"))

            # Add edge from parent
            if parent_id is not None:
                G.add_edge(parent_id, node_id)

            # Recursively add children
            if isinstance(node, OperatorNode):
                for operand in node.operands:
                    add_nodes_edges(operand, node_id, counter)
            elif isinstance(node, FunctionNode):
                for arg in node.args:
                    add_nodes_edges(arg, node_id, counter)
            elif isinstance(node, ParenthesesNode):
                add_nodes_edges(node.child, node_id, counter)
            elif isinstance(node, RelationalExprNode):
                add_nodes_edges(node.left, node_id, counter)
                add_nodes_edges(node.right, node_id, counter)

            return node_id

        # Build the graph
        add_nodes_edges(self.root)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use a hierarchical layout
        pos = self._hierarchy_pos(G, 0)

        # Draw the graph
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            font_weight="bold",
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            edge_color="gray",
        )

        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=15,
                label=name.replace("Node", ""),
            )
            for name, color in color_map.items()
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

        plt.tight_layout()
        plt.show()

        return fig, ax

    def AssignConnectorPositions(self):
        self.root.determine_positions(0)

    def DetermineEvaluationReadyNodes(self) -> deque[ASTNode]:
        ready_nodes = deque()
        for node in self.root.DFS(stop_at_eval_ready=True):
            if node.evaluation_ready:
                ready_nodes.append(node)

        return ready_nodes
