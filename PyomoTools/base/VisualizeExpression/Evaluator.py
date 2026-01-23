from .AST import AST
from typing import List, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class IterateData:
    expression_str: str
    start_groupings: deque[Tuple[int, int]] = field(default_factory=deque)
    end_groupings: deque[Tuple[int, int]] = field(default_factory=deque)


class Evaluator:
    def __init__(
        self,
        ast: AST,
        original_str: str,
    ):
        self.ast = ast
        self.ast.root.assign_parent_relationships()

        initial_iterate = self.CompleteInitialIterate(original_str)

        start_iterate = IterateData(
            expression_str=self.ast.display_str,
        )

        self.iterates: deque[IterateData] = deque([initial_iterate, start_iterate])

        while self.Step():  # Step all the way through.
            pass

    def CompleteInitialIterate(self, original_str: str):
        """
        The originally provided string is unlikely to align perfectly with the AST. Thus, the first iteration will always to be to map the original string groupings to the AST's starting groupings.

        The original string node positions are stored in ast itself immediately after parsing.

        Args:
            original_str: The original expression string
        """
        initial_iterate = IterateData(expression_str=original_str)
        initial_iterate.start_groupings = deque()
        initial_iterate.end_groupings = deque()
        for node in self.ast.root.LeafNodes():
            initial_iterate.start_groupings.append((node.startPos, node.endPos))
        self.ast.AssignConnectorPositions()
        for node in self.ast.root.LeafNodes():
            initial_iterate.end_groupings.append((node.startPos, node.endPos))

        return initial_iterate

    def Step(self):
        """
        Perform one evaluation step.

        Returns:
            True if a step was performed, False if no more steps are possible.
        """
        current_iterate = self.iterates[-1]

        ready_nodes = self.ast.DetermineEvaluationReadyNodes()
        if not ready_nodes:
            return False  # No more nodes to evaluate

        replacement_nodes = deque()
        parent_processed = False
        for node in ready_nodes:
            current_start = node.startPos
            current_end = node.endPos
            evaluated_node = node.evaluate()
            if node.parent_info is not None:
                parent_node, child_index = node.parent_info
            else:
                parent_node = None
                child_index = None
            if parent_node is not None:
                parent_node.children[child_index] = evaluated_node
            else:
                assert node == self.ast.root
                self.ast.root = evaluated_node
                parent_processed = True

            current_iterate.start_groupings.append((current_start, current_end))
            replacement_nodes.append(evaluated_node)

        self.ast.AssignConnectorPositions()
        for node in replacement_nodes:
            current_iterate.end_groupings.append((node.startPos, node.endPos))

        self.iterates.append(
            IterateData(
                expression_str=self.ast.display_str,
            )
        )

        return not parent_processed
