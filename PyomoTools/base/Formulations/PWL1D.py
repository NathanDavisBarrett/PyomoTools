from dataclasses import dataclass
from enum import Enum

from typing import List, Tuple
from functools import cached_property


class PWL1DType(Enum):
    CONVEX = 1
    CONCAVE = 2
    GENERAL = 3
    LINEAR = 4


@dataclass
class PWL1DParameters:
    points: List[Tuple[float, float]]  # List of (x,y) points defining the PWL function
    includeUB: bool = True
    includeLB: bool = True

    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError(
                "At least two points are required to define a PWL function."
            )
        self.points.sort(key=lambda pt: pt[0])
        if not (self.includeLB or self.includeUB):
            raise ValueError("At least one of includeLB or includeUB must be True.")

    @cached_property
    def num_points(self) -> int:
        return len(self.points)

    @cached_property
    def num_segments(self) -> int:
        return self.num_points - 1

    @cached_property
    def pwl_type(self):
        slopes = [
            (self.points[i + 1][1] - self.points[i][1])
            / (self.points[i + 1][0] - self.points[i][0])
            for i in range(len(self.points) - 1)
        ]
        is_convex = all(
            slopes[i] <= slopes[i + 1] + 1e-7 for i in range(len(slopes) - 1)
        )
        is_concave = all(
            slopes[i] >= slopes[i + 1] - 1e-7 for i in range(len(slopes) - 1)
        )

        if not is_convex and not is_concave:
            if self.includeLB and self.includeUB:
                return PWL1DType.GENERAL
            elif self.includeLB and not self.includeUB:
                raise ValueError(
                    "PWL functions that are neither convex nor concave must include an upper bound."
                )
            elif not self.includeLB and self.includeUB:
                raise ValueError(
                    "PWL functions that are neither convex nor concave must include a lower bound."
                )
        elif is_convex and not is_concave:
            if not self.includeLB:
                raise ValueError("Convex PWL functions must include a lower bound.")
            if not self.includeUB:
                return PWL1DType.CONVEX
            else:
                return PWL1DType.GENERAL
        elif is_concave and not is_convex:
            if not self.includeUB:
                raise ValueError("Concave PWL functions must include an upper bound.")
            if not self.includeLB:
                return PWL1DType.CONCAVE
            else:
                return PWL1DType.GENERAL
        else:
            return PWL1DType.LINEAR
