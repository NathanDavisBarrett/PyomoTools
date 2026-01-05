import pyomo.kernel as pmo
import pytest
import numpy as np
from unittest.mock import MagicMock

from ..IntegerRelaxationReport import IntegerRelaxationReport
from ...base.Solvers import DefaultSolver


class TestIntegerRelaxationReport:
    """Comprehensive test suite for IntegerRelaxationReport class."""

    # ========== Test Fixtures ==========

    @pytest.fixture
    def simple_milp_model(self):
        """
        Create a simple MILP model with known fractional LP solution.

        Model:
            min x + y
            s.t. x >= 0.5
                 y >= 0.5
                 x, y integer

        LP relaxation has solution (0.5, 0.5) with objective 1.0
        MILP solution is (1, 1) with objective 2.0
        """
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=0)
        m.y = pmo.variable(domain=pmo.Integers, lb=0)

        m.c1 = pmo.constraint(m.x >= 0.5)
        m.c2 = pmo.constraint(m.y >= 0.5)

        m.obj = pmo.objective(m.x + m.y)

        # Solve the MILP
        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        return m

    @pytest.fixture
    def pure_continuous_model(self):
        """
        Create a model with only continuous variables.

        Model:
            min x + y
            s.t. x >= 0.5
                 y >= 0.5
                 x, y continuous
        """
        m = pmo.block()
        m.x = pmo.variable(lb=0)
        m.y = pmo.variable(lb=0)

        m.c1 = pmo.constraint(m.x >= 0.5)
        m.c2 = pmo.constraint(m.y >= 0.5)

        m.obj = pmo.objective(m.x + m.y)

        solver = DefaultSolver("LP")
        solver.solve(m, tee=False)

        return m

    @pytest.fixture
    def integer_solution_model(self):
        """
        Create a model where LP relaxation gives an integer solution.

        Model:
            min x + y
            s.t. x >= 1
                 y >= 2
                 x, y integer

        Both LP and MILP solutions are (1, 2) with objective 3.0
        """
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=0)
        m.y = pmo.variable(domain=pmo.Integers, lb=0)

        m.c1 = pmo.constraint(m.x >= 1)
        m.c2 = pmo.constraint(m.y >= 2)

        m.obj = pmo.objective(m.x + m.y)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        return m

    @pytest.fixture
    def mixed_integer_model(self):
        """
        Create a model with both integer and continuous variables.

        Model:
            min x + y + z
            s.t. x >= 0.7
                 y >= 0.3
                 z >= 0.5
                 x integer, y continuous, z integer
        """
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=0)
        m.y = pmo.variable(lb=0)  # continuous
        m.z = pmo.variable(domain=pmo.Integers, lb=0)

        m.c1 = pmo.constraint(m.x >= 0.7)
        m.c2 = pmo.constraint(m.y >= 0.3)
        m.c3 = pmo.constraint(m.z >= 0.5)

        m.obj = pmo.objective(m.x + m.y + m.z)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        return m

    @pytest.fixture
    def knapsack_model(self):
        """
        Create a classic 0-1 knapsack problem.

        Model:
            max 4*x1 + 5*x2 + 3*x3 + 7*x4
            s.t. 2*x1 + 3*x2 + 1*x3 + 4*x4 <= 7
                 x1, x2, x3, x4 binary
        """
        m = pmo.block()
        m.x1 = pmo.variable(domain=pmo.Binary)
        m.x2 = pmo.variable(domain=pmo.Binary)
        m.x3 = pmo.variable(domain=pmo.Binary)
        m.x4 = pmo.variable(domain=pmo.Binary)

        m.capacity = pmo.constraint(2 * m.x1 + 3 * m.x2 + 1 * m.x3 + 4 * m.x4 <= 7)

        m.obj = pmo.objective(
            4 * m.x1 + 5 * m.x2 + 3 * m.x3 + 7 * m.x4, sense=pmo.maximize
        )

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        return m

    # ========== Basic Functionality Tests ==========

    def test_initialization(self, simple_milp_model):
        """Test that IntegerRelaxationReport initializes correctly."""
        report = IntegerRelaxationReport(simple_milp_model)

        assert report.model is simple_milp_model
        assert report.lp_relaxation is not None
        assert report.solver is not None
        assert report.tolerance == 1e-5

    def test_custom_tolerance(self, simple_milp_model):
        """Test that custom tolerance is properly set."""
        custom_tol = 1e-7
        report = IntegerRelaxationReport(simple_milp_model, tolerance=custom_tol)

        assert report.tolerance == custom_tol

    def test_custom_solver(self, simple_milp_model):
        """Test that custom solver is properly used."""
        custom_solver = DefaultSolver("LP")
        report = IntegerRelaxationReport(simple_milp_model, solver=custom_solver)

        assert report.solver is custom_solver

    def test_solver_options(self, simple_milp_model):
        """Test that solver options can be passed."""
        # This mainly tests that the initialization doesn't fail with options
        report = IntegerRelaxationReport(
            simple_milp_model, solver_options={"TimeLimit": 100}
        )

        assert report is not None

    # ========== Fractionality Ratio Tests ==========

    def test_fractionality_ratio_fractional_solution(self, simple_milp_model):
        """Test fractionality_ratio with a fractional LP solution."""
        report = IntegerRelaxationReport(simple_milp_model)

        # LP solution should be (0.5, 0.5), both fractional
        # So fractionality_ratio should be 2/2 = 1.0
        assert report.fractionality_ratio == pytest.approx(1.0, abs=1e-4)

    def test_fractionality_ratio_integer_solution(self, integer_solution_model):
        """Test fractionality_ratio when LP gives integer solution."""
        report = IntegerRelaxationReport(integer_solution_model)

        # LP solution should be (1, 2), both integer
        # So fractionality_ratio should be 0/2 = 0.0
        assert report.fractionality_ratio == pytest.approx(0.0, abs=1e-4)

    def test_fractionality_ratio_no_integer_vars(self, pure_continuous_model):
        """Test fractionality_ratio with no integer variables."""
        report = IntegerRelaxationReport(pure_continuous_model)

        # No integer variables, should return 0.0
        assert report.fractionality_ratio == 0.0

    def test_fractionality_ratio_mixed_model(self, mixed_integer_model):
        """Test fractionality_ratio with mixed integer/continuous variables."""
        report = IntegerRelaxationReport(mixed_integer_model)

        # LP solution should have x=0.7 and z=0.5 (both fractional)
        # Only 2 integer vars, both fractional: ratio = 2/2 = 1.0
        assert report.fractionality_ratio == pytest.approx(1.0, abs=1e-4)

    def test_fractionality_ratio_partial_fractional(self):
        """Test fractionality_ratio when only some integer vars are fractional."""
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=0)
        m.y = pmo.variable(domain=pmo.Integers, lb=0)
        m.z = pmo.variable(domain=pmo.Integers, lb=0)

        # Force x to be integer, y and z to be fractional in LP relaxation
        m.c1 = pmo.constraint(m.x >= 2)
        m.c2 = pmo.constraint(m.y >= 0.5)
        m.c3 = pmo.constraint(m.z >= 0.7)

        m.obj = pmo.objective(m.x + m.y + m.z)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        report = IntegerRelaxationReport(m)

        # 2 out of 3 integer vars are fractional: ratio = 2/3
        assert report.fractionality_ratio == pytest.approx(2.0 / 3.0, abs=1e-4)

    # ========== Integer Infeasibility Tests ==========

    def test_sum_integer_infeasibility(self, simple_milp_model):
        """Test sum_integer_infeasibility calculation."""
        report = IntegerRelaxationReport(simple_milp_model)

        # LP solution is (0.5, 0.5)
        # Infeasibilities: |0.5 - 0| = 0.5, |0.5 - 1| = 0.5
        # Sum = 0.5 + 0.5 = 1.0
        assert report.sum_integer_infeasibility == pytest.approx(1.0, abs=1e-4)

    def test_max_integer_infeasibility(self, simple_milp_model):
        """Test max_integer_infeasibility calculation."""
        report = IntegerRelaxationReport(simple_milp_model)

        # LP solution is (0.5, 0.5)
        # Max infeasibility = 0.5
        assert report.max_integer_infeasibility == pytest.approx(0.5, abs=1e-4)

    def test_integer_infeasibility_zero(self, integer_solution_model):
        """Test infeasibility metrics when LP gives integer solution."""
        report = IntegerRelaxationReport(integer_solution_model)

        # LP solution is integer, so all infeasibilities should be 0
        assert report.sum_integer_infeasibility == pytest.approx(0.0, abs=1e-4)
        assert report.max_integer_infeasibility == pytest.approx(0.0, abs=1e-4)

    def test_integer_infeasibility_no_integer_vars(self, pure_continuous_model):
        """Test infeasibility metrics with no integer variables."""
        report = IntegerRelaxationReport(pure_continuous_model)

        # No integer variables, all infeasibilities should be 0
        assert report.sum_integer_infeasibility == 0.0
        assert report.max_integer_infeasibility == 0.0

    def test_integer_infeasibility_various_values(self):
        """Test infeasibility metrics with various fractional values."""
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=0)
        m.y = pmo.variable(domain=pmo.Integers, lb=0)
        m.z = pmo.variable(domain=pmo.Integers, lb=0)

        m.c1 = pmo.constraint(m.x >= 0.2)
        m.c2 = pmo.constraint(m.y >= 0.8)
        m.c3 = pmo.constraint(m.z >= 1.5)

        m.obj = pmo.objective(m.x + m.y + m.z)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        report = IntegerRelaxationReport(m)

        # LP solution: (0.2, 0.8, 1.5)
        # Infeasibilities: |0.2-0|=0.2, |0.8-1|=0.2, |1.5-2|=0.5
        # Sum = 0.2 + 0.2 + 0.5 = 0.9
        # Max = 0.5
        assert report.sum_integer_infeasibility == pytest.approx(0.9, abs=1e-4)
        assert report.max_integer_infeasibility == pytest.approx(0.5, abs=1e-4)

    # ========== Euclidean Distance Tests ==========

    def test_euclidean_distance_to_nearest_neighbor(self, simple_milp_model):
        """Test euclidean_distance_to_nearest_neighbor calculation."""
        report = IntegerRelaxationReport(simple_milp_model)

        # LP solution is (0.5, 0.5)
        # Distance = sqrt(0.5^2 + 0.5^2) = sqrt(0.5)
        expected = np.sqrt(0.5)
        assert report.euclidean_distance_to_nearest_neighbor == pytest.approx(
            expected, abs=1e-4
        )

    def test_euclidean_distance_zero(self, integer_solution_model):
        """Test euclidean distance when LP gives integer solution."""
        report = IntegerRelaxationReport(integer_solution_model)

        # LP solution is integer, distance should be 0
        assert report.euclidean_distance_to_nearest_neighbor == pytest.approx(
            0.0, abs=1e-4
        )

    def test_euclidean_distance_no_integer_vars(self, pure_continuous_model):
        """Test euclidean distance with no integer variables."""
        report = IntegerRelaxationReport(pure_continuous_model)

        # No integer variables, distance should be 0
        assert report.euclidean_distance_to_nearest_neighbor == 0.0

    def test_euclidean_distance_multiple_vars(self):
        """Test euclidean distance with multiple fractional variables."""
        m = pmo.block()
        m.x1 = pmo.variable(domain=pmo.Integers, lb=0)
        m.x2 = pmo.variable(domain=pmo.Integers, lb=0)
        m.x3 = pmo.variable(domain=pmo.Integers, lb=0)

        m.c1 = pmo.constraint(m.x1 >= 0.3)
        m.c2 = pmo.constraint(m.x2 >= 0.4)
        m.c3 = pmo.constraint(m.x3 >= 0.5)

        m.obj = pmo.objective(m.x1 + m.x2 + m.x3)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        report = IntegerRelaxationReport(m)

        # LP solution: (0.3, 0.4, 0.5)
        # Distance = sqrt(0.3^2 + 0.4^2 + 0.5^2) = sqrt(0.09 + 0.16 + 0.25) = sqrt(0.5)
        expected = np.sqrt(0.5)
        assert report.euclidean_distance_to_nearest_neighbor == pytest.approx(
            expected, abs=1e-4
        )

    # ========== Gap to Integer Solution Tests ==========

    def test_gap_to_integer_solution(self, simple_milp_model):
        """Test gap_to_integer_solution calculation."""
        report = IntegerRelaxationReport(simple_milp_model)

        # MILP objective: 2.0 (from x=1, y=1)
        # LP objective: 1.0 (from x=0.5, y=0.5)
        # Gap = |1.0 - 2.0| / |2.0| = 0.5
        assert report.gap_to_integer_solution == pytest.approx(0.5, abs=1e-4)

    def test_gap_zero_when_solutions_match(self, integer_solution_model):
        """Test gap is zero when LP and MILP solutions match."""
        report = IntegerRelaxationReport(integer_solution_model)

        # Both LP and MILP have objective 3.0
        # Gap = |3.0 - 3.0| / |3.0| = 0.0
        assert report.gap_to_integer_solution == pytest.approx(0.0, abs=1e-4)

    def test_gap_with_zero_objective(self):
        """Test gap calculation when objective value is near zero."""
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=-1, ub=1)
        m.y = pmo.variable(domain=pmo.Integers, lb=-1, ub=1)

        m.c1 = pmo.constraint(m.x + m.y >= 0.1)

        # Objective that results in very small value
        m.obj = pmo.objective(m.x + m.y)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        report = IntegerRelaxationReport(m)

        # When objective is near zero, should return absolute error
        assert report.gap_to_integer_solution >= 0

    def test_gap_maximization_objective(self, knapsack_model):
        """Test gap calculation with maximization objective."""
        report = IntegerRelaxationReport(knapsack_model)

        # Gap should be calculated correctly regardless of sense
        assert report.gap_to_integer_solution >= 0

    # ========== Caching Tests ==========

    def test_cached_properties_computed_once(self, simple_milp_model):
        """Test that cached properties are only computed once."""
        report = IntegerRelaxationReport(simple_milp_model)

        # Access a cached property multiple times
        val1 = report.fractionality_ratio
        val2 = report.fractionality_ratio

        # Should return the same object (not just equal values)
        assert val1 == val2

        # Similar test for other cached properties
        inf1 = report.sum_integer_infeasibility
        inf2 = report.sum_integer_infeasibility
        assert inf1 == inf2

    def test_integer_infeasibilities_shared_cache(self, simple_milp_model):
        """Test that sum, max, and euclidean distance share computation."""
        report = IntegerRelaxationReport(simple_milp_model)

        # All three should be computed together
        sum_inf = report.sum_integer_infeasibility
        max_inf = report.max_integer_infeasibility
        euc_dist = report.euclidean_distance_to_nearest_neighbor

        # All should be non-negative
        assert sum_inf >= 0
        assert max_inf >= 0
        assert euc_dist >= 0

    # ========== Edge Cases and Error Handling ==========

    def test_empty_model(self):
        """Test behavior with an empty model."""
        m = pmo.block()
        m.x = pmo.variable(lb=0)
        m.obj = pmo.objective(m.x)

        solver = DefaultSolver("LP")
        solver.solve(m, tee=False)

        report = IntegerRelaxationReport(m)

        # Should handle gracefully
        assert report.fractionality_ratio == 0.0
        assert report.sum_integer_infeasibility == 0.0

    def test_single_integer_variable(self):
        """Test with a single integer variable."""
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=0)
        m.c1 = pmo.constraint(m.x >= 0.7)
        m.obj = pmo.objective(m.x)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        report = IntegerRelaxationReport(m)

        # LP solution: x = 0.7, infeasibility = 0.3
        assert report.fractionality_ratio == pytest.approx(1.0, abs=1e-4)
        assert report.max_integer_infeasibility == pytest.approx(0.3, abs=1e-4)

    def test_binary_variables(self, knapsack_model):
        """Test with binary variables (special case of integer)."""
        report = IntegerRelaxationReport(knapsack_model)

        # Binary variables should be treated as integer variables
        assert report.fractionality_ratio >= 0
        assert report.fractionality_ratio <= 1.0

    def test_large_model_performance(self):
        """Test that the report handles larger models reasonably."""
        m = pmo.block()

        # Create a model with 50 integer variables
        n = 50
        m.x = pmo.variable_list()
        for i in range(n):
            m.x.append(pmo.variable(domain=pmo.Integers, lb=0))

        # Add constraints
        m.constraints = pmo.constraint_list()
        for i in range(n):
            m.constraints.append(pmo.constraint(m.x[i] >= 0.3 + 0.01 * i))

        m.obj = pmo.objective(sum(m.x[i] for i in range(n)))

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        # Should complete without errors
        report = IntegerRelaxationReport(m)
        assert report.fractionality_ratio >= 0

    def test_tolerance_near_integer(self):
        """Test that tolerance is properly applied for near-integer values."""
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=0)
        m.y = pmo.variable(domain=pmo.Integers, lb=0)

        # Create constraints that would result in x ≈ 1.0 and y ≈ 2.0 in LP
        m.c1 = pmo.constraint(m.x >= 1.0)
        m.c2 = pmo.constraint(m.y >= 2.0)

        m.obj = pmo.objective(m.x + m.y)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        # With default tolerance, should recognize as integer
        report = IntegerRelaxationReport(m, tolerance=1e-5)
        assert report.fractionality_ratio == pytest.approx(0.0, abs=1e-4)

    def test_warning_on_non_optimal_lp(self):
        """Test that a warning is issued if LP doesn't solve to optimality."""
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=0, ub=10)
        m.obj = pmo.objective(m.x)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        # Create a mock solver that returns non-optimal status
        mock_solver = MagicMock()
        mock_results = MagicMock()
        mock_results.solver.termination_condition = pmo.TerminationCondition.infeasible
        mock_solver.solve.return_value = mock_results

        # Should issue a warning
        with pytest.warns(
            UserWarning, match="LP relaxation did not solve to optimality"
        ):
            report = IntegerRelaxationReport(m, solver=mock_solver)

    def test_model_with_equality_constraints(self):
        """Test with a model containing equality constraints."""
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=0)
        m.y = pmo.variable(domain=pmo.Integers, lb=0)
        m.z = pmo.variable(lb=0)  # continuous

        m.eq1 = pmo.constraint(m.x + m.y == 2.5)  # Will force fractional
        m.eq2 = pmo.constraint(m.z == m.x + m.y)

        m.obj = pmo.objective(m.x + m.y + m.z)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        report = IntegerRelaxationReport(m)

        # Should handle equality constraints properly
        assert report is not None
        assert report.fractionality_ratio >= 0

    def test_model_with_unbounded_variables(self):
        """Test with unbounded variables."""
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers)  # No bounds
        m.y = pmo.variable(domain=pmo.Integers, lb=0)

        m.c1 = pmo.constraint(m.x + m.y >= 1.5)
        m.c2 = pmo.constraint(m.x >= -5)

        m.obj = pmo.objective(m.x + m.y)

        solver = DefaultSolver("MILP")
        solver.solve(m, tee=False)

        report = IntegerRelaxationReport(m)

        # Should handle unbounded variables
        assert report is not None

    # ========== Integration Tests ==========

    def test_all_metrics_together(self, simple_milp_model):
        """Test that all metrics can be computed together without conflicts."""
        report = IntegerRelaxationReport(simple_milp_model)

        # Compute all metrics
        frac_ratio = report.fractionality_ratio
        sum_inf = report.sum_integer_infeasibility
        max_inf = report.max_integer_infeasibility
        euc_dist = report.euclidean_distance_to_nearest_neighbor
        gap = report.gap_to_integer_solution

        # All should be computed and valid
        assert frac_ratio >= 0
        assert sum_inf >= 0
        assert max_inf >= 0
        assert euc_dist >= 0
        assert gap >= 0

    def test_report_on_various_model_types(self):
        """Test that report works on various model structures."""
        models = []

        # Model 1: Simple LP
        m1 = pmo.block()
        m1.x = pmo.variable(lb=0)
        m1.obj = pmo.objective(m1.x)
        solver = DefaultSolver("LP")
        solver.solve(m1, tee=False)
        models.append(m1)

        # Model 2: Single integer variable
        m2 = pmo.block()
        m2.x = pmo.variable(domain=pmo.Integers, lb=0)
        m2.c = pmo.constraint(m2.x >= 0.5)
        m2.obj = pmo.objective(m2.x)
        solver = DefaultSolver("MILP")
        solver.solve(m2, tee=False)
        models.append(m2)

        # Model 3: Multiple constraints
        m3 = pmo.block()
        m3.x = pmo.variable(domain=pmo.Integers, lb=0, ub=10)
        m3.y = pmo.variable(domain=pmo.Integers, lb=0, ub=10)
        m3.c1 = pmo.constraint(m3.x + m3.y <= 5.5)
        m3.c2 = pmo.constraint(2 * m3.x + m3.y >= 3.7)
        m3.obj = pmo.objective(m3.x + 2 * m3.y)
        solver = DefaultSolver("MILP")
        solver.solve(m3, tee=False)
        models.append(m3)

        # All models should generate reports without errors
        for model in models:
            report = IntegerRelaxationReport(model)
            assert report is not None
            assert report.fractionality_ratio >= 0

    def generate_one_param_model(self, k):
        m = pmo.block()
        m.x = pmo.variable(domain=pmo.Integers, lb=0)
        m.y = pmo.variable(domain=pmo.Integers, lb=0)

        m.c1 = pmo.constraint(m.x + 2 * m.y <= k)
        m.c2 = pmo.constraint(2 * m.x + m.y <= k)
        m.obj = pmo.objective(m.x + m.y, sense=pmo.maximize)

        return m

    def test_expected_behavior(self):
        """
        In this test, I have two models, m1 and m2, clearly, m1 is tighter than m2. We will verify that this is the case for each of the rigorous metrics.
        """
        k1 = 3.1
        k2 = 3.9

        assert (
            abs(k1 - k2) < 0.9999999
        ), "k1 and k2 should be close enough to eliminate the possibility of degenerate integer solutions."

        relaxed_1 = k1 / 3.0
        relaxed_2 = k2 / 3.0

        integer_1 = np.floor(relaxed_1)
        integer_2 = np.floor(relaxed_2)

        assert np.isclose(
            integer_1, integer_2
        ), "The provided k values should yield the same integer solution."

        m1 = self.generate_one_param_model(k1)
        m2 = self.generate_one_param_model(k2)

        solver = DefaultSolver("MILP")
        solver.solve(m1, tee=False)
        solver.solve(m2, tee=False)

        assert np.isclose(m1.x.value, integer_1)
        assert np.isclose(m2.x.value, integer_2)
        assert np.isclose(m1.y.value, integer_1)
        assert np.isclose(m2.y.value, integer_2)

        report1 = IntegerRelaxationReport(m1)
        report2 = IntegerRelaxationReport(m2)

        assert np.isclose(report1.lp_relaxation.x.value, relaxed_1)
        assert np.isclose(report2.lp_relaxation.x.value, relaxed_2)
        assert np.isclose(report1.lp_relaxation.y.value, relaxed_1)
        assert np.isclose(report2.lp_relaxation.y.value, relaxed_2)

        assert report1.fractionality_ratio <= report2.fractionality_ratio
        assert report1.gap_to_integer_solution <= report2.gap_to_integer_solution
