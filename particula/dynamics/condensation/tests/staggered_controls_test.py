"""Staggered theta and batching contracts independent of legacy fixtures."""

import logging.handlers
import unittest

import numpy as np
import pytest
from particula.dynamics.condensation import CondensationIsothermalStaggered


@pytest.mark.usefixtures("forbid_legacy_facades")
class TestStaggeredControls(unittest.TestCase):
    """Preserve stochastic controls without constructing aerosol facades."""

    def test_defaults(self):
        """Defaults stored correctly for staggered strategy."""
        strategy = CondensationIsothermalStaggered(molar_mass=0.018)
        self.assertEqual(strategy.theta_mode, "half")
        self.assertEqual(strategy.num_batches, 1)
        self.assertTrue(strategy.shuffle_each_step)
        self.assertIsNone(strategy.random_state)

    def test_theta_mode_random_stores_seed(self):
        """Random theta mode stores provided random state."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=42
        )
        self.assertEqual(strategy.theta_mode, "random")
        self.assertEqual(strategy.random_state, 42)

    def test_theta_mode_batch_stores_batches(self):
        """Batch theta mode stores batch count."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="batch", num_batches=4
        )
        self.assertEqual(strategy.theta_mode, "batch")
        self.assertEqual(strategy.num_batches, 4)

    def test_invalid_theta_mode_raises(self):
        """Invalid theta_mode raises ValueError."""
        with self.assertRaises(ValueError):
            CondensationIsothermalStaggered(
                molar_mass=0.018, theta_mode="unsupported"
            )

    def test_num_batches_less_than_one_raises(self):
        """num_batches below one raises ValueError."""
        with self.assertRaises(ValueError):
            CondensationIsothermalStaggered(molar_mass=0.018, num_batches=0)

    def test_get_theta_values_half_mode_returns_half(self):
        """Half mode returns 0.5 with correct shape and dtype."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="half"
        )
        theta = strategy._get_theta_values(5)
        np.testing.assert_array_equal(theta, np.full(5, 0.5))
        self.assertEqual(theta.shape, (5,))
        self.assertEqual(theta.dtype, np.float64)

    def test_get_theta_values_random_mode_range_and_shape(self):
        """Random mode stays in [0, 1] with correct shape and dtype."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=123
        )
        theta = strategy._get_theta_values(100)
        self.assertTrue(np.all(theta >= 0.0))
        self.assertTrue(np.all(theta <= 1.0))
        self.assertEqual(theta.shape, (100,))
        self.assertEqual(theta.dtype, np.float64)

    def test_get_theta_values_random_mode_reproducible(self):
        """Random mode reproducible with same integer seed."""
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=42
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=42
        )
        theta_a = strategy_a._get_theta_values(50)
        theta_b = strategy_b._get_theta_values(50)
        np.testing.assert_array_equal(theta_a, theta_b)

    def test_get_theta_values_random_mode_different_seed(self):
        """Different seeds produce different theta arrays."""
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=1
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=2
        )
        theta_a = strategy_a._get_theta_values(20)
        theta_b = strategy_b._get_theta_values(20)
        self.assertFalse(np.allclose(theta_a, theta_b))

    def test_get_theta_values_random_mode_accepts_random_state(self):
        """Random mode supports np.random.RandomState."""
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="random",
            random_state=np.random.RandomState(7),
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="random",
            random_state=np.random.RandomState(7),
        )
        theta_a = strategy_a._get_theta_values(15)
        theta_b = strategy_b._get_theta_values(15)
        np.testing.assert_array_equal(theta_a, theta_b)

    def test_get_theta_values_random_mode_accepts_generator(self):
        """Random mode supports np.random.Generator."""
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="random",
            random_state=np.random.default_rng(9),
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="random",
            random_state=np.random.default_rng(9),
        )
        theta_a = strategy_a._get_theta_values(12)
        theta_b = strategy_b._get_theta_values(12)
        np.testing.assert_array_equal(theta_a, theta_b)

    def test_get_theta_values_batch_mode_returns_ones(self):
        """Batch mode returns ones with correct shape and dtype."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="batch"
        )
        theta = strategy._get_theta_values(8)
        np.testing.assert_array_equal(theta, np.ones(8))
        self.assertEqual(theta.shape, (8,))
        self.assertEqual(theta.dtype, np.float64)

    def test_get_theta_values_single_particle_all_modes(self):
        """Single particle handled for half, random, and batch modes."""
        strategy_half = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="half"
        )
        strategy_random = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=5
        )
        strategy_batch = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="batch"
        )

        theta_half = strategy_half._get_theta_values(1)
        theta_random = strategy_random._get_theta_values(1)
        theta_batch = strategy_batch._get_theta_values(1)

        np.testing.assert_array_equal(theta_half, np.array([0.5]))
        self.assertEqual(theta_half.shape, (1,))
        self.assertTrue(0.0 <= theta_random[0] <= 1.0)
        self.assertEqual(theta_random.shape, (1,))
        np.testing.assert_array_equal(theta_batch, np.array([1.0]))
        self.assertEqual(theta_batch.shape, (1,))

    def test_get_theta_values_zero_particles_returns_empty(self):
        """Zero particles returns empty float64 array."""
        for mode in ("half", "random", "batch"):
            strategy = CondensationIsothermalStaggered(
                molar_mass=0.018, theta_mode=mode, random_state=3
            )
            theta = strategy._get_theta_values(0)
            self.assertEqual(theta.shape, (0,))
            self.assertEqual(theta.dtype, np.float64)

    def test_get_theta_values_invalid_theta_mode_raises_value_error(self):
        """Mutating theta_mode to unsupported value triggers ValueError."""
        strategy = CondensationIsothermalStaggered(molar_mass=0.018)
        strategy.theta_mode = "unsupported"
        with self.assertRaises(ValueError):
            strategy._get_theta_values(3)

    def test_make_batches_correct_count(self):
        """_make_batches creates requested batch count when possible."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=4, shuffle_each_step=False
        )
        batches = strategy._make_batches(100)
        self.assertEqual(len(batches), 4)
        self.assertEqual(sum(len(batch) for batch in batches), 100)

    def test_make_batches_no_shuffle_when_disabled(self):
        """Shuffling disabled preserves the original order."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=1, shuffle_each_step=False
        )
        batches = strategy._make_batches(10)
        np.testing.assert_array_equal(batches[0], np.arange(10))

    def test_make_batches_shuffles_when_enabled(self):
        """Shuffling enabled permutes indices."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=1,
            shuffle_each_step=True,
            random_state=0,
        )
        batches = strategy._make_batches(20)
        self.assertFalse(np.array_equal(batches[0], np.arange(20)))

    def test_make_batches_preserves_all_indices(self):
        """All indices appear exactly once across batches."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=5,
            shuffle_each_step=True,
            random_state=123,
        )
        batches = strategy._make_batches(57)
        all_indices = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(57))

    def test_make_batches_reproducible(self):
        """Same seed yields identical batches."""
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=3,
            shuffle_each_step=True,
            random_state=21,
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=3,
            shuffle_each_step=True,
            random_state=21,
        )
        batches_a = strategy_a._make_batches(20)
        batches_b = strategy_b._make_batches(20)
        for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
            np.testing.assert_array_equal(batch_a, batch_b)

    def test_make_batches_single_batch(self):
        """num_batches=1 returns a single full batch."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=1, shuffle_each_step=False
        )
        batches = strategy._make_batches(7)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 7)

    def test_validate_num_batches_zero_raises_value_error(self):
        """_validate_num_batches raises when requested batches are zero."""
        strategy = CondensationIsothermalStaggered(molar_mass=0.018)
        strategy.num_batches = 0
        with self.assertRaisesRegex(
            ValueError, r"num_batches.*(>=|at least)\s*1"
        ):
            strategy._make_batches(5)

    def test_validate_num_batches_negative_raises_value_error(self):
        """_validate_num_batches raises when requested batches are negative."""
        with self.assertRaisesRegex(
            ValueError, r"num_batches.*(>=|at least)\s*1"
        ):
            CondensationIsothermalStaggered(molar_mass=0.018, num_batches=-1)

    def test_make_batches_more_batches_than_particles(self):
        """Excess batches clip to particle count."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=10, shuffle_each_step=False
        )
        batches = strategy._make_batches(3)
        self.assertEqual(len(batches), 3)
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(3))
        self.assertTrue(all(len(batch) == 1 for batch in batches))

    def test_make_batches_single_particle(self):
        """Single particle returns one-element batch."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=4, shuffle_each_step=True
        )
        batches = strategy._make_batches(1)
        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(batches[0], np.array([0], dtype=np.intp))

    def test_make_batches_zero_particles_returns_empty(self):
        """Zero particles returns empty list."""
        strategy = CondensationIsothermalStaggered(molar_mass=0.018)
        batches = strategy._make_batches(0)
        self.assertEqual(batches, [])

    def test_make_batches_random_state_shuffle_path(self):
        """Uses RandomState shuffle path when provided."""
        random_state_a = np.random.RandomState(5)
        random_state_b = np.random.RandomState(5)
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=2,
            shuffle_each_step=True,
            random_state=random_state_a,
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=2,
            shuffle_each_step=True,
            random_state=random_state_b,
        )
        batches_a = strategy_a._make_batches(6)
        batches_b = strategy_b._make_batches(6)
        self.assertFalse(
            np.array_equal(np.concatenate(batches_a), np.arange(6))
        )
        for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
            np.testing.assert_array_equal(batch_a, batch_b)

    def test_make_batches_generator_shuffle_path(self):
        """Uses Generator shuffle path when provided."""
        generator_a = np.random.default_rng(7)
        generator_b = np.random.default_rng(7)
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=2,
            shuffle_each_step=True,
            random_state=generator_a,
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=2,
            shuffle_each_step=True,
            random_state=generator_b,
        )
        batches_a = strategy_a._make_batches(6)
        batches_b = strategy_b._make_batches(6)
        self.assertFalse(
            np.array_equal(np.concatenate(batches_a), np.arange(6))
        )
        for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
            np.testing.assert_array_equal(batch_a, batch_b)

    def test_num_batches_exceeds_particles_clips_and_logs_info(self):
        """Clips to particle count when batches exceed size and logs info."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=1000, shuffle_each_step=False
        )
        with self.assertLogs("particula", level=logging.INFO) as cm:
            batches = strategy._make_batches(5)
        self.assertEqual(len(batches), 5)
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(5))
        # Verify clipping log message is present.
        self.assertTrue(
            any(
                "Clipping num_batches" in record.message
                for record in cm.records
            )
        )

    def _assert_no_clipping_log(self, batches_func, *args, **kwargs):
        """Helper to verify no clipping log is emitted during batch creation.

        Args:
            batches_func: Function to call that creates batches.
            *args: Positional arguments to pass to batches_func.
            **kwargs: Keyword arguments to pass to batches_func.

        Returns:
            The result from batches_func.
        """
        handler = logging.handlers.MemoryHandler(capacity=100)
        particula_logger = logging.getLogger("particula")
        particula_logger.addHandler(handler)
        original_level = particula_logger.level
        particula_logger.setLevel(logging.DEBUG)
        try:
            result = batches_func(*args, **kwargs)
            handler.flush()
            # Check that no clipping log was emitted.
            self.assertFalse(
                any(
                    "Clipping num_batches" in record.getMessage()
                    for record in handler.buffer
                )
            )
            return result
        finally:
            particula_logger.removeHandler(handler)
            particula_logger.setLevel(original_level)

    def test_num_batches_equals_particles_no_log(self):
        """Equal batches and particles should not clip or log."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=5, shuffle_each_step=False
        )
        batches = self._assert_no_clipping_log(strategy._make_batches, 5)
        self.assertEqual(len(batches), 5)
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(5))

    def test_num_batches_one_creates_single_batch(self):
        """Single batch should include all particles and avoid logging."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=1, shuffle_each_step=False
        )
        batches = self._assert_no_clipping_log(strategy._make_batches, 12)
        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(batches[0], np.arange(12))

    def test_zero_particles_returns_empty_batches_no_log(self):
        """Zero particles returns empty list without logging."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=3, shuffle_each_step=False
        )
        batches = self._assert_no_clipping_log(strategy._make_batches, 0)
        self.assertEqual(batches, [])
