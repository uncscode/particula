import unittest
from particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder import BrownianCoagulationBuilder
from particula.dynamics.coagulation.coagulation_strategy import BrownianCoagulationStrategy

class TestBrownianCoagulationBuilder(unittest.TestCase):
    def test_build_with_valid_parameters(self):
        builder = BrownianCoagulationBuilder()
        builder.set_distribution_type("discrete")
        strategy = builder.build()
        self.assertIsInstance(strategy, BrownianCoagulationStrategy)

    def test_build_missing_required_parameters(self):
        builder = BrownianCoagulationBuilder()
        with self.assertRaises(ValueError):
            _ = builder.build()

if __name__ == "__main__":
    unittest.main()
