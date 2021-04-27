"""
Structure of test should be:
1. Create your inputs
2. Execute the code being tested, capturing the output
3. Compare the output with an expected result

"""

import unittest
import pandas as pd

from project.func import energy_consumption2cost


class TestEnergyConsumption2cost(unittest.TestCase):
    def test_energy_consumption2cost(self):
        """

        """
        energy_prices = pd.DataFrame([[0.1, 0.12], [0.5, 0.55]], index=['Power', 'Natural gas'], columns=[2018, 2019])
        energy_consumption = pd.Series([10, 12, 14], index=['x', 'y', 'z'])
        result = energy_consumption2cost(energy_consumption, energy_prices)
        result_obj = []
        self.assertEqual(result, result_obj)


if __name__ == '__main__':
    unittest.main()

