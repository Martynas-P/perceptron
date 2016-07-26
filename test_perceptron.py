import unittest

from perceptron import Perceptron


class OutputTestCase(unittest.TestCase):

    def test_calculate_output1(self):
        perceptron = Perceptron(initial_weights=[1.0])

        self.assertEqual(perceptron.calculate_output([1.0]), 1)
        self.assertEqual(perceptron.calculate_output([0.0]), 1)
        self.assertEqual(perceptron.calculate_output([-100.0]), 0)

    def test_calculate_output2(self):
        perceptron = Perceptron(initial_weights=[1.0, 1.0, 1.0])

        self.assertEqual(perceptron.calculate_output([10.0, -5.0]), 1)
        self.assertEqual(perceptron.calculate_output([0.0, -2.0]), 0)
        self.assertEqual(perceptron.calculate_output([0.0, -0.5]), 1)
        self.assertEqual(perceptron.calculate_output([0.0, 0.0]), 1)
        self.assertEqual(perceptron.calculate_output([-1.0, -1.0]), 0)


class TrainingTestCase(unittest.TestCase):

    def test_weights_dont_change(self):
        perceptron = Perceptron(initial_weights=[0.0, 0.0, -1.0])

        data = [
            ([0, 0], 0),
        ]

        perceptron.train(data)

        self.assertAlmostEquals(perceptron.weights[0], 0.0)
        self.assertAlmostEquals(perceptron.weights[1], 0.0)
        self.assertAlmostEquals(perceptron.weights[2], -1.0)

    def test_single_update1(self):
        perceptron = Perceptron(initial_weights=[0.0, 0.0, 0.0], learning_rate=0.01)

        data = [
            ([0, 0], 0),
        ]

        perceptron.train(data)

        self.assertAlmostEquals(perceptron.weights[0], 0.0)
        self.assertAlmostEquals(perceptron.weights[1], 0.0)
        self.assertAlmostEquals(perceptron.weights[2], -0.01)

    def test_single_update2(self):
        perceptron = Perceptron(initial_weights=[1.0, 1.0, 1.0], learning_rate=0.01)

        data = [
            ([1, 1], 0),
        ]

        perceptron.train(data)

        self.assertAlmostEquals(perceptron.weights[0], 0.99)
        self.assertAlmostEquals(perceptron.weights[1], 0.99)
        self.assertAlmostEquals(perceptron.weights[2], 0.99)

    def test_single_update3(self):
        perceptron = Perceptron(initial_weights=[0.0, 0.0, 0.0], learning_rate=0.01)

        # Truth table of an AND function
        data = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 1),
        ]

        perceptron.train(data)

        self.assertAlmostEquals(perceptron.weights[0], 0.0)
        self.assertAlmostEquals(perceptron.weights[1], 0.01)
        self.assertAlmostEquals(perceptron.weights[2], 0.0)

    def test_converges1(self):
        perceptron = Perceptron(initial_weights=[0.0, 0.0, 0.0], learning_rate=0.01)

        # Truth table of an OR function
        data = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 1),
        ]

        converged = False

        while not converged:
            perceptron.train(data)

            converged = True

            for input, expected_outcome in data:
                converged = converged and perceptron.calculate_output(input) == expected_outcome

    def test_converges2(self):
        perceptron = Perceptron(initial_weights=[0.0, 0.0, 0.0], learning_rate=0.01)

        # Truth table of an AND function
        data = [
            ([0, 0], 0),
            ([0, 1], 0),
            ([1, 0], 0),
            ([1, 1], 1),
        ]

        converged = False

        while not converged:
            perceptron.train(data)

            converged = True

            for input, expected_outcome in data:
                converged = converged and perceptron.calculate_output(input) == expected_outcome

    def test_does_not_converge(self):
        perceptron = Perceptron(initial_weights=[0.0, 0.0, 0.0], learning_rate=0.01)

        # Truth table of an XOR function
        # This should never converge
        data = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0),
        ]

        for i in range(1000000):
            perceptron.train(data)

        converged = True

        for input, expected_outcome in data:
            converged = converged and perceptron.calculate_output(input) == expected_outcome

        self.assertFalse(converged)
