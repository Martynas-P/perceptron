from operator import mul


class Perceptron:

    def __init__(self, learning_rate=0.01, initial_weights=None):
        self.learning_rate = learning_rate

        # we will assume that the last item in the list is the bias
        # therefore it the input for it will always be 1
        self.weights = initial_weights

    def train(self, data):
        """Performs a single iteration of updating the weights"""

        for input, desired_outcome in data:
            actual_outcome = self.calculate_output(input)

            self.weights = [weight + self.learning_rate * (desired_outcome - actual_outcome) * (input + [1])[index] for index, weight in enumerate(self.weights)]

    def calculate_output(self, input):
        input_vector = input + [1] # 1 for the bias

        result = sum(map(mul, input_vector, self.weights))

        return 0 if result < 0 else 1
