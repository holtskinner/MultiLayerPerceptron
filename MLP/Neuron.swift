import Foundation

class Neuron {
    
    // incoming Weights & Biases
    var weights: [Double]
    var bias: Double

    var input: [Double]
    var output: Double

    var localGradient: Double
    var previousChangeInWeight: [Double]
    var previousChangeInBias: Double

    init(weights: [Double], bias: Double) {
        self.weights = weights
        self.bias = bias
        self.input = Array(repeating: 0, count: weights.count)
        self.output = 0
        self.localGradient = 0
        self.previousChangeInWeight = Array(repeating: 0, count: weights.count)
        self.previousChangeInBias = 0
    }

    func process(input: [Double]) -> Double {

        for i in 0..<input.count {
            self.input[i] = input[i]
            self.output += (input[i] * weights[i]) + bias
        }

        self.output = activation(input: output)

        return output

    }
    
    func update(error: Double, learningRate: Double, momentum: Double) {

        self.localGradient = error * partialDerivative()

        for i in 0..<weights.count {

            let changeInWeight = (learningRate * localGradient * input[i]) + (momentum * previousChangeInWeight[i])
            previousChangeInWeight[i] = changeInWeight
            weights[i] += changeInWeight

        }

        let changeInBias = (learningRate * localGradient) + (momentum * previousChangeInBias)
        previousChangeInBias = changeInBias
        bias += changeInBias

    }
    
    func activation(input: Double) -> Double {
        return (1 / (1 + exp(-input)))
    }

    func partialDerivative() -> Double {
        return output * (1 - output)
    }

}
