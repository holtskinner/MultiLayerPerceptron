import struct MLP.Neuron
import Foundation

class Network {
    
    var learningRate: Double
    var momentum: Double

    var layers: [[Neuron]]
    
    init(learningRate: Double, momentum: Double) {
        self.learningRate = learningRate
        self.momentum = momentum
        self.layers = [[Neuron]]()
    }
    
    func addLayer(weights: [[Double]], bias: [Double]) {

        var layer = [Neuron]()

        for i in 0..<weights.count {
            layer.append(Neuron(weights: weights[i], bias: bias[i]))
        }

        layers.append(layer)

    }
    
    func process(input: [Double]) -> [Double] {

        var output = [[Double]]()
        output.append(input)

        for i in 0..<layers.count {

            output.append([Double]())

            for neuron in layers[i] {
                output[i + 1].append(neuron.process(input: output[i])) // output of previous layer becomes input of current layer
            }

        }

        return output.last!

    }
    
    func train(inputs: [[Double]]) -> ([Double], Double) {

        var errorEnergy: [Double] = Array(repeating: 0, count: inputs.count)
        var averageErrorEnergy: Double = 0

        for (index, var input) in inputs.enumerated() {

            let expectedValue = input.removeLast()
            let actualValues = process(input: input)

            // update output layer
            for i in 0..<actualValues.count {
                let error = expectedValue - actualValues[i]
                var outputNeuron = layers.last![i]
                outputNeuron.update(error: error, learningRate: learningRate, momentum: momentum)
                errorEnergy[index] = pow(error, 2)
            }

            // Go backwards through the layers, skipping the output layer
            for i in (layers.count - 2)...0 {

                var currentLayer = layers[i]
                var nextLayer = layers[i + 1]
                
                // cycle through neurons in current layer
                for j in 0..<currentLayer.count {

                    var error: Double = 0

                    // cycle through neurons in next layer, picking up gradients
                    for k in 0..<nextLayer.count {
                        error += (nextLayer[k].localGradient * nextLayer[k].weights[j])
                    }

                    currentLayer[j].update(error: error, learningRate: learningRate, momentum: momentum)
                    errorEnergy[index] += pow(error, 2)

                }

            }

            averageErrorEnergy += errorEnergy[index]

        }
        
        return (errorEnergy, (averageErrorEnergy / Double(inputs.count)))

    }
    
}
