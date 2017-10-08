import Foundation

func partB() {

    let learningRate = 0.7
    let momentum = 0.3

    let inputSize = 4
    let hiddenLayerSize = 5
    let outputSize = 2

    let b1 = initArray(size: hiddenLayerSize)
    let b2 = initArray(size: outputSize)

    var w1 = init2DArray(inputSize: inputSize, outputSize: hiddenLayerSize)
    var w2 = init2DArray(inputSize: hiddenLayerSize, outputSize: outputSize)

    var gaussian = parseGaussian("Two_Class_FourDGaussians500.txt")

    for i in 0..<gaussian.count {
        var last = gaussian[i].count - 1
        gaussian[i][last] -= 1.0
    }

    let network = Network(learningRate: learningRate, momentum: momentum)
    network.addLayer(weights: w1, bias: b1)
    network.addLayer(weights: w2, bias: b2)
    
    var averageErrorEnergies: [Double] = [Double]()
    var averageErrorEnergy: Double = 0
    var previousErrorEnergy: Double = 0
    
    var epochNumber = 1
    
    repeat {
        previousErrorEnergy = averageErrorEnergy
        (averageErrorEnergies, averageErrorEnergy) = network.train(inputs: gaussian)
        print("Epoch \(epochNumber) \(averageErrorEnergy)")
        epochNumber += 1
    } while abs(previousErrorEnergy - averageErrorEnergy) > 0.0001

    var percentCorrect: Double = 0
    
    for var data in gaussian {
        let expectedValue = data.removeLast()
        let actualValue = round(network.process(input: data)[0])
        
        if expectedValue == actualValue {
            percentCorrect += 1
        }
        print("\(expectedValue) \(actualValue)")
    }
    
    percentCorrect = (percentCorrect / Double(gaussian.count)) * 100
    
    print("Percent Correct for Part B \(round(percentCorrect))%")

}

func initArray(size: Int) -> [Double] {
    return Array(repeating: 0, count: size)
}

func init2DArray(inputSize: Int, outputSize: Int) -> [[Double]] {
    
    var weights = [[Double]]()
    
    for _ in 0..<outputSize {
        weights.append(initArray(size: inputSize))
    }
    
    return weights

}
