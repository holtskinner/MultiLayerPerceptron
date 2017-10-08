import Foundation
import struct MLP.Network
import func MLP.round

func partA() {
    
    let learningRate = 0.7
    let momentum = 0.3

    let b1 = parseBias("b1.csv")
    let b2 = parseBias("b2.csv")
    
    let w1 = parseWeight("w1.csv")
    let w2 = parseWeight("w2.csv")
    
    let crossData = parseWeight("cross_data.csv")
    
    let network = Network(learningRate: learningRate, momentum: momentum)
    network.addLayer(weights: w1, bias: b1)
    network.addLayer(weights: w2, bias: b2)
    
    var averageErrorEnergies: [Double] = [Double]()
    var averageErrorEnergy: Double = 0
    var previousErrorEnergy: Double = 0

    var epochNumber = 1

    repeat {
        previousErrorEnergy = averageErrorEnergy
        (averageErrorEnergies, averageErrorEnergy) = network.train(inputs: crossData)
        print("Epoch \(epochNumber) \(averageErrorEnergy)")
        epochNumber += 1
    } while previousErrorEnergy - averageErrorEnergy > 0.001
    
    var percentCorrect: Double = 0
    
    for var data in crossData {
        let expectedValue = data.removeLast()
        let actualValue = round(network.process(input: data)[0], toDecimalPlaces: 4)
        percentCorrect += 1 - (expectedValue - actualValue)
        print("\(expectedValue) \(actualValue)")
    }
    
    percentCorrect = (percentCorrect / Double(crossData.count)) * 100
    
    print("Percent Correct for Part A \(round(percentCorrect))%")
    
}
