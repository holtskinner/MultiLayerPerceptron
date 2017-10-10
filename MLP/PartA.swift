import Foundation
import struct MLP.Network
import func MLP.round

func partA() {
    
    let b1 = parseBias("b1.csv")
    let b2 = parseBias("b2.csv")
    
    let w1 = parseWeight("w1.csv")
    let w2 = parseWeight("w2.csv")
    
    var crossData = parseWeight("cross_data.csv")
    
    let network = Network(learningRate: 0.7, momentum: 0.3)
    network.addLayer(weights: w1, bias: b1)
    network.addLayer(weights: w2, bias: b2)
    
    var averageErrorEnergies: [Double] = [Double]()
    var averageErrorEnergy: Double = 0
    var previousErrorEnergy: Double = 0

    var i = 1

    // train network
    repeat {
        previousErrorEnergy = averageErrorEnergy
        (averageErrorEnergies, averageErrorEnergy) = network.train(inputs: crossData)
        print("Epoch \(i) Error Energy: \(round(averageErrorEnergy, toDecimalPlaces: 4))")
        i += 1
    } while abs(previousErrorEnergy - averageErrorEnergy) > 0.001

    var percentCorrect: Double = 0

    // Run Training data back through net to see if it worked
    for var data in crossData {
        let expectedValue = data.removeLast()
        let actualValue = round(network.process(input: data)[0])
        percentCorrect += 1 - (expectedValue - actualValue)
    }
    
    percentCorrect = (percentCorrect / Double(crossData.count)) * 100
    
    print("Percent Correct for Part A \(percentCorrect)%")
    
}
