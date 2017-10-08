import Foundation
import struct MLP.Network

func partA() {

    let b1 = parseBias("b1.csv")
    let b2 = parseBias("b2.csv")
    
    let w1 = parseWeight("w1.csv")
    let w2 = parseWeight("w2.csv")
    
    var crossData = parseWeight("cross_data.csv")
    
    var network = Network(learningRate: 0.7, momentum: 0.3)
    network.addLayer(weights: w1, bias: b1)
    network.addLayer(weights: w2, bias: b2)
    
    var averageErrorEnergies: [Double] = [Double]()
    var averageErrorEnergy: Double = 0

    var i = 1
    repeat {
        (averageErrorEnergies, averageErrorEnergy) = network.train(inputs: crossData)
        print("Epoch \(i) \(averageErrorEnergy)")
        i += 1
    } while averageErrorEnergy > 0.001

    var percentCorrect: Double = 0

    for var data in crossData {
        var expectedValue = data.removeLast()
        var actualValue = round(network.process(input: data)[0])
        percentCorrect += 1 - (expectedValue - actualValue)
        print("\(expectedValue) \(actualValue)")
    }

    percentCorrect = (percentCorrect / Double(crossData.count)) * 100

    print("Percent Correct for Part A \(percentCorrect)%")

}

partA()
