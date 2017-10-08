import Foundation

func round(_ value: Double, toDecimalPlaces places: Int) -> Double {
    let divisor = pow(10.0, Double(places))
    return round(value * divisor) / divisor
}
