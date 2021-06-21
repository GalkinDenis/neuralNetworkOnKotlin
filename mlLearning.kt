import java.math.BigDecimal
import kotlin.math.exp
import kotlin.math.pow
import kotlin.random.Random.Default.nextDouble


fun main() {

    //OUTPUT ON CONSOLE
    //State of net from start
    println("\nState of net from start:" + "\n" +
            "1st hidden neuron weight to 1st input: " + weightsHiddenNeuron1[0] + "\n" +
            "1st hidden neuron weight to 2st input: " + weightsHiddenNeuron1[1] + " | output neuron weight to 1st hidden neuron: " + weightsOutputNeuron[0] + "\n" +
            "1st hidden neuron weight to 3st input: " + weightsHiddenNeuron1[2] + "\n" +
            "\n" +
            "2st hidden neuron weight to 1st input: " + weightsHiddenNeuron2[0] + "\n" +
            "2st hidden neuron weight to 2st input: " + weightsHiddenNeuron2[1] + " | output neuron weight to 2st hidden neuron: " + weightsOutputNeuron[1] + "\n" +
            "2st hidden neuron weight to 3st input: " + weightsHiddenNeuron2[2] + "\n" +
            "\n"
    )
    //OUTPUT ON CONSOLE

    //OUTPUT ON CONSOLE
    //Predict before training
    println("Predict before training:")
    for(a in trainingData.indices) {
        println(
            "Input set: " + "[" + trainingData[a][0] + " " + trainingData[a][1] + " " + trainingData[a][2] + "] " +
                "expected value: " + expectedValue[a] + " | actual predict value: ${predict(trainingData[a])}"
        )
    }
    //OUTPUT ON CONSOLE


    //Training
    println("\ntraining..." + "\n")
    for(c in 0..epoch) {
        lossSqrt.clear()
        for(b in trainingData.indices) {
            training(trainingData[b], expectedValue[b], c, b)
        }
        /*
        //DETAILS
        println("Mean square error of epoch #$c: ${meanSquareError(lossSqrt)}\n")
        //DETAILS
         */
        sumOfLoss = 0.0
    }

    //OUTPUT ON CONSOLE
    //State of net after training
    println("\nState of net after training:" + "\n" +
            "1st hidden neuron weight to 1st input: " + weightsHiddenNeuron1[0] + "\n" +
            "1st hidden neuron weight to 2st input: " + weightsHiddenNeuron1[1] + " | output neuron weight to 1st hidden neuron: " + weightsOutputNeuron[0] + "\n" +
            "1st hidden neuron weight to 3st input: " + weightsHiddenNeuron1[2] + "\n" +
            "\n" +
            "2st hidden neuron weight to 1st input: " + weightsHiddenNeuron2[0] + "\n" +
            "2st hidden neuron weight to 2st input: " + weightsHiddenNeuron2[1] + " | output neuron weight to 2st hidden neuron: " + weightsOutputNeuron[1] + "\n" +
            "2st hidden neuron weight to 3st input: " + weightsHiddenNeuron2[2] + "\n"
    )
    //OUTPUT ON CONSOLE

    //OUTPUT ON CONSOLE
    //Predict after training
    println("Predict after training:")
    for(a in trainingData.indices) {
        println("Input set: " + "[" + trainingData[a][0] + " " + trainingData[a][1] + " " + trainingData[a][2] + "] " +
                "expected value: " + expectedValue[a] + " | actual predict value: ${predict(trainingData[a])}"
        )
    }
    //OUTPUT ON CONSOLE

}

const val learningRate = 0.07
const val epoch = 6000

//Sets of training
val trainingData = arrayOf(
    arrayOf(0.0, 0.0, 0.0),
    arrayOf(0.0, 0.0, 1.0),
    arrayOf(0.0, 1.0, 0.0),
    arrayOf(0.0, 1.0, 1.0),
    arrayOf(1.0, 0.0, 0.0),
    arrayOf(1.0, 0.0, 1.0),
    arrayOf(1.0, 1.0, 0.0),
    arrayOf(1.0, 1.0, 1.0),
    )

//Expected values for each set training
val expectedValue = arrayOf(
    0.0,
    1.0,
    0.0,
    0.0,
    1.0,
    1.0,
    0.0,
    1.0
)

var sumOfHiddenNeuron1 = 0.0
var sumOfHiddenNeuron2 = 0.0
var sumOfOutputNeuron = 0.0
var sumOfLoss = 0.0

var activationValueOfOutputNeuron = 0.0

var weightsHiddenNeuron1: Array<Double> = arrayOf(nextDouble(), nextDouble(), nextDouble())
var weightsHiddenNeuron2: Array<Double> = arrayOf(nextDouble(), nextDouble(), nextDouble())
var weightsOutputNeuron: Array<Double> = arrayOf(nextDouble(), nextDouble())

var inputNeuronsValue = arrayOf(0.0, 0.0, 0.0)
var hiddenNeuronsWeightsAll = arrayOf(arrayOf(0.0, 0.0, 0.0), arrayOf(0.0, 0.0, 0.0))

var activationValuesOfAllHiddenNeurons = arrayOf(0.0, 0.0)
var errorOfAllHiddenNeurons = arrayOf(0.0, 0.0)

var lossSqrt = arrayListOf<Double>()

var weightDeltaOfOutputNeuron: Double = 0.0
var weightDeltaOfAllHiddenNeuron = arrayOf(0.0, 0.0)

fun training(trainingData: Array<Double>, expectedValue: Double, count: Int, trainingSet: Int) {
    inputNeuronsValue = arrayOf(trainingData[0], trainingData[1], trainingData[2])

    /*
    //DETAILS
    println("Epoch: $count\n" +
            "Training set: $trainingSet\n" +
            "Input neurons: " + trainingData[0] + " " + trainingData[1] + " " + trainingData[2])
    //DETAILS
     */

    hiddenNeuronsWeightsAll = arrayOf(
        weightsHiddenNeuron1,
        weightsHiddenNeuron2
    )

    sumOfHiddenNeuron1 = 0.0
    sumOfHiddenNeuron2 = 0.0
    for(a in hiddenNeuronsWeightsAll.indices) {
        for(b in inputNeuronsValue.indices) {
            when (a) {
                0 -> sumOfHiddenNeuron1 += inputNeuronsValue[b] * hiddenNeuronsWeightsAll[a][b]
                1 -> sumOfHiddenNeuron2 += inputNeuronsValue[b] * hiddenNeuronsWeightsAll[a][b]
            }
        }
    }

    activationValuesOfAllHiddenNeurons = arrayOf(
        activationFunction(sumOfHiddenNeuron1),
        activationFunction(sumOfHiddenNeuron2)
    )

    /*
    //DETAILS
    println("1st hidden neuron: sum of (weight * input) -> $sumOfHiddenNeuron1 | sigmoid(output) -> " + activationValuesOfAllHiddenNeurons[0] + "\n" +
            "2st hidden neuron: sum of (weight * input) -> $sumOfHiddenNeuron2 | sigmoid(output) -> " + activationValuesOfAllHiddenNeurons[1]
    )
    //DETAILS
     */

    sumOfOutputNeuron = 0.0
    for(a in activationValuesOfAllHiddenNeurons.indices) {
        sumOfOutputNeuron += (activationValuesOfAllHiddenNeurons[a] * weightsOutputNeuron[a])
    }

    activationValueOfOutputNeuron = activationFunction(sumOfOutputNeuron)

    /*
    //DETAILS
    println("Output neuron: sum of (weight * input) -> $sumOfOutputNeuron | sigmoid(output\in fact value) -> " + activationValueOfOutputNeuron.toBigDecimal() + " | predictable value -> " + expectedValue + "\n" +
            "Error: " + (activationValueOfOutputNeuron - expectedValue).pow(2).toBigDecimal() + "\n")
    //DETAILS
     */

    lossSqrt.add(activationValueOfOutputNeuron - expectedValue)

    weightDeltaOfOutputNeuron = weightsDelta(error(activationValueOfOutputNeuron, expectedValue), sigmoidDx(activationValueOfOutputNeuron))

    activationValueOfOutputNeuron = 0.0

    for(c in weightsOutputNeuron.indices) {
        weightsOutputNeuron[c] = backPropagation(weightsOutputNeuron[c], activationValuesOfAllHiddenNeurons[c], weightDeltaOfOutputNeuron, learningRate)
    }

    for(c in weightsOutputNeuron.indices) {
        errorOfAllHiddenNeurons[c] = weightsOutputNeuron[c] * weightDeltaOfOutputNeuron
    }

    for(c in hiddenNeuronsWeightsAll.indices) {
        weightDeltaOfAllHiddenNeuron[c] = weightsDelta(errorOfAllHiddenNeurons[c], sigmoidDx(activationValuesOfAllHiddenNeurons[c]))
        }

    for(c in hiddenNeuronsWeightsAll.indices) {
        for(b in weightsHiddenNeuron1.indices){
            hiddenNeuronsWeightsAll[c][b] = backPropagation(hiddenNeuronsWeightsAll[c][b], inputNeuronsValue[b], weightDeltaOfAllHiddenNeuron[c], learningRate)
            }
    }
}



fun predict(setOfData: Array<Double>): BigDecimal {
    inputNeuronsValue = arrayOf(setOfData[0], setOfData[1], setOfData[2])

    hiddenNeuronsWeightsAll = arrayOf(
        weightsHiddenNeuron1,
        weightsHiddenNeuron2
    )

    sumOfHiddenNeuron1 = 0.0
    sumOfHiddenNeuron2 = 0.0
    for (a in hiddenNeuronsWeightsAll.indices) {
        for (b in inputNeuronsValue.indices) {
            when (a) {
                0 -> sumOfHiddenNeuron1 += inputNeuronsValue[b] * hiddenNeuronsWeightsAll[a][b]
                1 -> sumOfHiddenNeuron2 += inputNeuronsValue[b] * hiddenNeuronsWeightsAll[a][b]
            }
        }
    }

    activationValuesOfAllHiddenNeurons = arrayOf(
        activationFunction(sumOfHiddenNeuron1),
        activationFunction(sumOfHiddenNeuron2)
    )

    sumOfOutputNeuron = 0.0
    for (a in activationValuesOfAllHiddenNeurons.indices) {
        sumOfOutputNeuron += activationValuesOfAllHiddenNeurons[a] * weightsOutputNeuron[a]
    }

    return activationFunction(sumOfOutputNeuron).toBigDecimal()
}

fun activationFunction(x: Double): Double {
    return 1 / (1 + exp(-x))
}

fun meanSquareError(array: ArrayList<Double>): Double {
    for(c in array.indices) {
        sumOfLoss += array[c].pow(2)
    }
    sumOfLoss /= array.size
    return sumOfLoss
}

fun error(actual: Double, expected: Double): Double = actual - expected
fun sigmoidDx(actual: Double): Double = actual * (1.0 - actual)
fun weightsDelta(error: Double, sigmoidDx: Double): Double = error * sigmoidDx
fun backPropagation(weight: Double, pastOutput: Double, weightDelta: Double, learningRate: Double): Double = weight - (pastOutput * weightDelta * learningRate)
