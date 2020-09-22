import org.ejml.simple.SimpleMatrix
import java.time.Instant
import java.util.*
import java.util.concurrent.Executors
import kotlin.random.Random

const val IMAGE_SIZE = 28*28

val random = Random(Instant.now().epochSecond)

fun main(args: Array<String>){

    val trainData = loadData("train-images.idx3-ubyte", "train-labels.idx1-ubyte")

    val nn = NeuralNetwork(listOf(IMAGE_SIZE, 16, 16, 5), Math.PI * Math.PI)

    val endTime = System.currentTimeMillis() + 12 * 60000

    while (System.currentTimeMillis() < endTime) {
        for (batch in trainData.chunked(80)) {
            nn.train(batch.map { Pair(imageToInputActivations(it), numToFeatures(it.label)) })
        }
        println("\rTime remaining: ${(endTime - System.currentTimeMillis()) / 1000} seconds")
    }

    val testData = loadData("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    var corrects = 0
    for (dataPoint in testData) {
        val actual = featuresToNum(nn.evaluate(imageToInputActivations(dataPoint)))
        if (actual == dataPoint.label)
            corrects++
    }

    println("Correct: ${corrects}\tIncorrect ${testData.size - corrects}\tRate ${corrects / testData.size.toDouble()}")



}

fun loadData(imagesPath: String, labelsPath: String): List<ImageData> {
    val imagesBytes = ImageData::class.java.getResource(imagesPath).readBytes()
    val labelsBytes = ImageData::class.java.getResource(labelsPath).readBytes()

    val images = mutableListOf<ImageData>()

    var l = 8
    for (i in 16 until imagesBytes.size step IMAGE_SIZE) {
        images.add(ImageData(imagesBytes.copyOfRange(i, i + IMAGE_SIZE), labelsBytes[l]))
        l++
    }

    return images
}

val featureNumMap = mapOf<Int, Int>(
        0b01000 to 0,
        0b00010 to 1,
        0b00110 to 2,
        0b10101 to 3,
        0b10011 to 4,
        0b00101 to 5,
        0b01011 to 6,
        0b10110 to 7,
        0b01001 to 8,
        0b11011 to 9
)

fun featuresToNum(features: SimpleMatrix): Int {
    var f = 0
    for (i in features.numRows() - 1 downTo 0) {
        f = f shl 1
        f += if (features[i, 0] > 0.5f) 1 else 0
    }

    return featureNumMap[f] ?: -1
}

fun numToFeatures(n: Int): SimpleMatrix {
    val m = SimpleMatrix(5, 1)
    var num = featureNumMap.entries.associate { (k,v) -> v to k }[n]!!
    for (i in 0 until 5) {
        m[i, 0] = (num and 0b1).toDouble()
        num = num shr 1
    }
    return m
}

fun SimpleMatrix.allElements(): Iterable<Double> {
    val list = mutableListOf<Double>()

    for (i in 0 until this.numElements)
        list.add(this[i])

    return list
}

fun imageToInputActivations(id: ImageData): SimpleMatrix {
    val inputActivations = SimpleMatrix(IMAGE_SIZE, 1)
    id.data.withIndex().forEach { inputActivations[it.index, 0] = it.value.toDouble() }

    return inputActivations
}