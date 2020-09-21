import org.ejml.simple.SimpleMatrix

const val IMAGE_SIZE = 28*28

fun main(args: Array<String>){
    val images = loadData()

    val nn = NeuralNetwork()

    nn.train(images.take(1000))

    println(nn.evaluate(images[1000]))
    println(images[1000].label)



}

fun loadData(): List<ImageData> {
    val imagesBytes = ImageData::class.java.getResource("train-images.idx3-ubyte").readBytes()
    val labelsBytes = ImageData::class.java.getResource("train-labels.idx1-ubyte").readBytes()

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