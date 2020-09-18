import org.ejml.simple.SimpleMatrix
import java.util.function.DoubleToLongFunction
import kotlin.math.max
import kotlin.random.Random

class NeuralNetwork {
    private class Layer(val weights: SimpleMatrix, val bias: SimpleMatrix, initRandom: Boolean = false) {
        init {
            if (initRandom)
                for (i in 0 until weights.numElements)
                    weights[i] = Random.nextDouble(-1.0,1.0)
        }

        fun zeroVersion(): Layer = Layer(SimpleMatrix(weights.numRows(), weights.numCols()), SimpleMatrix(bias.numRows(), 1))

    }

    private val layers = listOf(
        Layer(SimpleMatrix(16,IMAGE_SIZE), SimpleMatrix(16, 1), true),
        Layer(SimpleMatrix(16,16), SimpleMatrix(16, 1), true),
        Layer(SimpleMatrix(5, 16), SimpleMatrix(5,1), true)
    )

    private fun relu(d: Double): Double = max(0.0, d)
    private fun relumark(d: Double): Double = if (d > 0) 1.0 else 0.0

    fun train(images: List<ImageData>) {
        for (i in images)
            trainDataPoint(i)
    }

    private fun trainDataPoint(i: ImageData): List<Layer> {
        val actual = evaluate(i)
        val expected = numToFeatures(i.label)

        val deltaLayers = layers.map { it.zeroVersion() }

        for (l in layers.size - 1 downTo 0) {
            val w = layers[l].weights
            for (i in 0 until w.numRows()) {
                for (j in 0 until w.numCols()) {

                }
            }
        }

    }

    fun evaluate(id: ImageData): SimpleMatrix {
        var activations = SimpleMatrix(IMAGE_SIZE, 1)

        id.data.withIndex().forEach { activations[it.index] = it.value.toDouble() }

        for (l in layers) {
            activations = l.weights.mult(activations).plus(l.bias)
            for (i in 0 until activations.numRows())
                activations[i] = relu(activations[i])
        }

        return activations
    }

}