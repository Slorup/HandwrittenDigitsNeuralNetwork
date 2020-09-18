import org.ejml.simple.SimpleMatrix
import kotlin.math.max
import kotlin.random.Random

class NeuralNetwork {
    private class Layer(val weights: SimpleMatrix, val bias: SimpleMatrix, initRandom: Boolean = false) {
        init {
            if (initRandom)
                for (i in 0 until weights.numElements)
                    weights[i] = Random.nextDouble(-1.0,1.0)
        }

        val nodeCount = weights.numCols()

        fun zeroVersion(): Layer = Layer(SimpleMatrix(weights.numRows(), weights.numCols()), SimpleMatrix(bias.numRows(), 1))
    }

    private val layers = listOf(
            Layer(SimpleMatrix(0,0), SimpleMatrix(0,0)),
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


        for (l in layers.reversed()) {
            val w = l.weights

            if (l == layers.last()) {

            }


            for (i in 0 until w.numRows()) {
                for (j in 0 until w.numCols()) {

                }
            }
        }
        return deltaLayers
    }

    private fun evaluate(id: ImageData): Map<Layer, SimpleMatrix> {
        val activations = mutableMapOf<Layer, SimpleMatrix>()

        activations[layers.first()] = SimpleMatrix(layers.first().nodeCount, 1)
        id.data.withIndex().forEach { activations[layers.first()]!![it.index] = it.value.toDouble() } // Initialize activations in first layer

        for ((current, previous) in layers.drop(1).withIndex().map { Pair(it.value, layers[it.index-1]) }) { // For each layer, multiply previous activations by the weights and add bias and apply the relu function: relu(a * w + b)
            activations[current] = current.weights.mult(activations[previous]).plus(current.bias) // Calc a * w + b (also called z). li is incremented

            for (i in 0 until activations[current]!!.numRows())
                activations[current]!![i] = relu(activations[current]!![i]) // Apply relu function to each neuron on the new layer
        }

        return activations
    }
}