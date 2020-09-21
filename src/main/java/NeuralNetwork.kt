import org.ejml.simple.SimpleMatrix
import javax.imageio.ImageTranscoder
import kotlin.math.max
import kotlin.random.Random

class NeuralNetwork {
    private class Layer(var weights: SimpleMatrix, var bias: SimpleMatrix, initRandom: Boolean = false) {
        init {
            if (initRandom)
                for (i in 0 until weights.numElements)
                    weights[i] = Random.nextDouble(-1.0,1.0)
        }

        val nodeCount = weights.numCols()

        fun zeroVersion(): Layer = Layer(SimpleMatrix(weights.numRows(), weights.numCols()), SimpleMatrix(bias.numRows(), 1))
    }

    private val learningRate = 0.1

    private val layers = listOf(
            Layer(SimpleMatrix(), SimpleMatrix()),
            Layer(SimpleMatrix(16,IMAGE_SIZE), SimpleMatrix(16, 1), true),
            Layer(SimpleMatrix(16,16), SimpleMatrix(16, 1), true),
            Layer(SimpleMatrix(5, 16), SimpleMatrix(5,1), true)
    )

    private fun relu(d: Double): Double = max(0.0, d)
    private fun relumark(d: Double): Double = if (d > 0) 1.0 else 0.0

    fun train(images: List<ImageData>) {
        val gradient = layers.map { it.zeroVersion() }

        for (i in images) {
            val deltaLayer = trainDataPoint(i)
            for ((gl, dl) in gradient.zip(deltaLayer).drop(1)) {
                gl.weights = gl.weights.plus(dl.weights)
                gl.bias = gl.bias.plus(dl.bias)
            }
        }

        for ((l, g) in layers.zip(gradient)) {
            l.weights = l.weights.plus(g.weights.divide(images.size.toDouble()).scale(-learningRate))
            l.bias = l.weights.plus(g.bias.divide(images.size.toDouble()).scale(-learningRate))
        }
    }

    private fun trainDataPoint(id: ImageData): List<Layer> {
        val (activations, zvalues) = internalEvaluate(id)
        val expected = numToFeatures(id.label)

        val deltaLayers = layers.map { it.zeroVersion() }

        // The previous layers' activation influence on the cost function
        var layer_dcda = (activations.minus(expected)).scale(2.0)

        // Current and next as seen from the output side, so last layer is now layer 0
        for ((current, next) in layers.drop(1).reversed().withIndex().map { Pair(it.value, layers[it.index + 1]) }) {
            val deltaWeight = SimpleMatrix(current.weights.numRows(), current.weights.numCols())
            val deltaBias = SimpleMatrix(current.bias.numRows(), 1)

            for (i in 0 until current.weights.numRows()) {
                for (j in 0 until current.weights.numCols()) {
                    deltaWeight[i, j] = layer_dcda[i, 0] * relumark(zvalues[current]!![i, 0]) * relu(zvalues[next]!![j, 0])
                }

                deltaBias[i, 0] = layer_dcda[i, 0] * relumark(zvalues[current]!![i, 0])
            }

            layer_dcda = run {
                // Calculate this layers influence, used in next layer
                val (ncurrent, nprevious) = Pair(next, current)
                val newLayer_dcda = SimpleMatrix(layer_dcda.numRows(), layer_dcda.numCols())

                for (i in 0 until ncurrent.nodeCount) { // current
                    var sum = 0.0
                    for (j in 0 until nprevious.nodeCount) { // previous
                        sum += nprevious.weights[j, i] * relumark(zvalues[nprevious]!![j, 0]) * layer_dcda[j, 0]
                    }
                    newLayer_dcda[i] = sum
                }
                newLayer_dcda
            }
        }
        return deltaLayers
    }

    fun evaluate(id: ImageData): SimpleMatrix = internalEvaluate(id).first

    private fun internalEvaluate(id: ImageData): Pair<SimpleMatrix, Map<Layer, SimpleMatrix>> {
        val zvalues = mutableMapOf<Layer, SimpleMatrix>()
        var currentActivations = SimpleMatrix(layers.first().nodeCount, 1)

        id.data.withIndex().forEach { currentActivations[it.index, 0] = it.value.toDouble() } // Initialize activations in first layer

        for (layer in layers.drop(1)) { // For each layer, multiply previous activations by the weights and add bias and apply the relu function: relu(a * w + b)
            currentActivations = layer.weights.mult(currentActivations).plus(layer.bias) // Calc a * w + b (also called z).

            zvalues[layer] = currentActivations

            for (i in 0 until currentActivations.numRows())
                currentActivations[i] = relu(currentActivations[i]) // Apply relu function to each neuron on the new layer
        }

        return Pair(currentActivations, zvalues)
    }
}