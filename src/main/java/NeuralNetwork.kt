import org.ejml.simple.SimpleMatrix

class NeuralNetwork(layerNodeCount: List<Int>, var learningRate: Double) {
    private class Layer(var weights: SimpleMatrix, var bias: SimpleMatrix, initRandom: Boolean = false) {
        init {
            if (initRandom)
                for (i in 0 until weights.numElements)
                    weights[i] = random.nextDouble(-1.0, 1.0)
        }

        val nodeCount = weights.numRows()

        fun zeroVersion(): Layer = Layer(SimpleMatrix(weights.numRows(), weights.numCols()), SimpleMatrix(bias.numRows(), 1))
    }

    private val layers = layerNodeCount.zipWithNext().map { Layer(SimpleMatrix(it.second, it.first), SimpleMatrix(it.second, 1), true) }

    private fun relu(d: Double): Double = 1 / (1 + Math.exp(-d))
    private fun relumark(d: Double): Double = Math.exp(-d) / Math.pow(Math.exp(-d) + 1, 2.0)

    //                              Input       , Expected
    fun train(dataPoints: List<Pair<SimpleMatrix, SimpleMatrix>>) {
        val gradient = layers.map { it.zeroVersion() }

        for ((input, expected) in dataPoints) {
            val deltaLayer = trainDataPoint(input, expected)
            for ((gl, dl) in gradient.zip(deltaLayer)) {
                gl.weights = gl.weights.plus(dl.weights)
                gl.bias = gl.bias.plus(dl.bias)
            }
        }

        for ((l, g) in layers.zip(gradient)) {
            l.weights = l.weights.plus(g.weights.divide(dataPoints.size.toDouble()).scale(-learningRate))
            l.bias = l.bias.plus(g.bias.divide(dataPoints.size.toDouble()).scale(-learningRate))
        }
    }

    private fun trainDataPoint(input: SimpleMatrix, expected: SimpleMatrix): List<Layer> {
        val (activations, zvalues) = internalEvaluate(input)

        // Delta layers will be added in reverse order
        val deltaLayers = mutableListOf<Layer>()

        // The previous layers' activation influence on the cost function
        var layer_dcda = (activations.minus(expected)).scale(2.0)

        // Current and next as seen from the output side, so last layer is now layer 0
        for ((current, next) in layers.zip(listOf(null) + layers.dropLast(1)).reversed()) {
            val deltaWeights = SimpleMatrix(current.weights.numRows(), current.weights.numCols())
            val deltaBias = SimpleMatrix(current.bias.numRows(), 1)

            val nextzvalue = if (next == null) input else zvalues[next]!!

            for (i in 0 until current.weights.numRows()) {
                for (j in 0 until current.weights.numCols()) {
                    deltaWeights[i, j] = layer_dcda[i, 0] * relumark(zvalues[current]!![i, 0]) * relu(nextzvalue[j, 0])
                }

                deltaBias[i, 0] = layer_dcda[i, 0] * relumark(zvalues[current]!![i, 0])
            }

            deltaLayers.add(Layer(deltaWeights, deltaBias))

            // This code should be seen as calculating for the next iteration of this loop, therefore current and next
            // have been reassigned to nprevious and ncurrent
            if (next != null) {
                // Calculate this layers influence, used in next layer
                val (ncurrent, nprevious) = Pair(next, current)
                val newLayer_dcda = SimpleMatrix(ncurrent.nodeCount, 1)

                for (i in 0 until ncurrent.nodeCount) { // current
                    var sum = 0.0
                    for (j in 0 until nprevious.nodeCount) { // previous
                        sum += nprevious.weights[j, i] * relumark(zvalues[nprevious]!![j, 0]) * layer_dcda[j, 0]
                    }
                    newLayer_dcda[i] = sum
                }
                layer_dcda = newLayer_dcda
            }
        }
        // Restore correct order
        return deltaLayers.reversed()
    }

    fun evaluate(input: SimpleMatrix): SimpleMatrix = internalEvaluate(input).first

    private fun internalEvaluate(inputActivation: SimpleMatrix): Pair<SimpleMatrix, Map<Layer, SimpleMatrix>> {
        val zvalues = mutableMapOf<Layer, SimpleMatrix>()
        var currentActivations = inputActivation


        for (layer in layers) { // For each layer, multiply previous activations by the weights and add bias and apply the relu function: relu(a * w + b)
            currentActivations = layer.weights.mult(currentActivations).plus(layer.bias) // Calc a * w + b (also called z).

            zvalues[layer] = SimpleMatrix(currentActivations)

            for (i in 0 until currentActivations.numRows())
                currentActivations[i] = relu(currentActivations[i]) // Apply relu function to each neuron on the new layer
        }

        return Pair(currentActivations, zvalues)
    }
}