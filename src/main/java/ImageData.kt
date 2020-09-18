class ImageData(bytes: ByteArray, label: Byte) {
    val data: FloatArray
    val label: Int

    init {
        data = bytes.map { it.toUByte().toFloat() / 255f }.toFloatArray()
        this.label = label.toInt()
    }

    override fun toString(): String = data.toList().chunked(28).joinToString("") { it.joinToString("") {if (it < 0.01f) "X" else "W" } + "\n" } + "Expected $label\n"
}