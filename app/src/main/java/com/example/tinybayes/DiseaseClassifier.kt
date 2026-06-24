package com.example.tinybayes

import android.content.Context
import android.graphics.Bitmap
import com.google.gson.Gson
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class DiseaseClassifier(context: Context) {

    private val mobilenet: Interpreter

    private val healthy: FloatArray
    private val cssvd: FloatArray
    private val anthracnose: FloatArray

    init {

        mobilenet = Interpreter(
            FileUtil.loadMappedFile(
                context,
                "mobilenet_v3_small.tflite"
            )
        )

        val json = context.assets
            .open("jacobi_coefficients.json")
            .bufferedReader()
            .use { it.readText() }

        val coeffs = Gson().fromJson(
            json,
            JacobiCoefficients::class.java
        )

        healthy = coeffs.healthy
            .map { it.toFloat() }
            .toFloatArray()

        cssvd = coeffs.cssvd
            .map { it.toFloat() }
            .toFloatArray()

        anthracnose = coeffs.anthracnose
            .map { it.toFloat() }
            .toFloatArray()
    }

    fun predict(bitmap: Bitmap): String {

        val resized = Bitmap.createScaledBitmap(
            bitmap.copy(Bitmap.Config.ARGB_8888, true),
            224,
            224,
            true
        )

        val input = ByteBuffer.allocateDirect(
            1 * 224 * 224 * 3 * 4
        ).order(ByteOrder.nativeOrder())

        for (y in 0 until 224) {
            for (x in 0 until 224) {

                val pixel = resized.getPixel(x, y)

                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                val rn = (r - 0.485f) / 0.229f
                val gn = (g - 0.456f) / 0.224f
                val bn = (b - 0.406f) / 0.225f

                input.putFloat(rn)
                input.putFloat(gn)
                input.putFloat(bn)
            }
        }

        input.rewind()

        val output = Array(1) {
            FloatArray(576)
        }

        mobilenet.run(input, output)

        val features = output[0]

        val healthyScore = dot(features, healthy)
        val cssvdScore = dot(features, cssvd)
        val anthracnoseScore = dot(features, anthracnose)

        return when {
            healthyScore >= cssvdScore &&
                    healthyScore >= anthracnoseScore -> "Healthy"

            cssvdScore >= healthyScore &&
                    cssvdScore >= anthracnoseScore -> "CSSVD"

            else -> "Anthracnose"
        }
    }

    private fun dot(
        a: FloatArray,
        b: FloatArray
    ): Float {

        var sum = 0f

        for (i in a.indices) {
            sum += a[i] * b[i]
        }

        return sum
    }
}