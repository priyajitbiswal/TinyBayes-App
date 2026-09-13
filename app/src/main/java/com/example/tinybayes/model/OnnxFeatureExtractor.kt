package com.example.tinybayes.model

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

class OnnxFeatureExtractor(context: Context) {

    private val environment = OrtEnvironment.getEnvironment()

    private val session: OrtSession

    init {
        val modelFile = copyModelFiles(context)

        session = environment.createSession(
            modelFile.absolutePath
        )
    }

    private fun copyModelFiles(context: Context): File {

        val modelName = "mobilenet_v3_small_features.onnx"
        val dataName = "mobilenet_v3_small_features.onnx.data"

        val modelFile = File(context.filesDir, modelName)
        val dataFile = File(context.filesDir, dataName)

        if (!modelFile.exists()) {
            context.assets
                .open("models/$modelName")
                .use { input ->
                    modelFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
        }

        if (!dataFile.exists()) {
            context.assets
                .open("models/$dataName")
                .use { input ->
                    dataFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
        }

        return modelFile
    }

    fun extractFeatures(bitmap: Bitmap): FloatArray {

        val input = preprocess(bitmap)

        val inputTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(input),
            longArrayOf(1, 3, 224, 224)
        )

        inputTensor.use { tensor ->

            val inputs = mapOf(
                session.inputNames.first() to tensor
            )

            session.run(inputs).use { outputs ->

                val output = outputs[0].value

                @Suppress("UNCHECKED_CAST")
                val features = output as Array<FloatArray>

                require(features.size == 1) {
                    "Unexpected ONNX batch size: ${features.size}"
                }

                require(features[0].size == 576) {
                    "Expected 576 features, got ${features[0].size}"
                }

                return features[0]
            }
        }
    }

    private fun preprocess(bitmap: Bitmap): FloatArray {

        val size = minOf(bitmap.width, bitmap.height)
        val xOffset = (bitmap.width - size) / 2
        val yOffset = (bitmap.height - size) / 2

        val squareBitmap = if (bitmap.width == bitmap.height) {
            bitmap
        } else {
            Bitmap.createBitmap(bitmap, xOffset, yOffset, size, size)
        }

        val resized = Bitmap.createScaledBitmap(
            squareBitmap,
            224,
            224,
            true
        )

        try {
            val pixels = IntArray(224 * 224)

            resized.getPixels(
                pixels,
                0,
                224,
                0,
                0,
                224,
                224
            )

            val input = FloatArray(3 * 224 * 224)

            val mean = floatArrayOf(
                0.485f,
                0.456f,
                0.406f
            )

            val std = floatArrayOf(
                0.229f,
                0.224f,
                0.225f
            )

            val channelSize = 224 * 224

            for (y in 0 until 224) {
                for (x in 0 until 224) {

                    val pixel = pixels[y * 224 + x]

                    val r = ((pixel shr 16) and 0xFF) / 255f
                    val g = ((pixel shr 8) and 0xFF) / 255f
                    val b = (pixel and 0xFF) / 255f

                    val index = y * 224 + x

                    input[index] =
                        (r - mean[0]) / std[0]

                    input[channelSize + index] =
                        (g - mean[1]) / std[1]

                    input[2 * channelSize + index] =
                        (b - mean[2]) / std[2]
                }
            }

            return input

        } finally {
            if (squareBitmap !== bitmap) {
                squareBitmap.recycle()
            }
            if (resized !== bitmap && resized !== squareBitmap) {
                resized.recycle()
            }
        }
    }

    fun close() {
        session.close()
    }
}