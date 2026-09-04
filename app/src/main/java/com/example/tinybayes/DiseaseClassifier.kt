package com.example.tinybayes

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

class DiseaseClassifier(
    private val context: Context
) {

    companion object {
        private const val FEATURE_SIZE = 576
        private const val NUM_CLASSES = 3
    }

    private val module: Module

    // beta[class][feature]
    private val betas: Array<FloatArray>

    private val classNames = arrayOf(
        "Anthracnose",
        "CSSVD",
        "Healthy"
    )

    private val mean = floatArrayOf(
        0.485f,
        0.456f,
        0.406f
    )

    private val std = floatArrayOf(
        0.229f,
        0.224f,
        0.225f
    )

    init {

        module = Module.load(
            assetFilePath(
                context,
                "mobilenet_feature_extractor.pt"
            )
        )

        betas = loadBetas()
    }

    private fun loadBetas(): Array<FloatArray> {

        val bytes = context.assets
            .open("jacobi_betas.bin")
            .readBytes()

        val buffer = ByteBuffer
            .wrap(bytes)
            .order(ByteOrder.LITTLE_ENDIAN)

        val result = Array(NUM_CLASSES) {
            FloatArray(FEATURE_SIZE)
        }

        for (c in 0 until NUM_CLASSES) {

            for (i in 0 until FEATURE_SIZE) {

                result[c][i] = buffer.float
            }
        }

        return result
    }

    fun predict(bitmap: Bitmap): String {

        val resized = Bitmap.createScaledBitmap(
            bitmap,
            224,
            224,
            true
        )

        val inputTensor =
            TensorImageUtils.bitmapToFloat32Tensor(
                resized,
                mean,
                std
            )

        val outputTensor =
            module.forward(
                IValue.from(inputTensor)
            ).toTensor()

        val features =
            outputTensor.dataAsFloatArray

        if (features.size != FEATURE_SIZE) {

            throw RuntimeException(
                "Expected $FEATURE_SIZE features but got ${features.size}"
            )
        }

        var bestScore = Double.NEGATIVE_INFINITY
        var bestClass = 0

        for (c in 0 until NUM_CLASSES) {

            var linear = 0.0

            for (i in 0 until FEATURE_SIZE) {

                linear +=
                    features[i].toDouble() *
                            betas[c][i].toDouble()
            }

            val score = exp(linear)

            if (score > bestScore) {

                bestScore = score
                bestClass = c
            }
        }

//        Log.d(
//            "TinyBayes",
//            """
//            Features=${features.size}
//            Anthracnose=${exp(features.zip(betas[0]) { f, b -> f.toDouble() * b.toDouble() }.sum())}
//            CSSVD=${exp(features.zip(betas[1]) { f, b -> f.toDouble() * b.toDouble() }.sum())}
//            Healthy=${exp(features.zip(betas[2]) { f, b -> f.toDouble() * b.toDouble() }.sum())}
//            Prediction=${classNames[bestClass]}
//            """.trimIndent()
//        )

        return classNames[bestClass]
    }
}