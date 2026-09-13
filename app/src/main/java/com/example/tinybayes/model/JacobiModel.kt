package com.example.tinybayes.model

import android.content.Context
import org.json.JSONObject
import kotlin.math.exp

data class PredictionResult(
    val className: String,
    val score: Double
)

class JacobiModel(
    context: Context,
    coefficientPath: String
) {

    private val coefficients: Map<String, FloatArray>

    init {
        val jsonText = context.assets
            .open(coefficientPath)
            .bufferedReader()
            .use { it.readText() }

        val json = JSONObject(jsonText)

        val loadedCoefficients = mutableMapOf<String, FloatArray>()

        val keys = json.keys()

        while (keys.hasNext()) {
            val className = keys.next()
            val array = json.getJSONArray(className)

            require(array.length() == 576) {
                "Class '$className' has ${array.length()} coefficients. Expected 576."
            }

            val beta = FloatArray(576)

            for (i in 0 until 576) {
                beta[i] = array.getDouble(i).toFloat()
            }

            loadedCoefficients[className] = beta
        }

        require(loadedCoefficients.isNotEmpty()) {
            "No Jacobi coefficients found in $coefficientPath"
        }

        coefficients = loadedCoefficients
    }

    fun predict(features: FloatArray): PredictionResult {

        require(features.size == 576) {
            "Expected 576 features, got ${features.size}"
        }

        var bestClass: String? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for ((className, beta) in coefficients) {

            var dotProduct = 0.0

            for (i in 0 until 576) {
                dotProduct += features[i].toDouble() * beta[i].toDouble()
            }

            val score = exp(dotProduct)

            if (score > bestScore) {
                bestScore = score
                bestClass = className
            }
        }

        return PredictionResult(
            className = bestClass
                ?: error("Unable to determine prediction"),
            score = bestScore
        )
    }

    fun getClassNames(): List<String> {
        return coefficients.keys.toList()
    }
}