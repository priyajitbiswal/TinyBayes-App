package com.example.tinybayes.inference

import android.content.Context
import android.graphics.Bitmap
import com.example.tinybayes.model.JacobiModel
import com.example.tinybayes.model.OnnxFeatureExtractor

data class TinyBayesPrediction(
    val className: String,
    val confidence: Double,
    val probabilities: Map<String, Double>
)

class TinyBayesClassifier(context: Context) {

    private val featureExtractor = OnnxFeatureExtractor(context)

    private val jacobiModels = mapOf(
        "Cocoa" to JacobiModel(
            context,
            "models/cocoa/jacobi_coefficients.json"
        ),
        "Potato" to JacobiModel(
            context,
            "models/potato/jacobi_coefficients.json"
        ),
        "Tomato" to JacobiModel(
            context,
            "models/tomato/jacobi_coefficients.json"
        ),
        "Rice" to JacobiModel(
            context,
            "models/rice/jacobi_coefficients.json"
        ),
        "Cotton" to JacobiModel(
            context,
            "models/cotton/jacobi_coefficients.json"
        )
    )

    fun predict(
        bitmap: Bitmap,
        crop: String
    ): TinyBayesPrediction {

        val jacobiModel = jacobiModels[crop]
            ?: throw IllegalArgumentException(
                "Unknown crop: $crop"
            )

        val features = featureExtractor.extractFeatures(bitmap)

        val result = jacobiModel.predict(features)

        return TinyBayesPrediction(
            className = result.className,
            confidence = result.confidence,
            probabilities = result.probabilities
        )
    }

    fun close() {
        featureExtractor.close()
    }
}