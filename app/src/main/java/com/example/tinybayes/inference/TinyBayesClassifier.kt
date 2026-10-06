package com.example.tinybayes.inference

import android.content.Context
import android.graphics.Bitmap
import com.example.tinybayes.model.JacobiModel
import com.example.tinybayes.model.OnnxFeatureExtractor

class CropMismatchException(message: String) : Exception(message)

data class TinyBayesPrediction(
    val className: String,
    val confidence: Double,
    val probabilities: Map<String, Double>,
    val remedy: String? = null
)

class TinyBayesClassifier(context: Context) {

    private val featureExtractor = OnnxFeatureExtractor(context)

    private val cropIdentifierModel = JacobiModel(
        context,
        "models/crop_identifier_coefficients.json"
    )

    private val jacobiModels = mapOf(
        "Cocoa" to JacobiModel(
            context,
            "models/jacobi_coefficients.json",
            "cocoa"
        ),
        "Potato" to JacobiModel(
            context,
            "models/jacobi_coefficients.json",
            "potato"
        ),
        "Tomato" to JacobiModel(
            context,
            "models/jacobi_coefficients.json",
            "tomato"
        ),
        "Rice" to JacobiModel(
            context,
            "models/jacobi_coefficients.json",
            "rice"
        ),
        "Cotton" to JacobiModel(
            context,
            "models/jacobi_coefficients.json",
            "cotton"
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

        // Pre-prediction check: verify leaf belongs to the selected crop
        val cropCheck = cropIdentifierModel.predict(features)
        if (!cropCheck.className.equals(crop, ignoreCase = true)) {
            throw CropMismatchException(
                "This image does not appear to belong to a $crop leaf. Kindly select the correct leaf image."
            )
        }

        val result = jacobiModel.predict(features)

        return TinyBayesPrediction(
            className = result.className,
            confidence = result.confidence,
            probabilities = result.probabilities,
            remedy = result.remedy
        )
    }

    fun close() {
        featureExtractor.close()
    }
}