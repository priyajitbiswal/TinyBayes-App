package com.example.tinybayes

data class JacobiCoefficients(
    val healthy: List<Double>,
    val cssvd: List<Double>,
    val anthracnose: List<Double>
)