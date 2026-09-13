package com.example.tinybayes

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ColorSpace
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import coil3.compose.AsyncImage
import com.example.tinybayes.inference.TinyBayesClassifier
import com.example.tinybayes.ui.theme.TinyBayesTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private var selectedImageUri by mutableStateOf<Uri?>(null)
    private var selectedCrop by mutableStateOf("Potato")

    private var prediction by mutableStateOf<String?>(null)
    private var confidence by mutableStateOf<Double?>(null)
    private var probabilities by mutableStateOf<Map<String, Double>>(emptyMap())
    private var isPredicting by mutableStateOf(false)
    private var errorMessage by mutableStateOf<String?>(null)

    private lateinit var classifier: TinyBayesClassifier

    private val galleryLauncher =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->
            selectedImageUri = uri
            prediction = null
            confidence = null
            probabilities = emptyMap()
            errorMessage = null

            if (uri != null) {
                runPrediction(uri, selectedCrop)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        classifier = TinyBayesClassifier(this)

        setContent {
            TinyBayesTheme {
                Scaffold { innerPadding ->
                    TinyBayesScreen(
                        modifier = Modifier.padding(innerPadding),
                        selectedImageUri = selectedImageUri,
                        selectedCrop = selectedCrop,
                        prediction = prediction,
                        confidence = confidence,
                        probabilities = probabilities,
                        isPredicting = isPredicting,
                        errorMessage = errorMessage,
                        onCropSelected = { crop ->
                            selectedCrop = crop
                            selectedImageUri?.let { uri ->
                                runPrediction(uri, crop)
                            }
                        },
                        onChooseImage = {
                            galleryLauncher.launch("image/*")
                        }
                    )
                }
            }
        }
    }

    private fun loadBitmapWithExif(uri: Uri): Bitmap {
        val orientation = contentResolver.openInputStream(uri)?.use { stream ->
            val exif = ExifInterface(stream)
            exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        } ?: ExifInterface.ORIENTATION_NORMAL

        val boundsOptions = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream, null, boundsOptions)
        }

        val rawWidth = boundsOptions.outWidth
        val rawHeight = boundsOptions.outHeight

        var inSampleSize = 1
        val targetMaxDim = 1024
        val maxDim = maxOf(rawWidth, rawHeight)
        while (maxDim / (inSampleSize * 2) >= targetMaxDim) {
            inSampleSize *= 2
        }

        val decodeOptions = BitmapFactory.Options().apply {
            this.inSampleSize = inSampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                inPreferredColorSpace = ColorSpace.get(ColorSpace.Named.SRGB)
            }
        }

        val decodedBitmap = contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream, null, decodeOptions)
        } ?: throw IllegalArgumentException("Unable to decode selected image")

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
        }

        return if (matrix.isIdentity) {
            decodedBitmap
        } else {
            val rotated = Bitmap.createBitmap(
                decodedBitmap,
                0,
                0,
                decodedBitmap.width,
                decodedBitmap.height,
                matrix,
                true
            )
            decodedBitmap.recycle()
            rotated
        }
    }

    private fun runPrediction(
        uri: Uri,
        crop: String
    ) {
        isPredicting = true
        prediction = null
        confidence = null
        probabilities = emptyMap()
        errorMessage = null

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val bitmap = loadBitmapWithExif(uri)

                val result = classifier.predict(
                    bitmap = bitmap,
                    crop = crop
                )

                withContext(Dispatchers.Main) {
                    prediction = result.className
                    confidence = result.confidence
                    probabilities = result.probabilities
                    isPredicting = false
                }

                bitmap.recycle()

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    errorMessage = e.message ?: "Prediction failed"
                    isPredicting = false
                }
            }
        }
    }

    override fun onDestroy() {
        classifier.close()
        super.onDestroy()
    }
}

@Composable
fun TinyBayesScreen(
    modifier: Modifier = Modifier,
    selectedImageUri: Uri?,
    selectedCrop: String,
    prediction: String?,
    confidence: Double?,
    probabilities: Map<String, Double>,
    isPredicting: Boolean,
    errorMessage: String?,
    onCropSelected: (String) -> Unit,
    onChooseImage: () -> Unit
) {
    val crops = listOf(
        "Potato",
        "Cotton",
        "Rice",
        "Tomato",
        "Cocoa"
    )

    val scrollState = rememberScrollState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(horizontal = 20.dp, vertical = 16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "TinyBayes",
            style = MaterialTheme.typography.headlineMedium.copy(
                fontWeight = FontWeight.Bold
            )
        )
        Text(
            text = "Offline Crop Disease Classifier",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(20.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Crop",
                style = MaterialTheme.typography.titleMedium.copy(
                    fontWeight = FontWeight.SemiBold
                )
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            crops.forEach { crop ->
                val isSelected = crop == selectedCrop
                FilterChip(
                    selected = isSelected,
                    onClick = { onCropSelected(crop) },
                    label = {
                        Text(crop)
                    }
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                if (selectedImageUri != null) {
                    AsyncImage(
                        model = selectedImageUri,
                        contentDescription = "Selected leaf image",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(240.dp)
                            .clip(RoundedCornerShape(12.dp)),
                        contentScale = ContentScale.Crop
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    OutlinedButton(
                        onClick = onChooseImage,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Change Image")
                    }
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(180.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(MaterialTheme.colorScheme.surface),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "Select a leaf image to evaluate",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    Button(
                        onClick = onChooseImage,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("Select Image")
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        if (isPredicting) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(20.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp))
                    Spacer(modifier = Modifier.width(16.dp))
                    Text(
                        text = "Classifying...",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        } else if (prediction != null && confidence != null) {
            val formattedDisease = prediction.replace('_', ' ')
            val confidencePct = (confidence * 100).coerceIn(0.0, 100.0)

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(20.dp)
                ) {
                    Text(
                        text = "Prediction",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = formattedDisease,
                        style = MaterialTheme.typography.headlineSmall.copy(
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    )
                    Spacer(modifier = Modifier.height(12.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Confidence",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = "%.1f%%".format(confidencePct),
                            style = MaterialTheme.typography.titleMedium.copy(
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            )
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    LinearProgressIndicator(
                        progress = { (confidencePct / 100f).toFloat() },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(6.dp)
                            .clip(RoundedCornerShape(3.dp))
                    )

                    if (probabilities.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "Class Probabilities",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(modifier = Modifier.height(8.dp))

                        val sortedProbabilities = probabilities.entries
                            .sortedByDescending { it.value }

                        sortedProbabilities.forEach { (name, prob) ->
                            val probPct = (prob * 100).coerceIn(0.0, 100.0)
                            val cleanName = name.replace('_', ' ')
                            val isTop = (name == prediction)
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 3.dp),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = cleanName,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (isTop) {
                                        MaterialTheme.colorScheme.primary
                                    } else {
                                        MaterialTheme.colorScheme.onSurfaceVariant
                                    },
                                    fontWeight = if (isTop) FontWeight.Bold else FontWeight.Normal
                                )
                                Text(
                                    text = "%.1f%%".format(probPct),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (isTop) {
                                        MaterialTheme.colorScheme.primary
                                    } else {
                                        MaterialTheme.colorScheme.onSurfaceVariant
                                    },
                                    fontWeight = if (isTop) FontWeight.Bold else FontWeight.Normal
                                )
                            }
                        }
                    }
                }
            }

        } else if (errorMessage != null) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    Text(
                        text = "Error Occurred",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = errorMessage,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onErrorContainer
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
    }
}

@Preview(showBackground = true)
@Composable
fun TinyBayesPreview() {
    TinyBayesTheme {
        TinyBayesScreen(
            selectedImageUri = null,
            selectedCrop = "Potato",
            prediction = "Late_Blight",
            confidence = 0.986,
            probabilities = mapOf(
                "Late_Blight" to 0.986,
                "Early_Blight" to 0.012,
                "Healthy" to 0.002
            ),
            isPredicting = false,
            errorMessage = null,
            onCropSelected = {},
            onChooseImage = {}
        )
    }
}