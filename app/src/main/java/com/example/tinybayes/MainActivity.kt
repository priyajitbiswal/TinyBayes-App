package com.example.tinybayes

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
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
    private var selectedCrop by mutableStateOf("Cocoa")

    private var prediction by mutableStateOf<String?>(null)
    private var score by mutableStateOf<Double?>(null)
    private var isPredicting by mutableStateOf(false)
    private var errorMessage by mutableStateOf<String?>(null)

    private lateinit var classifier: TinyBayesClassifier

    private val galleryLauncher =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->

            selectedImageUri = uri
            prediction = null
            score = null
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
                TinyBayesScreen(
                    selectedImageUri = selectedImageUri,
                    selectedCrop = selectedCrop,
                    prediction = prediction,
                    score = score,
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

    private fun runPrediction(
        uri: Uri,
        crop: String
    ) {

        isPredicting = true
        prediction = null
        score = null
        errorMessage = null

        lifecycleScope.launch(Dispatchers.Default) {

            try {

                val bitmap = contentResolver
                    .openInputStream(uri)
                    .use { inputStream ->
                        BitmapFactory.decodeStream(inputStream)
                    }

                if (bitmap == null) {
                    throw IllegalArgumentException(
                        "Unable to decode selected image"
                    )
                }

                val result = classifier.predict(
                    bitmap = bitmap,
                    crop = crop
                )

                withContext(Dispatchers.Main) {
                    prediction = result.className
                    score = result.score
                    isPredicting = false
                }

                bitmap.recycle()

            } catch (e: Exception) {

                withContext(Dispatchers.Main) {
                    errorMessage =
                        e.message ?: "Prediction failed"
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
    selectedImageUri: Uri?,
    selectedCrop: String,
    prediction: String?,
    score: Double?,
    isPredicting: Boolean,
    errorMessage: String?,
    onCropSelected: (String) -> Unit,
    onChooseImage: () -> Unit
) {

    val crops = listOf(
        "Cocoa",
        "Potato",
        "Tomato",
        "Rice",
        "Cotton"
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Top
    ) {

        Spacer(modifier = Modifier.height(32.dp))

        Text(
            text = "TinyBayes",
            fontSize = 32.sp
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "Offline Crop Disease Classification"
        )

        Spacer(modifier = Modifier.height(32.dp))

        Text(
            text = "Select Crop",
            style = MaterialTheme.typography.titleMedium
        )

        Spacer(modifier = Modifier.height(12.dp))

        crops.forEach { crop ->

            if (crop == selectedCrop) {

                Button(
                    onClick = {
                        onCropSelected(crop)
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(crop)
                }

            } else {

                OutlinedButton(
                    onClick = {
                        onCropSelected(crop)
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(crop)
                }
            }

            Spacer(modifier = Modifier.height(6.dp))
        }

        Spacer(modifier = Modifier.height(18.dp))

        Button(
            onClick = onChooseImage,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Choose Image")
        }

        Spacer(modifier = Modifier.height(20.dp))

        if (selectedImageUri != null) {

            AsyncImage(
                model = selectedImageUri,
                contentDescription = "Selected leaf image",
                modifier = Modifier.size(250.dp),
                contentScale = ContentScale.Fit
            )
        }

        Spacer(modifier = Modifier.height(20.dp))

        if (isPredicting) {

            Text(
                text = "Classifying...",
                style = MaterialTheme.typography.titleMedium
            )

        } else if (prediction != null) {

            Text(
                text = "Prediction: $prediction",
                style = MaterialTheme.typography.headlineSmall
            )

            if (score != null) {
                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Score: %.4f".format(score),
                    style = MaterialTheme.typography.bodyLarge
                )
            }

        } else if (errorMessage != null) {

            Text(
                text = "Error: $errorMessage",
                style = MaterialTheme.typography.bodyLarge
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun TinyBayesPreview() {
    TinyBayesTheme {
        TinyBayesScreen(
            selectedImageUri = null,
            selectedCrop = "Cocoa",
            prediction = null,
            score = null,
            isPredicting = false,
            errorMessage = null,
            onCropSelected = {},
            onChooseImage = {}
        )
    }
}