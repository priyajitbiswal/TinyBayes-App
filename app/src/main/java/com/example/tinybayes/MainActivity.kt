package com.example.tinybayes

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        setContent {

            val classifier = remember {
                DiseaseClassifier(this)
            }

            var bitmap by remember {
                mutableStateOf<Bitmap?>(null)
            }

            var prediction by remember {
                mutableStateOf("No diagnosis yet")
            }

            val galleryLauncher =
                rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.GetContent()
                ) { uri ->

                    if (uri != null) {

                        bitmap =
                            if (Build.VERSION.SDK_INT >= 28) {

                                val source =
                                    ImageDecoder.createSource(
                                        contentResolver,
                                        uri
                                    )

                                ImageDecoder.decodeBitmap(source)
                                    .copy(
                                        Bitmap.Config.ARGB_8888,
                                        true
                                    )

                            } else {

                                MediaStore.Images.Media.getBitmap(
                                    contentResolver,
                                    uri
                                ).copy(
                                    Bitmap.Config.ARGB_8888,
                                    true
                                )
                            }

                        prediction = "No diagnosis yet"
                    }
                }

            val cameraLauncher =
                rememberLauncherForActivityResult(
                    contract =
                        ActivityResultContracts.TakePicturePreview()
                ) { capturedBitmap ->

                    if (capturedBitmap != null) {

                        bitmap =
                            capturedBitmap.copy(
                                Bitmap.Config.ARGB_8888,
                                true
                            )

                        prediction = "No diagnosis yet"
                    }
                }

            Scaffold { padding ->

                Column(

                    modifier = Modifier
                        .fillMaxSize()
                        .padding(padding)
                        .padding(20.dp),

                    horizontalAlignment =
                        Alignment.CenterHorizontally,

                    verticalArrangement =
                        Arrangement.spacedBy(16.dp)

                ) {

                    Spacer(
                        modifier = Modifier.height(8.dp)
                    )

                    Text(
                        text = "Cocoa Disease Detection",
                        style =
                            MaterialTheme.typography.headlineMedium
                    )

                    Text(
                        text =
                            "Analyze cocoa leaf images directly on your device",
                        style =
                            MaterialTheme.typography.bodyMedium
                    )

                    Card(

                        modifier =
                            Modifier.fillMaxWidth(),

                        shape =
                            RoundedCornerShape(20.dp),

                        elevation =
                            CardDefaults.cardElevation(6.dp)

                    ) {

                        Column(

                            modifier =
                                Modifier.padding(12.dp),

                            horizontalAlignment =
                                Alignment.CenterHorizontally

                        ) {

                            if (bitmap != null) {

                                Image(

                                    bitmap =
                                        bitmap!!.asImageBitmap(),

                                    contentDescription =
                                        null,

                                    contentScale =
                                        ContentScale.Fit,

                                    modifier =
                                        Modifier
                                            .fillMaxWidth()
                                            .height(340.dp)
                                            .clip(
                                                RoundedCornerShape(
                                                    16.dp
                                                )
                                            )

                                )

                            } else {

                                Column(

                                    modifier =
                                        Modifier
                                            .fillMaxWidth()
                                            .height(340.dp),

                                    horizontalAlignment =
                                        Alignment.CenterHorizontally,

                                    verticalArrangement =
                                        Arrangement.Center

                                ) {

                                    Text(
                                        text =
                                            "No image selected"
                                    )
                                }
                            }
                        }
                    }

                    Row(

                        modifier =
                            Modifier.fillMaxWidth(),

                        horizontalArrangement =
                            Arrangement.spacedBy(12.dp)

                    ) {

                        Button(

                            modifier =
                                Modifier.weight(1f),

                            onClick = {
                                cameraLauncher.launch(null)
                            }

                        ) {

                            Text("Take Photo")
                        }

                        Button(

                            modifier =
                                Modifier.weight(1f),

                            onClick = {
                                galleryLauncher.launch("image/*")
                            }

                        ) {

                            Text("Select Image")
                        }
                    }

                    Button(

                        modifier =
                            Modifier.fillMaxWidth(),

                        enabled =
                            bitmap != null,

                        onClick = {

                            prediction =
                                try {

                                    classifier.predict(
                                        bitmap!!
                                    )

                                } catch (e: Exception) {

                                    e.message
                                        ?: "Prediction Error"
                                }
                        }

                    ) {

                        Text(
                            "Detect Disease"
                        )
                    }

                    if (prediction != "No diagnosis yet") {

                        Card(

                            modifier =
                                Modifier.fillMaxWidth(),

                            shape =
                                RoundedCornerShape(20.dp),

                            elevation =
                                CardDefaults.cardElevation(6.dp)

                        ) {

                            Column(

                                modifier =
                                    Modifier.padding(20.dp),

                                horizontalAlignment =
                                    Alignment.CenterHorizontally

                            ) {

                                Text(

                                    text =
                                        prediction,

                                    style =
                                        MaterialTheme
                                            .typography
                                            .displaySmall

                                )
                            }
                        }
                    }
                }
            }
        }
    }
}