# TinyBayes: On-Device Crop Leaf Disease Classifier

**TinyBayes** is a lightweight, fully offline Android application for real-time agricultural crop disease diagnosis. It runs deep neural feature extraction combined with Jacobi-DMR regression entirely on-device, delivering instant diagnoses and calibrated class probabilities without requiring an internet connection or cloud backend.

---

## Features

- **100% Offline Edge Inference**: No cloud APIs, data transmission, or network required. All inference runs locally in milliseconds using ONNX Runtime.
- **5 Supported Crop Types**:
  - **Potato** (Early Blight, Late Blight, Healthy)
  - **Cotton** (Bacterial Blight, Curl Virus, Fusarium Wilt, Healthy)
  - **Rice** (Brown Spot, Healthy, Hispa, Leaf Blast)
  - **Tomato** (10 conditions including Blights, Molds, Viruses, and Healthy)
  - **Cocoa** (Anthracnose, CSSVD, Healthy)
- **Mathematical Parity with Research Notebooks**: Achieves **100.0% prediction agreement** against the original Jupyter research pipeline across 4,384 benchmark validation images with zero discrepancies.
- **Calibrated Softmax Probabilities**: Displays true confidence percentages (`0.0%` to `100.0%`) alongside full multi-class probability rankings.
- **Robust Image Preprocessing**:
  - **EXIF Orientation Correction**: Automatically handles phone camera rotation metadata to ensure leaves are upright.
  - **Aspect-Ratio Preservation**: Center-crops rectangular photos to square frames to prevent squashing or distorting lesion geometry.
  - **Memory-Safe Downsampling**: Safely decodes multi-megapixel phone images without out-of-memory crashes.
- **Native Material 3 UI**: Clean, responsive interface built with Jetpack Compose, supporting both Light and Dark modes.
- **Optimized APK Size**: Configured with ABI splits to produce a lean **~46 MB** APK for physical phones (down from ~150 MB).

---

## Validation Accuracy

Validated image-by-image on disk comparing the Android inference pipeline against the Jupyter research notebooks:

| Crop | Validation Dataset Size | Notebook Accuracy | Android App Accuracy | Prediction Agreement |
| :--- | :---: | :---: | :---: | :---: |
| **Potato** | 431 images | **98.61%** | **98.61%** | **100.0%** (0 discrepancies) |
| **Cotton** | 428 images | **94.39%** | **94.39%** | **100.0%** (0 discrepancies) |
| **Rice** | 420 images | **94.05%** | **94.05%** | **100.0%** (0 discrepancies) |
| **Tomato** | 2,000 images | **94.50%** | **94.50%** | **100.0%** (0 discrepancies) |
| **Cocoa** | 1,105 images | **81.27%** | **81.27%** | **100.0%** (0 discrepancies) |
| **Total** | **4,384 images** | **91.51%** | **91.51%** | **100.0%** (0 discrepancies) |

---

## Technical Architecture

```
                                  [ Input Image ]
                                         │
                                         ▼
                     [ EXIF Rotation & Safe Downsampling ]
                                         │
                                         ▼
                        [ Center-Square Crop & 224x224 ]
                                         │
                                         ▼
                        [ ImageNet Normalization (sRGB) ]
                                         │
                                         ▼
                    [ MobileNetV3-Small ONNX Runtime ]
                               (576-dim Embeddings)
                                         │
                                         ▼
                     [ Jacobi-DMR Linear Inference (Offline) ]
                             Scores = X · β_class
                                         │
                                         ▼
                       [ Numerically Stable Softmax ]
                                         │
                                         ▼
                   [ Predicted Class & Probability Distribution ]
```

### 1. Feature Extractor
- **Model**: Pre-trained MobileNetV3-Small (`mobilenet_v3_small_features.onnx`).
- **Engine**: Microsoft ONNX Runtime for Android (`com.microsoft.onnxruntime:onnxruntime-android`).
- **Output**: 576-dimensional feature embedding vector.

### 2. Jacobi-DMR Classifier
- Pre-computed coefficients ($\beta \in \mathbb{R}^{576}$) loaded from JSON assets per crop.
- Computes dot-product logits for each class: $z_c = \mathbf{x}^T \beta_c$.
- Computes calibrated probabilities via Softmax:
  $$P(\text{class } c) = \frac{\exp(z_c - \max_j z_j)}{\sum_k \exp(z_k - \max_j z_j)}$$

---

## Project Structure

```
AndroidApps/
├── app/
│   ├── src/main/
│   │   ├── assets/models/
│   │   │   ├── mobilenet_v3_small_features.onnx       # Backbone ONNX graph
│   │   │   ├── mobilenet_v3_small_features.onnx.data  # Model weight tensors
│   │   │   ├── potato/jacobi_coefficients.json        # Potato coefficients
│   │   │   ├── cotton/jacobi_coefficients.json        # Cotton coefficients
│   │   │   ├── rice/jacobi_coefficients.json          # Rice coefficients
│   │   │   ├── tomato/jacobi_coefficients.json        # Tomato coefficients
│   │   │   └── cocoa/jacobi_coefficients.json         # Cocoa coefficients
│   │   ├── java/com/example/tinybayes/
│   │   │   ├── MainActivity.kt                        # Jetpack Compose UI & image loading
│   │   │   ├── model/
│   │   │   │   ├── OnnxFeatureExtractor.kt            # Image preprocessing & ONNX execution
│   │   │   │   └── JacobiModel.kt                     # Jacobi-DMR dot products & Softmax
│   │   │   ├── inference/
│   │   │   │   └── TinyBayesClassifier.kt             # Crop model router & coordinator
│   │   │   └── ui/theme/                              # Material 3 theme definitions
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts                               # Dependencies & ABI splits configuration
├── gradle/                                            # Gradle wrapper & version catalogs
├── gradlew.bat / gradlew                              # Gradle build scripts
└── README.md
```

---

## Building the Project

### Prerequisites
- **JDK**: Java 17 or Java 21 (e.g. Android Studio JBR).
- **Android SDK**: `minSdk = 26` (Android 8.0 Oreo), `targetSdk = 37`.
- **Android Studio**: Ladybug / Meerkat or newer recommended.

### Building via Command Line
Set `JAVA_HOME` to your JDK / Android Studio JBR path and run:

```bash
# Windows (cmd.exe)
set JAVA_HOME=C:\Program Files\Android\Android Studio\jbr
gradlew.bat assembleDebug

# Windows (PowerShell)
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
.\gradlew.bat assembleDebug

# macOS / Linux
export JAVA_HOME=/Applications/Android\ Studio.app/Contents/jbr/Contents/Home
./gradlew assembleDebug
```

---

## Choosing the Right APK to Install

The build outputs several architecture-specific APKs inside `app/build/outputs/apk/debug/`:

| APK File | Target Device | Size | Note |
| :--- | :--- | :---: | :--- |
| **`app-arm64-v8a-debug.apk`** | **Modern Android Phones** | **~46 MB** | **Recommended for all modern phones** (Samsung, Pixel, Xiaomi, OnePlus, etc.) |
| `app-armeabi-v7a-debug.apk` | Legacy 32-bit Phones | ~37 MB | For older 32-bit devices |
| `app-x86_64-debug.apk` | PC Emulators | ~52 MB | For Android Studio emulator testing |
| `app-universal-debug.apk` | All Devices (Fat APK) | ~142 MB | Contains all architectures bundled together |

### Installing to a Connected Phone via ADB
```bash
adb install -r app/build/outputs/apk/debug/app-arm64-v8a-debug.apk
```

---

## Photography Tips for Best Accuracy

The classification model is trained on close-up leaf disease specimens:
1. **Select the Correct Crop First**: Tap the corresponding crop chip (`Potato`, `Cotton`, `Rice`, `Tomato`, or `Cocoa`) before analyzing.
2. **Fill the Frame**: Frame the leaf so that it occupies 80–90% of the viewfinder. Avoid heavy backgrounds with soil, fingers, or clutter.
3. **Focus on Symptoms**: Position the distinct lesions, spots, or discoloration clearly in focus.
4. **Even Lighting**: Avoid harsh glare, deep shadows, or dark indoor illumination.
