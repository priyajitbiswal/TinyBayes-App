# TinyBayes: On-Device Crop Leaf Disease Classifier

**TinyBayes** is a lightweight, fully offline Android application for real-time agricultural crop disease diagnosis. It runs deep neural feature extraction combined with Jacobi-DMR regression entirely on-device, delivering instant diagnoses and calibrated class probabilities without requiring an internet connection or cloud backend.

---

## Features

- **100% Offline Edge Inference**: No cloud APIs, data transmission, or network required. All inference runs locally in milliseconds using ONNX Runtime.
- **5 Supported Crop Types (27 Diagnostic Classes)**:
  - **Potato** (3 classes: Early Blight, Late Blight, Healthy)
  - **Cotton** (7 classes: Bacterial Blight, Curl Virus, Fusarium Wilt, Healthy, Leaf Hopper / Jassids, Leaf Redding, Leaf Variegation)
  - **Rice** (6 classes: Bacterial Leaf Blight, Brown Spot, Healthy, Leaf Blast, Leaf Scald, Sheath Blight)
  - **Tomato** (8 classes: Bacterial Spot, Early Blight, Healthy, Late Blight, Leaf Mold, Mosaic Virus, Septoria Spot, Yellow Virus)
  - **Cocoa** (3 classes: Anthracnose, CSSVD, Healthy)
- **Two-Stage Intelligent Pipeline**:
  - **Stage 1 (Gatekeeper)**: Pre-prediction verification model (77.3% validation accuracy) checks whether the leaf matches the selected crop, preventing misdiagnosis on wrong inputs.
  - **Stage 2 (Disease Diagnosis)**: Closed-form Jacobi-DMR head diagnoses the exact condition with full multi-class calibrated probabilities in < 1 ms.
- **Embedded Agronomic Treatments**: Integrated treatment database renders cultural sanitation, biological antagonists, and chemical spray dosages directly on-device.
- **In-App Camera Capture & Gallery Selection**: Supports direct camera snapshotting in the field using Android's native `FileProvider` (`ActivityResultContracts.TakePicture()`) as well as photo selection from local device storage (`ActivityResultContracts.GetContent()`).
- **Robust Image Preprocessing**:
  - **EXIF Orientation Correction**: Automatically handles phone camera rotation metadata to ensure leaves are upright.
  - **Context-Preserving Scaling**: Full-image scaling to 224x224 (no center-cropping) preserves complete lesion outlines and natural background context.
  - **Memory-Safe Downsampling**: Safely decodes multi-megapixel phone images without out-of-memory crashes.
- **Native Material 3 UI**: Clean, responsive interface built with Jetpack Compose, supporting both Light and Dark modes.
- **Optimized APK Size**: Configured with ABI splits to produce a lean **~46 MB** APK for physical phones (down from ~150 MB).

---

## Validation Accuracy

Evaluated across the 16,438-image active dataset (curated from PlantDoc field photography and TinyBayes benchmarks) using a strict stratified 80/20 train/validation split:

| Crop | Diagnostic Classes | Validation Split | Jacobi-DMR Accuracy | On-Device Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Cotton** | 7 classes | 714 images | **94.82%** | < 1 ms |
| **Rice** | 6 classes | 766 images | **86.81%** | < 1 ms |
| **Potato** | 3 classes | 172 images | **85.47%** | < 1 ms |
| **Cocoa** | 3 classes | 1,105 images | **82.71%** | < 1 ms |
| **Tomato** | 8 classes | 533 images | **76.17%** | < 1 ms |
| **Overall** | **27 Classes** | **3,290 images** | **~85.20% (Average)** | **~30 ms End-to-End** |

---

## Technical Architecture

```
                                  [ Input Image ]
                                         │
                                         ▼
                     [ EXIF Rotation & Safe Downsampling ]
                                         │
                                         ▼
                         [ Full-Image Resize to 224x224 ]
                                         │
                                         ▼
                        [ ImageNet Normalization (sRGB) ]
                                         │
                                         ▼
                    [ MobileNetV3-Small ONNX Runtime ]
                               (576-dim Embeddings)
                                         │
                                         ▼
                   [ Stage 1: Crop Identifier Verification ]
                       (Validates Leaf Matches Selected Crop)
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
            [ Mismatch Detected ]                   [ Crop Verified ]
         Show Error & Block Flow                             │
                                                             ▼
                                          [ Stage 2: Jacobi-DMR Linear Inference ]
                                                   Scores = X · β_class
                                                             │
                                                             ▼
                                                [ Numerically Stable Softmax ]
                                                             │
                                                             ▼
                                            [ Prediction, Probabilities & Remedy ]
```

### 1. Feature Extractor
- **Model**: Pre-trained MobileNetV3-Small (`mobilenet_v3_small_features.onnx`).
- **Engine**: Microsoft ONNX Runtime for Android (`com.microsoft.onnxruntime:onnxruntime-android`).
- **Output**: 576-dimensional feature embedding vector.

### 2. Crop Verification & Disease Inference
- **Gatekeeper Crop Identifier**: Pre-prediction model (`crop_identifier_coefficients.json`) that verifies whether the photographed leaf matches the crop selected by the user.
- **Jacobi-DMR Classifier**: Closed-form Bayesian coefficient heads ($\beta \in \mathbb{R}^{576}$) loaded with integrated agronomic remedies from JSON assets.
- Computes dot-product logits for each class: $z_c = \mathbf{x}^T \beta_c$.
- Computes calibrated probabilities via Softmax:
  $$P(\text{class } c) = \frac{\exp(z_c - \max_j z_j)}{\sum_k \exp(z_k - \max_j z_j)}$$
- If the diagnosed leaf is diseased, the corresponding treatment protocol is displayed directly from the coefficients model.

---

## Project Structure

```
TinyBayes-App/
├── app/
│   ├── src/main/
│   │   ├── assets/models/
│   │   │   ├── mobilenet_v3_small_features.onnx       # Backbone ONNX graph
│   │   │   ├── mobilenet_v3_small_features.onnx.data  # Model weight tensors
│   │   │   ├── crop_identifier_coefficients.json      # Gatekeeper crop verifier
│   │   │   └── jacobi_coefficients.json               # Unified 5-crop coefficients & remedies
│   │   ├── java/com/example/tinybayes/
│   │   │   ├── MainActivity.kt                        # Jetpack Compose UI, camera & image loading
│   │   │   ├── model/
│   │   │   │   ├── OnnxFeatureExtractor.kt            # Image preprocessing & ONNX execution
│   │   │   │   └── JacobiModel.kt                     # Jacobi-DMR dot products, Softmax & Remedies
│   │   │   ├── inference/
│   │   │   │   └── TinyBayesClassifier.kt             # Crop verification router & coordinator
│   │   │   └── ui/theme/                              # Material 3 theme definitions
│   │   ├── res/xml/
│   │   │   └── file_paths.xml                         # FileProvider secure cache paths for camera
│   │   └── AndroidManifest.xml                        # App configuration & FileProvider declaration
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
