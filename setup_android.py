import os

files = {
    "android-app/settings.gradle.kts": """pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "OfflineClassifier"
include(":app")
""",

    "android-app/build.gradle.kts": """plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
}
""",

    "android-app/gradle.properties": """org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8
android.useAndroidX=true
kotlin.code.style=official
android.nonTransitiveRClass=true
""",

    "android-app/app/build.gradle.kts": """plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.classifier.offlineapp"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.classifier.offlineapp"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    aaptOptions {
        noCompress("onnx")
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.0")
}
""",

    "android-app/app/src/main/AndroidManifest.xml": """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <uses-feature android:name="android.hardware.camera" android:required="false" />

    <application
        android:allowBackup="true"
        android:label="Rice &amp; Flower Classifier"
        android:supportsRtl="true"
        android:theme="@style/Theme.MaterialComponents.DayNight.DarkActionBar">
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>

</manifest>
""",

    "android-app/app/src/main/res/layout/activity_main.xml": """<?xml version="1.0" encoding="utf-8"?>
<ScrollView xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:fillViewport="true">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical"
        android:padding="20dp"
        android:gravity="center_horizontal">

        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Select Classification Model"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <Spinner
            android:id="@+id/modelSpinner"
            android:layout_width="match_parent"
            android:layout_height="48dp"
            android:layout_marginBottom="16dp" />

        <ImageView
            android:id="@+id/imagePreview"
            android:layout_width="280dp"
            android:layout_height="280dp"
            android:scaleType="centerCrop"
            android:background="#E0E0E0"
            android:layout_marginBottom="16dp" />

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="12dp">

            <Button
                android:id="@+id/btnCamera"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="Take Photo"
                android:layout_marginEnd="8dp" />

            <Button
                android:id="@+id/btnGallery"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="Upload Gallery"
                android:layout_marginStart="8dp" />
        </LinearLayout>

        <Button
            android:id="@+id/btnPredict"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="Predict"
            android:layout_marginBottom="20dp" />

        <TextView
            android:id="@+id/txtResult"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="Select or capture an image to classify."
            android:textSize="18sp"
            android:textStyle="bold"
            android:gravity="center"
            android:padding="12dp" />

    </LinearLayout>
</ScrollView>
""",

    "android-app/app/src/main/java/com/classifier/offlineapp/MainActivity.kt": """package com.classifier.offlineapp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp

class MainActivity : AppCompatActivity() {

    private lateinit var modelSpinner: Spinner
    private lateinit var imagePreview: ImageView
    private lateinit var btnCamera: Button
    private lateinit var btnGallery: Button
    private lateinit var btnPredict: Button
    private lateinit var txtResult: TextView

    private var selectedBitmap: Bitmap? = null
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var riceSession: OrtSession? = null
    private var flowerSession: OrtSession? = null
    private var riceMapping: Map<Int, String> = emptyMap()
    private var flowerMapping: Map<Int, String> = emptyMap()

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap: Bitmap? ->
        if (bitmap != null) {
            selectedBitmap = bitmap
            imagePreview.setImageBitmap(bitmap)
            txtResult.text = "Image ready. Tap Predict."
        }
    }

    private val pickGalleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            val bitmap = uriToBitmap(uri)
            selectedBitmap = bitmap
            imagePreview.setImageBitmap(bitmap)
            txtResult.text = "Image ready. Tap Predict."
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modelSpinner = findViewById(R.id.modelSpinner)
        imagePreview = findViewById(R.id.imagePreview)
        btnCamera = findViewById(R.id.btnCamera)
        btnGallery = findViewById(R.id.btnGallery)
        btnPredict = findViewById(R.id.btnPredict)
        txtResult = findViewById(R.id.txtResult)

        val models = arrayOf("Rice Variety Classifier", "Flower Variety Classifier")
        modelSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            models
        )

        riceMapping = loadMapping("rice_mapping.json")
        flowerMapping = loadMapping("flower_mapping.json")

        btnCamera.setOnClickListener {
            takePictureLauncher.launch(null)
        }

        btnGallery.setOnClickListener {
            pickGalleryLauncher.launch("image/*")
        }

        btnPredict.setOnClickListener {
            val bitmap = selectedBitmap
            if (bitmap == null) {
                txtResult.text = "Please capture or upload an image first."
                return@setOnClickListener
            }
            runClassification(bitmap)
        }
    }

    private fun getSession(isRice: Boolean): OrtSession {
        return if (isRice) {
            if (riceSession == null) {
                val bytes = assets.open("rice_model.onnx").readBytes()
                riceSession = ortEnv.createSession(bytes)
            }
            riceSession!!
        } else {
            if (flowerSession == null) {
                val bytes = assets.open("flower_model.onnx").readBytes()
                flowerSession = ortEnv.createSession(bytes)
            }
            flowerSession!!
        }
    }

    private fun runClassification(bitmap: Bitmap) {
        val isRice = modelSpinner.selectedItemPosition == 0
        val targetSize = if (isRice) 200 else 224
        val mapping = if (isRice) riceMapping else flowerMapping
        val session = getSession(isRice)

        val resized = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
        val floatBuffer = preprocessImage(resized, targetSize)
        val inputName = session.inputNames.iterator().next()
        val shape = longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong())

        OnnxTensor.createTensor(ortEnv, floatBuffer, shape).use { inputTensor ->
            session.run(Collections.singletonMap(inputName, inputTensor)).use { results ->
                val outputArray = results[0].value as Array<FloatArray>
                val logits = outputArray[0]
                val probabilities = softmax(logits)

                var maxIdx = 0
                var maxProb = probabilities[0]
                for (i in 1 until probabilities.size) {
                    if (probabilities[i] > maxProb) {
                        maxProb = probabilities[i]
                        maxIdx = i
                    }
                }

                val predictedLabel = mapping[maxIdx] ?: "Class $maxIdx"
                val confidencePct = String.format("%.2f", maxProb * 100f)
                txtResult.text = "Prediction: $predictedLabel\\nConfidence: $confidencePct%"
            }
        }
    }

    private fun preprocessImage(bitmap: Bitmap, size: Int): FloatBuffer {
        val floatBuffer = FloatBuffer.allocate(3 * size * size)
        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val area = size * size

        for (i in 0 until area) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            floatBuffer.put(i, (r - mean[0]) / std[0])
            floatBuffer.put(area + i, (g - mean[1]) / std[1])
            floatBuffer.put(2 * area + i, (b - mean[2]) / std[2])
        }
        floatBuffer.rewind()
        return floatBuffer
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        var sum = 0.0f
        val exps = FloatArray(logits.size)
        for (i in logits.indices) {
            exps[i] = exp((logits[i] - maxLogit).toDouble()).toFloat()
            sum += exps[i]
        }
        for (i in exps.indices) {
            exps[i] /= sum
        }
        return exps
    }

    private fun loadMapping(fileName: String): Map<Int, String> {
        val jsonStr = assets.open(fileName).bufferedReader().use { it.readText() }
        val jsonObj = JSONObject(jsonStr)
        val map = mutableMapOf<Int, String>()
        val keys = jsonObj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val value = jsonObj.get(key)
            val keyInt = key.toIntOrNull()
            if (keyInt != null) {
                map[keyInt] = value.toString()
            } else {
                val valInt = value.toString().toIntOrNull()
                if (valInt != null) {
                    map[valInt] = key
                }
            }
        }
        return map
    }

    private fun uriToBitmap(uri: Uri): Bitmap {
        val rawBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(source)
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
        return rawBitmap.copy(Bitmap.Config.ARGB_8888, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        riceSession?.close()
        flowerSession?.close()
    }
}
""",

    ".github/workflows/build-apk.yml": """name: Build Offline Android APK

on:
  push:
    branches: [ "main", "master" ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install PyTorch and Export ONNX Models
        run: |
          pip install torch torchvision onnx
          python export_mobile.py

      - name: Set up JDK 17
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'

      - name: Setup Gradle
        uses: gradle/actions/setup-gradle@v3

      - name: Build Debug APK
        working-directory: ./android-app
        run: gradle assembleDebug

      - name: Upload Compiled APK
        uses: actions/upload-artifact@v4
        with:
          name: offline-classifier-apk
          path: android-app/app/build/outputs/apk/debug/app-debug.apk
"""
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# Ensure .onnx is in .gitignore so GitHub doesn't reject large files
gitignore_path = ".gitignore"
existing = open(gitignore_path, "r", encoding="utf-8").read() if os.path.exists(gitignore_path) else ""
if "*.onnx" not in existing:
    with open(gitignore_path, "a", encoding="utf-8") as f:
        f.write("\n*.onnx\n")

print("All Android project files written successfully!")