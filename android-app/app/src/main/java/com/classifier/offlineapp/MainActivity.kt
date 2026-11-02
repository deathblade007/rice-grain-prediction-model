package com.classifier.offlineapp

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
                txtResult.text = "Prediction: $predictedLabel\nConfidence: $confidencePct%"
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
