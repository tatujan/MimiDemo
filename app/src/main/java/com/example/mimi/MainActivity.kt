package com.example.mimi

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

enum class ModelVariant(val label: String, val filename: String) {
    FP32("FP32", "mimi_model.safetensors"),
    Q8("Q8", "mimi_q8.gguf"),
    Q4("Q4", "mimi_q4_0.gguf");
}

class MainActivity : ComponentActivity() {

    companion object {
        private const val SAMPLE_RATE = 24000
        private const val FRAME_SIZE = 1920 // 80ms at 24kHz
        private const val MAX_RECORD_SECONDS = 60
        private const val MODEL_URL =
            "https://huggingface.co/kyutai/moshiko-pytorch-bf16/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors"
    }

    // UI state
    private var modelStatus = mutableStateOf("Checking models...")
    private var modelReady = mutableStateOf(false)
    private var selectedModel = mutableStateOf(ModelVariant.Q4)
    private var availableModels = mutableStateOf(emptySet<ModelVariant>())
    private var selectedCodebooks = mutableIntStateOf(8)
    private var isRecording = mutableStateOf(false)
    private var isProcessing = mutableStateOf(false)
    private var recordingInfo = mutableStateOf("")
    private var encodeStats = mutableStateOf("")
    private var decodeStats = mutableStateOf("")

    // Data (not Compose state — too large)
    private var recordedPcm: FloatArray? = null
    private var encodedTokens: List<IntArray>? = null
    private var encodedCodebooks: Int = 0

    // Recording thread handle
    @Volatile
    private var recording = false

    private val codebookOptions = listOf(1, 2, 4, 8, 16)

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) toggleRecording()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MimiDemoScreen()
                }
            }
        }

        detectAndPrepareModels()
    }

    @Composable
    private fun MimiDemoScreen() {
        val scrollState = rememberScrollState()
        val status by modelStatus
        val ready by modelReady
        val model by selectedModel
        val available by availableModels
        val codebooks by selectedCodebooks
        val recording by isRecording
        val processing by isProcessing
        val recInfo by recordingInfo
        val encStats by encodeStats
        val decStats by decodeStats

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(scrollState)
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                "Mimi Codec Demo",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold
            )

            Text(status, color = if (ready) MaterialTheme.colorScheme.primary
                                 else MaterialTheme.colorScheme.onSurfaceVariant)

            HorizontalDivider()

            // Model selector
            Text("Model:", fontWeight = FontWeight.Medium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ModelVariant.entries.forEach { variant ->
                    val isAvailable = variant in available
                    FilterChip(
                        selected = model == variant,
                        onClick = {
                            selectedModel.value = variant
                            encodedTokens = null
                            encodeStats.value = ""
                            decodeStats.value = ""
                        },
                        label = { Text(variant.label) },
                        enabled = isAvailable && !processing && !recording
                    )
                }
            }

            HorizontalDivider()

            // Codebook selector
            Text("Codebooks:", fontWeight = FontWeight.Medium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                codebookOptions.forEach { n ->
                    FilterChip(
                        selected = codebooks == n,
                        onClick = {
                            selectedCodebooks.intValue = n
                            encodedTokens = null
                            encodeStats.value = ""
                            decodeStats.value = ""
                        },
                        label = { Text("$n") },
                        enabled = !processing && !recording
                    )
                }
            }

            HorizontalDivider()

            // Record
            Button(
                onClick = { onRecordPressed() },
                enabled = ready && !processing,
                colors = if (recording) ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error
                ) else ButtonDefaults.buttonColors(),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (recording) "Stop Recording" else "Record")
            }

            if (recInfo.isNotEmpty()) {
                Text(recInfo, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
            }

            HorizontalDivider()

            // Encode
            Button(
                onClick = { encode() },
                enabled = ready && recordedPcm != null && !processing && !recording,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Encode")
            }

            if (encStats.isNotEmpty()) {
                Text(encStats, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
            }

            HorizontalDivider()

            // Decode & Play
            Button(
                onClick = { decodeAndPlay() },
                enabled = ready && encodedTokens != null && !processing && !recording,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Decode & Play")
            }

            if (decStats.isNotEmpty()) {
                Text(decStats, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
            }

            Spacer(modifier = Modifier.height(32.dp))
        }
    }

    // --- Model Detection & Download ---

    private fun detectAndPrepareModels() {
        lifecycleScope.launch(Dispatchers.IO) {
            val found = mutableSetOf<ModelVariant>()

            for (variant in ModelVariant.entries) {
                val localFile = File(filesDir, variant.filename)
                if (localFile.exists()) {
                    found.add(variant)
                    continue
                }
                // Check adb push location
                val adbFile = File("/data/local/tmp/${variant.filename}")
                if (adbFile.exists()) {
                    withContext(Dispatchers.Main) {
                        modelStatus.value = "Copying ${variant.label} from adb..."
                    }
                    try {
                        adbFile.copyTo(localFile, overwrite = true)
                        found.add(variant)
                    } catch (_: Exception) { }
                }
            }

            withContext(Dispatchers.Main) {
                availableModels.value = found
                if (found.isNotEmpty()) {
                    // Auto-select best available: Q4 > Q8 > FP32
                    val best = when {
                        ModelVariant.Q4 in found -> ModelVariant.Q4
                        ModelVariant.Q8 in found -> ModelVariant.Q8
                        else -> ModelVariant.FP32
                    }
                    selectedModel.value = best
                    modelStatus.value = "Ready: ${found.joinToString(", ") { it.label }}"
                    modelReady.value = true
                } else {
                    modelStatus.value = "No models found. Downloading FP32..."
                    downloadFp32Model()
                }
            }
        }
    }

    private fun downloadFp32Model() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val targetFile = File(filesDir, ModelVariant.FP32.filename)
                val tmpFile = File(filesDir, "${ModelVariant.FP32.filename}.tmp")
                val url = URL(MODEL_URL)
                val conn = url.openConnection() as HttpURLConnection
                conn.connectTimeout = 15000
                conn.readTimeout = 30000
                conn.connect()

                val totalBytes = conn.contentLength.toLong()
                var downloadedBytes = 0L

                conn.inputStream.use { input ->
                    tmpFile.outputStream().use { output ->
                        val buffer = ByteArray(65536)
                        while (true) {
                            val read = input.read(buffer)
                            if (read == -1) break
                            output.write(buffer, 0, read)
                            downloadedBytes += read

                            if (totalBytes > 0) {
                                val pct = (downloadedBytes * 100 / totalBytes).toInt()
                                withContext(Dispatchers.Main) {
                                    modelStatus.value = "Downloading FP32... $pct%"
                                }
                            }
                        }
                    }
                }

                tmpFile.renameTo(targetFile)

                withContext(Dispatchers.Main) {
                    availableModels.value = availableModels.value + ModelVariant.FP32
                    selectedModel.value = ModelVariant.FP32
                    modelStatus.value = "Ready: FP32"
                    modelReady.value = true
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    modelStatus.value = "Download failed: ${e.message}"
                }
            }
        }
    }

    private fun resolveModelPath(): String {
        return File(filesDir, selectedModel.value.filename).absolutePath
    }

    // --- Recording ---

    private fun onRecordPressed() {
        if (isRecording.value) {
            recording = false
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            return
        }

        toggleRecording()
    }

    private fun toggleRecording() {
        if (isRecording.value) {
            recording = false
            return
        }

        isRecording.value = true
        recording = true
        recordedPcm = null
        encodedTokens = null
        encodeStats.value = ""
        decodeStats.value = ""
        recordingInfo.value = "Recording..."

        Thread {
            val bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT
            )
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
                maxOf(bufferSize, FRAME_SIZE * 4 * 2)
            )

            val maxSamples = SAMPLE_RATE * MAX_RECORD_SECONDS
            val buffer = FloatArray(maxSamples)
            var writePos = 0
            val readBuf = FloatArray(FRAME_SIZE)

            recorder.startRecording()

            while (recording && writePos + FRAME_SIZE <= maxSamples) {
                val read = recorder.read(readBuf, 0, FRAME_SIZE, AudioRecord.READ_BLOCKING)
                if (read > 0 && writePos + read <= maxSamples) {
                    System.arraycopy(readBuf, 0, buffer, writePos, read)
                    writePos += read

                    val secs = writePos.toFloat() / SAMPLE_RATE
                    runOnUiThread {
                        recordingInfo.value = "Recording... %.1fs".format(secs)
                    }
                }
            }

            recorder.stop()
            recorder.release()

            val pcm = buffer.copyOf(writePos)
            recordedPcm = pcm
            val durationSecs = pcm.size.toFloat() / SAMPLE_RATE
            val sizeKb = pcm.size * 4f / 1024f

            runOnUiThread {
                isRecording.value = false
                recordingInfo.value = "Recorded: %.1fs  |  %.0f KB PCM".format(durationSecs, sizeKb)
            }
        }.start()
    }

    // --- Encode ---

    private fun encode() {
        val pcm = recordedPcm ?: return
        isProcessing.value = true
        encodeStats.value = "Encoding..."
        decodeStats.value = ""
        encodedTokens = null

        val numCb = selectedCodebooks.intValue
        val modelPath = resolveModelPath()
        val modelLabel = selectedModel.value.label

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val codec = MimiCodec.create(modelPath, numCb)
                codec.reset()

                // Pad to multiple of FRAME_SIZE
                val padded = if (pcm.size % FRAME_SIZE == 0) pcm
                else {
                    val newSize = ((pcm.size / FRAME_SIZE) + 1) * FRAME_SIZE
                    pcm.copyOf(newSize) // pads with zeros
                }

                val numFrames = padded.size / FRAME_SIZE
                val tokens = mutableListOf<IntArray>()

                val t0 = System.nanoTime()
                for (i in 0 until numFrames) {
                    val chunk = padded.copyOfRange(i * FRAME_SIZE, (i + 1) * FRAME_SIZE)
                    val result = codec.encodeStep(chunk)
                    if (result != null) tokens.add(result)
                }
                val encodeMs = (System.nanoTime() - t0) / 1_000_000.0

                codec.destroy()

                val durationSecs = pcm.size.toDouble() / SAMPLE_RATE
                val originalBytes = pcm.size * 4
                val tokenBytes = tokens.sumOf { it.size } * 4
                val ratio = if (tokenBytes > 0) originalBytes.toDouble() / tokenBytes else 0.0
                val realtimeFactor = if (encodeMs > 0) durationSecs / (encodeMs / 1000.0) else 0.0

                encodedTokens = tokens
                encodedCodebooks = numCb

                withContext(Dispatchers.Main) {
                    encodeStats.value = buildString {
                        appendLine("Model:        $modelLabel")
                        appendLine("Codebooks:    $numCb")
                        appendLine("Duration:     %.2fs".format(durationSecs))
                        appendLine("PCM size:     %.1f KB".format(originalBytes / 1024.0))
                        appendLine("Token size:   %.1f KB".format(tokenBytes / 1024.0))
                        appendLine("Compression:  %.0fx".format(ratio))
                        appendLine("Encode time:  %.0f ms".format(encodeMs))
                        append("Realtime:     %.1fx".format(realtimeFactor))
                    }
                    isProcessing.value = false
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    encodeStats.value = "Encode failed: ${e.message}"
                    isProcessing.value = false
                }
            }
        }
    }

    // --- Decode & Play ---

    private fun decodeAndPlay() {
        val tokens = encodedTokens ?: return
        isProcessing.value = true
        decodeStats.value = "Decoding..."

        val numCb = encodedCodebooks
        val modelPath = resolveModelPath()
        val modelLabel = selectedModel.value.label

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val codec = MimiCodec.create(modelPath, numCb)
                codec.reset()

                val pcmChunks = mutableListOf<FloatArray>()

                val t0 = System.nanoTime()
                for (tokenFrame in tokens) {
                    val result = codec.decodeStep(tokenFrame)
                    if (result != null) pcmChunks.add(result)
                }
                val decodeMs = (System.nanoTime() - t0) / 1_000_000.0

                codec.destroy()

                // Concatenate PCM
                val totalSamples = pcmChunks.sumOf { it.size }
                val fullPcm = FloatArray(totalSamples)
                var offset = 0
                for (chunk in pcmChunks) {
                    System.arraycopy(chunk, 0, fullPcm, offset, chunk.size)
                    offset += chunk.size
                }

                val durationSecs = totalSamples.toDouble() / SAMPLE_RATE
                val realtimeFactor = if (decodeMs > 0) durationSecs / (decodeMs / 1000.0) else 0.0

                withContext(Dispatchers.Main) {
                    decodeStats.value = buildString {
                        appendLine("Model:        $modelLabel")
                        appendLine("Decode time:  %.0f ms".format(decodeMs))
                        appendLine("Realtime:     %.1fx".format(realtimeFactor))
                        append("Playing...")
                    }
                }

                // Play audio
                val track = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setSampleRate(SAMPLE_RATE)
                            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build()
                    )
                    .setBufferSizeInBytes(FRAME_SIZE * 4 * 4)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()

                track.play()
                track.write(fullPcm, 0, fullPcm.size, AudioTrack.WRITE_BLOCKING)
                track.stop()
                track.release()

                withContext(Dispatchers.Main) {
                    decodeStats.value = buildString {
                        appendLine("Model:        $modelLabel")
                        appendLine("Decode time:  %.0f ms".format(decodeMs))
                        appendLine("Realtime:     %.1fx".format(realtimeFactor))
                        append("Playback complete")
                    }
                    isProcessing.value = false
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    decodeStats.value = "Decode failed: ${e.message}"
                    isProcessing.value = false
                }
            }
        }
    }
}
