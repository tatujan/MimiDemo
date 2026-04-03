package com.example.mimi

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
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
        private const val MAX_RECORD_SECONDS = 60
        private const val MIN_PLAYBACK_SECONDS = 1.0f
        private const val MODEL_URL =
            "https://huggingface.co/kyutai/moshiko-pytorch-bf16/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors"
    }

    // UI state
    private var modelStatus = mutableStateOf("Checking models...")
    private var modelReady = mutableStateOf(false)
    private var selectedModel = mutableStateOf(ModelVariant.Q4)
    private var availableModels = mutableStateOf(emptySet<ModelVariant>())
    private var selectedCodebooks = mutableIntStateOf(8)
    private var selectedChunkMs = mutableIntStateOf(320)
    private var isPttActive = mutableStateOf(false)
    private var isRecording = mutableStateOf(false)
    private var isPlaying = mutableStateOf(false)
    private var pipelineStatus = mutableStateOf("")
    private var encodeStats = mutableStateOf("")
    private var decodeStats = mutableStateOf("")

    // Pipeline control
    @Volatile
    private var recording = false

    private val codebookOptions = listOf(1, 2, 4, 8, 16)
    private val chunkMsOptions = listOf(160, 240, 320, 400, 480, 560)

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startPttPipeline()
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
        val chunkMs by selectedChunkMs
        val pttActive by isPttActive
        val recActive by isRecording
        val playing by isPlaying
        val pStatus by pipelineStatus
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
                        onClick = { selectedModel.value = variant },
                        label = { Text(variant.label) },
                        enabled = isAvailable && !pttActive
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
                        onClick = { selectedCodebooks.intValue = n },
                        label = { Text("$n") },
                        enabled = !pttActive
                    )
                }
            }

            HorizontalDivider()

            // Chunk size selector (dropdown)
            Text("Chunk size (latency):", fontWeight = FontWeight.Medium)
            var chunkDropdownExpanded by remember { mutableStateOf(false) }
            @OptIn(ExperimentalMaterial3Api::class)
            ExposedDropdownMenuBox(
                expanded = chunkDropdownExpanded,
                onExpandedChange = { if (!pttActive) chunkDropdownExpanded = it }
            ) {
                OutlinedTextField(
                    value = "${chunkMs}ms",
                    onValueChange = {},
                    readOnly = true,
                    enabled = !pttActive,
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = chunkDropdownExpanded) },
                    modifier = Modifier.menuAnchor().fillMaxWidth(),
                    singleLine = true
                )
                ExposedDropdownMenu(
                    expanded = chunkDropdownExpanded,
                    onDismissRequest = { chunkDropdownExpanded = false }
                ) {
                    chunkMsOptions.forEach { ms ->
                        DropdownMenuItem(
                            text = { Text("${ms}ms") },
                            onClick = {
                                selectedChunkMs.intValue = ms
                                chunkDropdownExpanded = false
                            }
                        )
                    }
                }
            }

            HorizontalDivider()

            // Push-to-talk button
            Button(
                onClick = { /* handled by pointerInput */ },
                enabled = ready && !playing,
                colors = if (recActive) ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error
                ) else ButtonDefaults.buttonColors(),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(64.dp)
                    .pointerInput(ready, playing) {
                        if (!ready || playing) return@pointerInput
                        awaitEachGesture {
                            awaitFirstDown(requireUnconsumed = false)
                            onPttPressed()
                            // Wait for all pointers to be released
                            do {
                                val event = awaitPointerEvent()
                            } while (event.changes.any { it.pressed })
                            onPttReleased()
                        }
                    }
            ) {
                Text(
                    when {
                        recActive -> "Recording... (release to stop)"
                        playing -> "Playing back..."
                        pttActive -> "Processing..."
                        else -> "Hold to Talk"
                    },
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            if (pStatus.isNotEmpty()) {
                Text(pStatus, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
            }

            if (encStats.isNotEmpty()) {
                HorizontalDivider()
                Text("Encode Stats", fontWeight = FontWeight.Medium)
                Text(encStats, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
            }

            if (decStats.isNotEmpty()) {
                HorizontalDivider()
                Text("Decode Stats", fontWeight = FontWeight.Medium)
                Text(decStats, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
            }

            Spacer(modifier = Modifier.height(32.dp))
        }
    }

    // --- PTT Pipeline ---

    private fun onPttPressed() {
        if (isPttActive.value) return

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            return
        }

        startPttPipeline()
    }

    private fun onPttReleased() {
        recording = false
        isRecording.value = false
    }

    private fun startPttPipeline() {
        if (isPttActive.value) return
        isPttActive.value = true
        isRecording.value = true
        recording = true
        encodeStats.value = ""
        decodeStats.value = ""
        pipelineStatus.value = "Initializing..."

        val numCb = selectedCodebooks.intValue
        val chunkMs = selectedChunkMs.intValue
        val chunkSize = SAMPLE_RATE * chunkMs / 1000
        val modelPath = resolveModelPath()
        val modelLabel = selectedModel.value.label

        // Channels for pipeline stages
        val pcmChannel = Channel<FloatArray>(Channel.UNLIMITED)
        val tokenChannel = Channel<IntArray>(Channel.UNLIMITED)

        // Track per-step timings
        val encodeStepMs = mutableListOf<Double>()
        val decodeStepMs = mutableListOf<Double>()

        // 1. Recording thread — fills chunks and sends to pcmChannel
        val recordJob = lifecycleScope.launch(Dispatchers.IO) {
            try {
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
                    maxOf(bufferSize, chunkSize * 4 * 2)
                )

                val maxSamples = SAMPLE_RATE * MAX_RECORD_SECONDS
                var totalRecorded = 0
                val chunkBuf = FloatArray(chunkSize)
                var chunkPos = 0
                val readBuf = FloatArray(1920) // small read buffer

                recorder.startRecording()

                while (recording && totalRecorded < maxSamples) {
                    val toRead = minOf(readBuf.size, chunkSize - chunkPos, maxSamples - totalRecorded)
                    val read = recorder.read(readBuf, 0, toRead, AudioRecord.READ_BLOCKING)
                    if (read <= 0) continue

                    // Fill the current chunk
                    System.arraycopy(readBuf, 0, chunkBuf, chunkPos, read)
                    chunkPos += read
                    totalRecorded += read

                    if (chunkPos >= chunkSize) {
                        pcmChannel.send(chunkBuf.copyOf())
                        chunkPos = 0
                        val secs = totalRecorded.toFloat() / SAMPLE_RATE
                        withContext(Dispatchers.Main) {
                            pipelineStatus.value = "Recording... %.1fs | enc: %d | dec: %d".format(
                                secs, encodeStepMs.size, decodeStepMs.size
                            )
                        }
                    }
                }

                // Send partial last chunk (zero-padded)
                if (chunkPos > 0) {
                    // Zero the rest
                    for (i in chunkPos until chunkSize) chunkBuf[i] = 0f
                    pcmChannel.send(chunkBuf.copyOf())
                }

                recorder.stop()
                recorder.release()
            } finally {
                pcmChannel.close()
            }
        }

        // 2. Encode coroutine — reads PCM chunks, encodes, sends tokens
        val encodeJob = lifecycleScope.launch(Dispatchers.Default) {
            try {
                val encoder = MimiCodec.create(modelPath, numCb)
                encoder.reset()
                encoder.resetTimings()

                for (pcmChunk in pcmChannel) {
                    val t0 = System.nanoTime()
                    val tokens = encoder.encodeStep(pcmChunk)
                    val stepMs = (System.nanoTime() - t0) / 1_000_000.0

                    if (tokens != null) {
                        encodeStepMs.add(stepMs)
                        tokenChannel.send(tokens)

                        withContext(Dispatchers.Main) {
                            encodeStats.value = formatStepStats("Encode", encodeStepMs, chunkMs)
                        }
                    }
                }

                val timings = encoder.getTimings()
                encoder.destroy()

                withContext(Dispatchers.Main) {
                    encodeStats.value = formatFinalEncodeStats(
                        modelLabel, numCb, chunkMs, encodeStepMs, timings
                    )
                }
            } finally {
                tokenChannel.close()
            }
        }

        // 3. Decode coroutine — decode immediately as tokens arrive
        val decodedPcmChunks = mutableListOf<FloatArray>()
        val decodeJob = lifecycleScope.launch(Dispatchers.Default) {
            val decoder = MimiCodec.create(modelPath, numCb)
            decoder.reset()
            decoder.resetTimings()

            for (tokens in tokenChannel) {
                val t0 = System.nanoTime()
                val pcm = decoder.decodeStep(tokens)
                val stepMs = (System.nanoTime() - t0) / 1_000_000.0
                decodeStepMs.add(stepMs)
                if (pcm != null) decodedPcmChunks.add(pcm)

                withContext(Dispatchers.Main) {
                    decodeStats.value = formatStepStats("Decode", decodeStepMs, chunkMs)
                }
            }

            val timings = decoder.getTimings()
            decoder.destroy()

            withContext(Dispatchers.Main) {
                decodeStats.value = formatFinalDecodeStats(
                    modelLabel, chunkMs, decodeStepMs, timings
                )
            }
        }

        // Wait for pipeline to finish, then play back decoded audio
        lifecycleScope.launch {
            recordJob.join()
            encodeJob.join()
            decodeJob.join()

            // Concatenate all decoded PCM chunks
            val totalSamples = decodedPcmChunks.sumOf { it.size }
            val durationSecs = totalSamples.toFloat() / SAMPLE_RATE

            // Only play back if more than 1 second of audio (buffering threshold)
            if (durationSecs >= MIN_PLAYBACK_SECONDS) {
                withContext(Dispatchers.Main) {
                    pipelineStatus.value = "Done — %d enc, %d dec. Playing %.1fs...".format(
                        encodeStepMs.size, decodeStepMs.size, durationSecs
                    )
                    isPlaying.value = true
                }

                val fullPcm = FloatArray(totalSamples)
                var offset = 0
                for (chunk in decodedPcmChunks) {
                    System.arraycopy(chunk, 0, fullPcm, offset, chunk.size)
                    offset += chunk.size
                }

                withContext(Dispatchers.IO) {
                    val bufSize = AudioTrack.getMinBufferSize(
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_FLOAT
                    )
                    val track = AudioTrack.Builder()
                        .setAudioFormat(
                            AudioFormat.Builder()
                                .setSampleRate(SAMPLE_RATE)
                                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                                .build()
                        )
                        .setBufferSizeInBytes(maxOf(bufSize, totalSamples * 4))
                        .setTransferMode(AudioTrack.MODE_STATIC)
                        .build()

                    track.write(fullPcm, 0, fullPcm.size, AudioTrack.WRITE_BLOCKING)
                    track.play()

                    val durationMs = (totalSamples * 1000L) / SAMPLE_RATE
                    Thread.sleep(durationMs + 200)

                    track.stop()
                    track.release()
                }
            } else {
                withContext(Dispatchers.Main) {
                    pipelineStatus.value = "Done — %d enc, %d dec (%.1fs, too short to play)".format(
                        encodeStepMs.size, decodeStepMs.size, durationSecs
                    )
                }
            }

            withContext(Dispatchers.Main) {
                val duration = decodedPcmChunks.sumOf { it.size }.toFloat() / SAMPLE_RATE
                pipelineStatus.value = "Done — %d enc, %d dec, played %.1fs".format(
                    encodeStepMs.size, decodeStepMs.size, duration
                )
                isPlaying.value = false
                isPttActive.value = false
            }
        }
    }

    // --- Stats Formatting ---

    private fun formatStepStats(label: String, steps: List<Double>, chunkMs: Int): String {
        if (steps.isEmpty()) return ""
        val avg = steps.average()
        val min = steps.min()
        val max = steps.max()
        val total = steps.sum()
        val rtFactor = if (avg > 0) chunkMs / avg else 0.0
        return buildString {
            appendLine("Steps:     ${steps.size}")
            appendLine("Per step:  avg %.0f ms | min %.0f | max %.0f".format(avg, min, max))
            appendLine("Total:     %.0f ms".format(total))
            append("Realtime:  %.1fx".format(rtFactor))
        }
    }

    private fun formatFinalEncodeStats(
        modelLabel: String, numCb: Int, chunkMs: Int,
        steps: List<Double>, timings: MimiCodec.Timings?
    ): String {
        if (steps.isEmpty()) return "No encode steps"
        val avg = steps.average()
        val min = steps.min()
        val max = steps.max()
        val total = steps.sum()
        val rtFactor = if (avg > 0) chunkMs / avg else 0.0
        return buildString {
            appendLine("Model:     $modelLabel | CB: $numCb | Chunk: ${chunkMs}ms")
            appendLine("Steps:     ${steps.size}")
            appendLine("Per step:  avg %.0f ms | min %.0f | max %.0f".format(avg, min, max))
            appendLine("Total:     %.0f ms".format(total))
            appendLine("Realtime:  %.1fx".format(rtFactor))
            if (timings != null) {
                appendLine("─── Component totals ───")
                appendLine("  SEANet:       %.0f ms".format(timings.seanetEncode * 1000))
                appendLine("  Transformer:  %.0f ms".format(timings.encoderTransformer * 1000))
                appendLine("  Downsample:   %.0f ms".format(timings.downsample * 1000))
                append("  Quantizer:    %.0f ms".format(timings.quantizerEncode * 1000))
            }
        }
    }

    private fun formatFinalDecodeStats(
        modelLabel: String, chunkMs: Int,
        steps: List<Double>, timings: MimiCodec.Timings?
    ): String {
        if (steps.isEmpty()) return "No decode steps"
        val avg = steps.average()
        val min = steps.min()
        val max = steps.max()
        val total = steps.sum()
        val rtFactor = if (avg > 0) chunkMs / avg else 0.0
        return buildString {
            appendLine("Model:     $modelLabel | Chunk: ${chunkMs}ms")
            appendLine("Steps:     ${steps.size}")
            appendLine("Per step:  avg %.0f ms | min %.0f | max %.0f".format(avg, min, max))
            appendLine("Total:     %.0f ms".format(total))
            appendLine("Realtime:  %.1fx".format(rtFactor))
            if (timings != null) {
                appendLine("─── Component totals ───")
                appendLine("  Quantizer:    %.0f ms".format(timings.quantizerDecode * 1000))
                appendLine("  Upsample:     %.0f ms".format(timings.upsample * 1000))
                appendLine("  Transformer:  %.0f ms".format(timings.decoderTransformer * 1000))
                append("  SEANet:       %.0f ms".format(timings.seanetDecode * 1000))
            }
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
}
