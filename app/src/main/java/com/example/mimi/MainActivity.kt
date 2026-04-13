@file:OptIn(ExperimentalLayoutApi::class)

package com.example.mimi

import android.Manifest
import android.util.Log
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
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
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
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

enum class ModelVariant(val label: String, val filename: String, val hfSubdir: String) {
    ONNX_8CB("ONNX-8cb", "onnx_8cb", "streaming-8cb"),
    ONNX_16CB("ONNX-16cb", "onnx_16cb", "streaming-16cb"),
    ONNX_8CB_FP16("ONNX-8cb-fp16", "onnx_8cb_fp16", "streaming-8cb-fp16"),
    ONNX_16CB_FP16("ONNX-16cb-fp16", "onnx_16cb_fp16", "streaming-16cb-fp16");

    val isOnnx: Boolean get() = true

    /** ONNX models have a fixed codebook count baked into the model. */
    val fixedCodebooks: Int? get() = when (this) {
        ONNX_8CB, ONNX_8CB_FP16 -> 8
        ONNX_16CB, ONNX_16CB_FP16 -> 16
    }

    companion object {
        const val HF_REPO = "BMekiker/mimi-onnx-streaming"
        const val HF_BASE_URL = "https://huggingface.co/$HF_REPO/resolve/main"
        val MODEL_FILES = listOf("encoder_model.onnx", "decoder_model.onnx", "state_spec.txt")
    }
}

class MainActivity : ComponentActivity() {

    companion object {
        private const val SAMPLE_RATE = 24000
        private const val MAX_RECORD_SECONDS = 60
        private const val MIN_PLAYBACK_SECONDS = 1.0f
    }

    // UI state
    private var modelStatus = mutableStateOf("Checking models...")
    private var modelReady = mutableStateOf(false)
    private var selectedModel = mutableStateOf(ModelVariant.ONNX_8CB)
    private var availableModels = mutableStateOf(emptySet<ModelVariant>())
    private var selectedCodebooks = mutableIntStateOf(8)
    private var selectedChunkMs = mutableIntStateOf(320)
    private var isPttActive = mutableStateOf(false)
    private var isRecording = mutableStateOf(false)
    private var isPlaying = mutableStateOf(false)
    private var pipelineStatus = mutableStateOf("")
    private var encodeStats = mutableStateOf("")
    private var decodeStats = mutableStateOf("")

    // Pre-created codecs (avoids loading models on each PTT press)
    private var encoderCodec: MimiCodec? = null
    private var decoderCodec: MimiCodec? = null
    private var cachedVariant: ModelVariant? = null
    private var cachedNumCb: Int = 0
    private var codecsReady = mutableStateOf(false)

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

    override fun onDestroy() {
        super.onDestroy()
        encoderCodec?.destroy()
        decoderCodec?.destroy()
        encoderCodec = null
        decoderCodec = null
    }

    private var prepareJob: Job? = null

    /** Pre-load codec instances so PTT starts instantly. Downloads model if needed. */
    private fun prepareCodecs(variant: ModelVariant, numCb: Int) {
        if (variant == cachedVariant && numCb == cachedNumCb && codecsReady.value) return
        cachedVariant = variant
        cachedNumCb = numCb
        codecsReady.value = false
        modelStatus.value = "Loading ${variant.label}..."
        prepareJob?.cancel()
        prepareJob = lifecycleScope.launch(Dispatchers.IO) {
            encoderCodec?.destroy()
            decoderCodec?.destroy()
            encoderCodec = null
            decoderCodec = null
            try {
                if (!ensureModelDownloaded(variant)) return@launch
                withContext(Dispatchers.Main) {
                    modelStatus.value = "Loading ${variant.label}..."
                }
                val enc = createCodec(variant, numCb)
                val dec = createCodec(variant, numCb)
                encoderCodec = enc
                decoderCodec = dec
                withContext(Dispatchers.Main) {
                    codecsReady.value = true
                    modelStatus.value = "Ready: ${variant.label}"
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    modelStatus.value = "Failed to load ${variant.label}: ${e.message}"
                }
            }
        }
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
        val codecReady by codecsReady

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
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ModelVariant.entries.forEach { variant ->
                    val isAvailable = variant in available
                    FilterChip(
                        selected = model == variant,
                        onClick = {
                            selectedModel.value = variant
                            val numCb = variant.fixedCodebooks ?: codebooks
                            prepareCodecs(variant, numCb)
                        },
                        label = { Text(variant.label) },
                        enabled = isAvailable && !pttActive
                    )
                }
            }

            HorizontalDivider()

            // Codebook selector
            val isOnnxModel = model.isOnnx
            val effectiveCodebooks = model.fixedCodebooks ?: codebooks
            Text(
                if (isOnnxModel) "Codebooks: $effectiveCodebooks (fixed by ONNX model)"
                else "Codebooks:",
                fontWeight = FontWeight.Medium
            )
            if (!isOnnxModel) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    codebookOptions.forEach { n ->
                        FilterChip(
                            selected = codebooks == n,
                            onClick = {
                                selectedCodebooks.intValue = n
                                prepareCodecs(model, n)
                            },
                            label = { Text("$n") },
                            enabled = !pttActive
                        )
                    }
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
                enabled = ready && codecReady && !playing,
                colors = if (recActive) ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error
                ) else ButtonDefaults.buttonColors(),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(64.dp)
                    .pointerInput(ready, codecReady, playing) {
                        if (!ready || !codecReady || playing) return@pointerInput
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

        val variant = selectedModel.value
        val numCb = variant.fixedCodebooks ?: selectedCodebooks.intValue
        val chunkMs = selectedChunkMs.intValue
        val chunkSize = SAMPLE_RATE * chunkMs / 1000
        val modelLabel = variant.label

        // Grab pre-created codecs
        val encoder = encoderCodec ?: return
        val decoder = decoderCodec ?: return
        encoder.reset()
        encoder.resetTimings()
        decoder.reset()
        decoder.resetTimings()

        // Channels for pipeline stages
        val pcmChannel = Channel<FloatArray>(Channel.UNLIMITED)
        val tokenChannel = Channel<IntArray>(Channel.UNLIMITED)

        // Track per-step timings and data sizes
        val encodeStepMs = mutableListOf<Double>()
        val decodeStepMs = mutableListOf<Double>()
        var totalPcmSamples = 0
        var totalEncodedCodes = 0

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
                var frameIdx = 0
                for (pcmChunk in pcmChannel) {
                    totalPcmSamples += pcmChunk.size
                    val t0 = System.nanoTime()
                    val tokens = encoder.encodeStep(pcmChunk)
                    val stepMs = (System.nanoTime() - t0) / 1_000_000.0

                    if (frameIdx < 10 || frameIdx % 50 == 0) {
                        Log.d("MimiEncode", "frame=$frameIdx pcm=${pcmChunk.size} tokens=${tokens?.size} ms=${"%.1f".format(stepMs)}")
                    }
                    frameIdx++

                    if (tokens != null) {
                        encodeStepMs.add(stepMs)
                        totalEncodedCodes += tokens.size
                        tokenChannel.send(tokens)

                        withContext(Dispatchers.Main) {
                            encodeStats.value = formatStepStats("Encode", encodeStepMs, chunkMs)
                        }
                    }
                }

                val timings = encoder.getTimings()

                withContext(Dispatchers.Main) {
                    encodeStats.value = formatFinalEncodeStats(
                        modelLabel, numCb, chunkMs, encodeStepMs, timings,
                        totalPcmSamples, totalEncodedCodes
                    )
                }
            } finally {
                tokenChannel.close()
            }
        }

        // 3. Decode coroutine — decode immediately as tokens arrive
        val decodedPcmChunks = mutableListOf<FloatArray>()
        val decodeJob = lifecycleScope.launch(Dispatchers.Default) {
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
        steps: List<Double>, timings: MimiCodec.Timings?,
        totalPcmSamples: Int, totalEncodedCodes: Int
    ): String {
        if (steps.isEmpty()) return "No encode steps"
        val avg = steps.average()
        val min = steps.min()
        val max = steps.max()
        val total = steps.sum()
        val rtFactor = if (avg > 0) chunkMs / avg else 0.0
        val audioDurationSecs = totalPcmSamples.toDouble() / SAMPLE_RATE
        // Each code = 11 bits (2048 codebook entries = 2^11)
        val bitsPerCode = 11
        val encodedBits = totalEncodedCodes.toLong() * bitsPerCode
        val encodedBytes = (encodedBits + 7) / 8
        val rawPcmBytes = totalPcmSamples.toLong() * 2  // 16-bit PCM
        val bitrate = if (audioDurationSecs > 0) encodedBits / audioDurationSecs else 0.0
        val compressionRatio = if (encodedBytes > 0) rawPcmBytes.toDouble() / encodedBytes else 0.0
        return buildString {
            appendLine("Model:     $modelLabel | CB: $numCb | Chunk: ${chunkMs}ms")
            appendLine("Steps:     ${steps.size}")
            appendLine("Per step:  avg %.0f ms | min %.0f | max %.0f".format(avg, min, max))
            appendLine("Total:     %.0f ms".format(total))
            appendLine("Realtime:  %.1fx".format(rtFactor))
            appendLine("─── Compression ───")
            appendLine("Audio:     %.1f s (%s PCM @ 16-bit)".format(
                audioDurationSecs, formatBytes(rawPcmBytes)))
            appendLine("Encoded:   %d codes × %d bits = %s".format(
                totalEncodedCodes, bitsPerCode, formatBytes(encodedBytes)))
            appendLine("Bitrate:   %.1f kbps".format(bitrate / 1000.0))
            appendLine("Compress:  %.0fx".format(compressionRatio))
            if (timings != null) {
                appendLine("─── Component totals ───")
                appendLine("  SEANet:       %.0f ms".format(timings.seanetEncode * 1000))
                appendLine("  Transformer:  %.0f ms".format(timings.encoderTransformer * 1000))
                appendLine("  Downsample:   %.0f ms".format(timings.downsample * 1000))
                append("  Quantizer:    %.0f ms".format(timings.quantizerEncode * 1000))
            }
        }
    }

    private fun formatBytes(bytes: Long): String {
        return when {
            bytes < 1024 -> "$bytes B"
            bytes < 1024 * 1024 -> "%.1f KB".format(bytes / 1024.0)
            else -> "%.1f MB".format(bytes / (1024.0 * 1024.0))
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
        // All variants are always available — models download on demand from HuggingFace
        val allVariants = ModelVariant.entries.toSet()
        availableModels.value = allVariants

        // Check which are already downloaded
        val local = allVariants.filter { isModelLocal(it) }
        val defaultVariant = if (local.isNotEmpty()) local.first() else ModelVariant.ONNX_8CB_FP16
        selectedModel.value = defaultVariant
        modelReady.value = true

        if (local.isNotEmpty()) {
            modelStatus.value = "Cached: ${local.joinToString(", ") { it.label }}"
        } else {
            modelStatus.value = "Select a model — downloads from HuggingFace on first use"
        }

        val numCb = defaultVariant.fixedCodebooks ?: selectedCodebooks.intValue
        prepareCodecs(defaultVariant, numCb)
    }

    private fun isModelLocal(variant: ModelVariant): Boolean {
        val dir = File(filesDir, variant.filename)
        return File(dir, "encoder_model.onnx").exists() && File(dir, "decoder_model.onnx").exists()
    }

    /** Ensure model files exist locally, downloading from HuggingFace if needed. */
    private suspend fun ensureModelDownloaded(variant: ModelVariant): Boolean {
        val localDir = File(filesDir, variant.filename)
        val enc = File(localDir, "encoder_model.onnx")
        val dec = File(localDir, "decoder_model.onnx")
        if (enc.exists() && dec.exists()) return true

        // Try adb fallback first (fast, for development)
        val adbDir = File("/data/local/tmp/${variant.filename}")
        if (File(adbDir, "encoder_model.onnx").exists() && File(adbDir, "decoder_model.onnx").exists()) {
            withContext(Dispatchers.Main) {
                modelStatus.value = "Copying ${variant.label} from local..."
            }
            try {
                localDir.mkdirs()
                for (name in ModelVariant.MODEL_FILES) {
                    val src = File(adbDir, name)
                    if (src.exists()) src.copyTo(File(localDir, name), overwrite = true)
                }
                return true
            } catch (_: Exception) { }
        }

        // Download from HuggingFace
        localDir.mkdirs()
        for (name in ModelVariant.MODEL_FILES) {
            val dest = File(localDir, name)
            if (dest.exists()) continue
            val url = "${ModelVariant.HF_BASE_URL}/${variant.hfSubdir}/$name"
            withContext(Dispatchers.Main) {
                modelStatus.value = "Downloading ${variant.label}: $name..."
            }
            try {
                downloadFile(url, dest) { progress ->
                    modelStatus.value = "Downloading ${variant.label}: $name ${progress}%"
                }
            } catch (e: Exception) {
                Log.e("MimiDownload", "Failed to download ${variant.label}: $name", e)
                localDir.deleteRecursively()
                withContext(Dispatchers.Main) {
                    modelStatus.value = "Download failed: ${e.message}"
                }
                return false
            }
        }
        return true
    }

    private suspend fun downloadFile(url: String, dest: File, onProgress: (Int) -> Unit) {
        withContext(Dispatchers.IO) {
            val tmp = File(dest.parent, "${dest.name}.tmp")
            val maxRetries = 3
            for (attempt in 1..maxRetries) {
                try {
                    // Follow redirects manually (HuggingFace uses 307)
                    var currentUrl = url
                    var conn: HttpURLConnection
                    var redirects = 0
                    while (true) {
                        conn = URL(currentUrl).openConnection() as HttpURLConnection
                        conn.connectTimeout = 15_000
                        conn.readTimeout = 300_000
                        conn.instanceFollowRedirects = false
                        conn.setRequestProperty("User-Agent", "MimiDemo/1.0")
                        // Resume partial download
                        val existingBytes = if (tmp.exists()) tmp.length() else 0L
                        if (existingBytes > 0) {
                            conn.setRequestProperty("Range", "bytes=$existingBytes-")
                        }
                        conn.connect()
                        val code = conn.responseCode
                        if (code in 301..308) {
                            val location = conn.getHeaderField("Location") ?: break
                            currentUrl = if (location.startsWith("http")) location
                                         else "${URL(currentUrl).protocol}://${URL(currentUrl).host}$location"
                            conn.disconnect()
                            if (++redirects > 5) throw RuntimeException("Too many redirects")
                            continue
                        }
                        break
                    }

                    if (conn.responseCode != 200 && conn.responseCode != 206) {
                        throw RuntimeException("HTTP ${conn.responseCode}")
                    }

                    val resumed = conn.responseCode == 206
                    val existingBytes = if (resumed && tmp.exists()) tmp.length() else 0L
                    val contentLength = conn.contentLengthLong
                    val total = if (resumed) existingBytes + contentLength else contentLength

                    var downloaded = existingBytes

                    conn.inputStream.buffered().use { input ->
                        tmp.outputStream(resumed).buffered().use { output ->
                            val buf = ByteArray(256 * 1024)
                            while (true) {
                                val n = input.read(buf)
                                if (n < 0) break
                                output.write(buf, 0, n)
                                downloaded += n
                                if (total > 0 && downloaded % (1024 * 1024) < buf.size) {
                                    withContext(Dispatchers.Main) {
                                        onProgress((downloaded * 100 / total).toInt())
                                    }
                                }
                            }
                        }
                    }
                    tmp.renameTo(dest)
                    return@withContext
                } catch (e: Exception) {
                    if (attempt == maxRetries) {
                        tmp.delete()
                        throw e
                    }
                    Log.w("MimiDownload", "Attempt $attempt/$maxRetries failed, retrying: ${e.message}")
                    kotlinx.coroutines.delay(2000L * attempt)
                }
            }
        }
    }

    /** Open file for writing — append mode when resuming. */
    private fun File.outputStream(append: Boolean): java.io.FileOutputStream {
        return java.io.FileOutputStream(this, append)
    }

    private fun createCodec(variant: ModelVariant, numCodebooks: Int): MimiCodec {
        val dir = File(filesDir, variant.filename)
        return MimiCodec.createOnnx(
            encoderPath = File(dir, "encoder_model.onnx").absolutePath,
            decoderPath = File(dir, "decoder_model.onnx").absolutePath,
            numCodebooks = variant.fixedCodebooks ?: numCodebooks,
            useNnapi = true,
            streaming = true,
        )
    }
}
