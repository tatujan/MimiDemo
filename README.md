# Mimi Codec Demo

Android demo app for the [Mimi](https://github.com/kyutai-labs/moshi) streaming neural audio codec (24 kHz, 12.5 Hz frame rate). Record audio from the mic, encode it to compact codec tokens, then decode and play back — all on-device.

## Features

- **Model selector** — switch between 8-codebook (~1.1 kbps) and 16-codebook (~2.2 kbps) variants
- **Live stats** — encode/decode time, realtime factor, compression ratio, token size
- **Streaming codec** — frame-by-frame encode/decode with causal attention masking (320 ms chunks)
- **ONNX Runtime** — fast inference via ONNX Runtime with NNAPI hardware acceleration

## Architecture

```
app/src/main/java/com/example/mimi/
  MainActivity.kt   — Compose UI, recording, encode/decode orchestration
  MimiCodec.kt      — JNI wrapper (thin, ~40 lines)

app/src/main/jniLibs/
  arm64-v8a/libmimi_jni.so   — native ONNX Runtime bridge (built from mimi-codec)

app/src/main/assets/
  onnx_8cb/    — 8-codebook streaming ONNX models
  onnx_16cb/   — 16-codebook streaming ONNX models
```

Two Kotlin files. No ViewModel, no DI, no navigation. Models are bundled in the APK as assets and copied to internal storage on first launch.

## ONNX Models

Streaming ONNX models are available on Hugging Face: [BMekiker/mimi-onnx-streaming](https://huggingface.co/BMekiker/mimi-onnx-streaming)

| Variant | Encoder | Decoder | Bitrate |
|---------|---------|---------|---------|
| 8 codebooks | 194 MB | 170 MB | ~1.1 kbps |
| 16 codebooks | 242 MB | 186 MB | ~2.2 kbps |

Each model directory contains:
- `encoder_model.onnx` — streaming encoder (PCM → codes)
- `decoder_model.onnx` — streaming decoder (codes → PCM)
- `state_spec.txt` — conv state and KV cache tensor specifications

The models use causal attention masks and explicit conv state / KV cache I/O for frame-by-frame streaming without re-processing past audio.

### Downloading models manually

```bash
pip install huggingface-hub

# 8-codebook variant
huggingface-cli download BMekiker/mimi-onnx-streaming streaming-8cb/ --local-dir onnx-models/

# 16-codebook variant
huggingface-cli download BMekiker/mimi-onnx-streaming streaming-16cb/ --local-dir onnx-models/
```

## Prerequisites

- Android Studio or the Android SDK (compileSdk 34, minSdk 24)
- Pre-built `libmimi_jni.so` for your target architecture (see [Building the native library](#building-the-native-library))

## Building the native library

The JNI bridge lives in a separate Rust crate (`mimi-codec/mimi-jni`). You need the Android NDK and [`cargo-ndk`](https://github.com/nicokosi/cargo-ndk):

```bash
# Install cargo-ndk
cargo install cargo-ndk

# Build for ARM64 (physical devices)
cd mimi-codec/mimi-jni
cargo ndk -t arm64-v8a -P 24 build --release

# Copy into the Android project
mkdir -p ../MimiDemo/app/src/main/jniLibs/arm64-v8a
cp target/aarch64-linux-android/release/libmimi_jni.so ../MimiDemo/app/src/main/jniLibs/arm64-v8a/
```

## Build & run

```bash
# Set your SDK path
echo "sdk.dir=/path/to/android/sdk" > local.properties

# Build
./gradlew assembleDebug

# Install
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Models are bundled in the APK — no manual file transfers needed.

## Performance

Typical performance on modern ARM64 devices is ~2x realtime for both encode and decode, with NNAPI acceleration enabled. The app displays realtime factors after each run.

## License

This project contains code derived from [Moshi](https://github.com/kyutai-labs/moshi) by [Kyutai](https://kyutai.ai/).

- **Code** (Rust codec implementation, JNI bridge): MIT / Apache-2.0 dual license — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE)
- **Model weights** (Mimi codec): [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Kyutai

## Acknowledgments

The Mimi streaming neural audio codec was developed by Kyutai as part of the [Moshi](https://github.com/kyutai-labs/moshi) project.
