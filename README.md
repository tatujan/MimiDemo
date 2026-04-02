# Mimi Codec Demo

Android demo app for the [Mimi](https://github.com/kyutai-labs/moshi) streaming neural audio codec (24 kHz, 12.5 Hz frame rate, ~1.1 kbps at 8 codebooks).

Record audio from the mic, encode it to compact codec tokens, then decode and play back — all on-device using a Rust native library via JNI.

## Features

- **Model selector** — switch between FP32 (safetensors), Q8 (GGUF Q8_0), and Q4 (GGUF Q4_0) at runtime to compare speed vs quality
- **Codebook selector** — 1 / 2 / 4 / 8 / 16 codebooks to tune compression ratio
- **Live stats** — encode/decode time, realtime factor, compression ratio, token size
- **Streaming codec** — frame-by-frame encode/decode (80 ms frames, 1920 samples)

## Architecture

```
app/src/main/java/com/example/mimi/
  MainActivity.kt   — Compose UI, recording, encode/decode orchestration
  MimiCodec.kt      — JNI wrapper (thin, ~40 lines)

app/src/main/jniLibs/
  arm64-v8a/libmimi_jni.so   — native codec (built from mimi-codec)
  x86_64/libmimi_jni.so
```

Two Kotlin files. No ViewModel, no DI, no navigation.

## Prerequisites

- Android Studio or the Android SDK (compileSdk 34, minSdk 24)
- Pre-built `libmimi_jni.so` for your target architecture (see [Building the native library](#building-the-native-library))
- A Mimi model file (see [Model files](#model-files))

## Building the native library

The JNI bridge lives in a separate Rust crate (`mimi-codec/mimi-jni`). You need the Android NDK and [`cargo-ndk`](https://github.com/nicokosi/cargo-ndk):

```bash
# Install cargo-ndk
cargo install cargo-ndk

# Build for ARM64 (physical devices)
cd mimi-codec/mimi-jni
cargo ndk -t arm64-v8a -P 24 build --release

# Build for x86_64 (emulators)
cargo ndk -t x86_64 -P 24 build --release

# Copy into the Android project
mkdir -p ../MimiDemo/app/src/main/jniLibs/arm64-v8a
mkdir -p ../MimiDemo/app/src/main/jniLibs/x86_64
cp target/aarch64-linux-android/release/libmimi_jni.so ../MimiDemo/app/src/main/jniLibs/arm64-v8a/
cp target/x86_64-linux-android/release/libmimi_jni.so  ../MimiDemo/app/src/main/jniLibs/x86_64/
```

## Model files

The app supports three model formats. On launch it scans for all available models and lets you switch between them.

| Variant | Filename | Size | Source |
|---------|----------|------|--------|
| FP32 | `mimi_model.safetensors` | 367 MB | Auto-downloaded from HuggingFace on first launch |
| Q8 | `mimi_q8.gguf` | 138 MB | Convert with `convert_to_gguf.py -q q8_0` |
| Q4 | `mimi_q4_0.gguf` | 114 MB | Convert with `convert_to_gguf.py -q q4_0` |

### Converting to GGUF

```bash
pip install safetensors gguf numpy

# Q8_0 (good quality, ~2x faster than FP32)
python mimi-codec/scripts/convert_to_gguf.py <safetensors_path> -q q8_0

# Q4_0 (smaller, ~3x faster than FP32, slight quality loss)
python mimi-codec/scripts/convert_to_gguf.py <safetensors_path> -q q4_0
```

### Pushing models to the device

GGUF models are not auto-downloaded — push them via adb before launching the app:

```bash
adb push mimi_q4_0.gguf /data/local/tmp/mimi_q4_0.gguf
adb push mimi_q8.gguf    /data/local/tmp/mimi_q8.gguf
```

The app copies them from `/data/local/tmp/` to its private storage on first launch. If no models are found, it falls back to downloading FP32 from HuggingFace.

## Build & run

```bash
# Set your SDK path
echo "sdk.dir=/path/to/android/sdk" > local.properties

# Build
./gradlew assembleDebug

# Install
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Performance

Performance varies by device. The app displays realtime factors for encode and decode after each run. Quantized models (Q4, Q8) are significantly faster than FP32 due to reduced memory bandwidth and optimized integer matmuls.

## License

This project contains code derived from [Moshi](https://github.com/kyutai-labs/moshi) by [Kyutai](https://kyutai.ai/).

- **Code** (Rust codec implementation, JNI bridge): MIT / Apache-2.0 dual license — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE)
- **Model weights** (Mimi codec): [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Kyutai

## Acknowledgments

The Mimi streaming neural audio codec was developed by Kyutai as part of the [Moshi](https://github.com/kyutai-labs/moshi) project.
