package com.example.mimi

class MimiCodec private constructor(private var handle: Long) {

    companion object {
        init {
            System.loadLibrary("mimi_jni")
        }

        fun create(modelPath: String, numCodebooks: Int): MimiCodec {
            val handle = nativeCreate(modelPath, numCodebooks)
            if (handle == 0L) throw RuntimeException("Failed to create Mimi codec")
            return MimiCodec(handle)
        }

        fun createOnnx(
            encoderPath: String,
            decoderPath: String,
            numCodebooks: Int,
            useNnapi: Boolean,
            streaming: Boolean = true,
        ): MimiCodec {
            val handle = nativeCreateOnnx(encoderPath, decoderPath, numCodebooks, useNnapi, streaming)
            if (handle == 0L) throw RuntimeException("Failed to create ONNX codec")
            return MimiCodec(handle)
        }

        @JvmStatic
        private external fun nativeCreate(modelPath: String, numCodebooks: Int): Long

        @JvmStatic
        private external fun nativeCreateOnnx(
            encoderPath: String,
            decoderPath: String,
            numCodebooks: Int,
            useNnapi: Boolean,
            streaming: Boolean,
        ): Long

        @JvmStatic
        private external fun nativeEncode(handle: Long, pcm: FloatArray): IntArray

        @JvmStatic
        private external fun nativeDecode(handle: Long, codes: IntArray, numCodebooks: Int): FloatArray

        @JvmStatic
        private external fun nativeEncodeStep(handle: Long, pcm: FloatArray): IntArray?

        @JvmStatic
        private external fun nativeDecodeStep(handle: Long, codes: IntArray): FloatArray?

        @JvmStatic
        private external fun nativeReset(handle: Long)

        @JvmStatic
        private external fun nativeDestroy(handle: Long)

        @JvmStatic
        private external fun nativeGetTimings(handle: Long): FloatArray?

        @JvmStatic
        private external fun nativeResetTimings(handle: Long)
    }

    /**
     * Per-component timing breakdown (accumulated seconds).
     * Order: seanetEnc, encTransformer, downsample, quantEnc,
     *        quantDec, upsample, decTransformer, seanetDec, steps
     */
    data class Timings(
        val seanetEncode: Float,
        val encoderTransformer: Float,
        val downsample: Float,
        val quantizerEncode: Float,
        val quantizerDecode: Float,
        val upsample: Float,
        val decoderTransformer: Float,
        val seanetDecode: Float,
        val steps: Int,
    )

    /** Batch encode entire audio buffer at once. Returns flat codes (time-major). */
    fun encode(pcm: FloatArray): IntArray = nativeEncode(handle, pcm)

    /** Batch decode entire code sequence at once. Returns PCM samples. */
    fun decode(codes: IntArray, numCodebooks: Int): FloatArray = nativeDecode(handle, codes, numCodebooks)

    fun encodeStep(pcm: FloatArray): IntArray? = nativeEncodeStep(handle, pcm)

    fun decodeStep(codes: IntArray): FloatArray? = nativeDecodeStep(handle, codes)

    fun reset() = nativeReset(handle)

    fun getTimings(): Timings? {
        val arr = nativeGetTimings(handle) ?: return null
        if (arr.size < 9) return null
        return Timings(
            seanetEncode = arr[0],
            encoderTransformer = arr[1],
            downsample = arr[2],
            quantizerEncode = arr[3],
            quantizerDecode = arr[4],
            upsample = arr[5],
            decoderTransformer = arr[6],
            seanetDecode = arr[7],
            steps = arr[8].toInt(),
        )
    }

    fun resetTimings() = nativeResetTimings(handle)

    fun destroy() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }
}
