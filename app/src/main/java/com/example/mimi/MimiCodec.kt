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

        @JvmStatic
        private external fun nativeCreate(modelPath: String, numCodebooks: Int): Long

        @JvmStatic
        private external fun nativeEncodeStep(handle: Long, pcm: FloatArray): IntArray?

        @JvmStatic
        private external fun nativeDecodeStep(handle: Long, codes: IntArray): FloatArray?

        @JvmStatic
        private external fun nativeReset(handle: Long)

        @JvmStatic
        private external fun nativeDestroy(handle: Long)
    }

    fun encodeStep(pcm: FloatArray): IntArray? = nativeEncodeStep(handle, pcm)

    fun decodeStep(codes: IntArray): FloatArray? = nativeDecodeStep(handle, codes)

    fun reset() = nativeReset(handle)

    fun destroy() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }
}
