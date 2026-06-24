package com.example.tinybayes

import android.content.Context
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

object FileUtil {

    fun loadMappedFile(
        context: Context,
        filename: String
    ): MappedByteBuffer {

        val fileDescriptor =
            context.assets.openFd(filename)

        val inputStream =
            fileDescriptor.createInputStream()

        val channel =
            inputStream.channel

        return channel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }
}