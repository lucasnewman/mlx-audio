//
//  FixedChineseTest.swift
//  MLXAudio
//
//  Fixed Chinese TTS test - automatically downloads Chinese model from HuggingFace
//
//  Key fix: The original code only loaded G2P dictionaries but still used English model weights.
//  This version automatically downloads the full Chinese model (weights + G2P + voices) from Hub.
//

import Foundation

// MARK: - Fixed Chinese TTS Test

/// Fixed version of runChineseTest()
///
/// The issue with the original code was that `initializeChineseSupport()` only loaded
/// G2P dictionaries, but the underlying model was still the bundled English kokoro-v1_0.safetensors.
/// This caused garbled audio because Chinese phonemes were being fed into English model weights.
///
/// This fixed version automatically downloads the complete Chinese model from HuggingFace
/// when a Chinese voice is used.
private func runChineseTest() {
    Task {
        // No need to manually initialize Chinese support anymore!
        // When you use a Chinese voice, the model is automatically downloaded from HuggingFace.

        status = "Generating Chinese (will auto-download model on first use)..."

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        print("📂 App Documents Directory: \(docs.path)")

        // Just call say() with a Chinese voice - it will:
        // 1. Detect it's a Chinese voice
        // 2. Download the Chinese model from HuggingFace (FluidInference/kokoro-82m-v1.1-zh-mlx)
        // 3. Download Chinese G2P dictionaries
        // 4. Download Chinese voice files (voices/zf_001.npy, etc.)
        // 5. Generate audio using the correct Chinese model
        kokoroTTSModel.say(
            "你好，世界！这是一项测试。",
            .zfXiaoxiao,
            speed: 1.0,
            autoPlay: true
        )
    }
}

// MARK: - Alternative: Manual Control

/// If you want more control over the download process, you can use KokoroTTS.fromHub() directly:
private func runChineseTestManual() {
    Task {
        do {
            // 1. Download Chinese model from HuggingFace
            status = "Downloading Chinese model..."
            let chineseTTS = try await KokoroTTS.fromHub(
                repoId: "FluidInference/kokoro-82m-v1.1-zh-mlx"
            ) { progress in
                DispatchQueue.main.async {
                    status = "Downloading: \(Int(progress.fractionCompleted * 100))%"
                }
            }

            // 2. Generate audio
            status = "Generating Chinese speech..."
            var audioSamples: [Float] = []

            try chineseTTS.generateAudio(
                voice: .zfXiaoxiao,
                text: "你好，世界！这是一项测试。",
                speed: 1.0
            ) { chunk in
                chunk.eval()
                if chunk.shape.count == 1 {
                    audioSamples.append(contentsOf: chunk.asArray(Float.self))
                } else {
                    let batch = chunk[0]
                    batch.eval()
                    audioSamples.append(contentsOf: batch.asArray(Float.self))
                }
            }

            // Wait for async generation
            try await Task.sleep(nanoseconds: 2_000_000_000)

            // 3. Save WAV
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let wavPath = docs.appendingPathComponent("chinese_output.wav")
            saveWAV(samples: audioSamples, to: wavPath)

            status = "Done! Saved to: \(wavPath.path)"
            print("📂 WAV saved: \(wavPath.path)")

        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Helper

private func saveWAV(samples: [Float], to url: URL) {
    var data = Data()
    let dataSize = UInt32(samples.count * 2)

    // WAV header
    data.append(contentsOf: "RIFF".utf8)
    data.append(contentsOf: withUnsafeBytes(of: (36 + dataSize).littleEndian) { Array($0) })
    data.append(contentsOf: "WAVEfmt ".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // Mono
    data.append(contentsOf: withUnsafeBytes(of: UInt32(24000).littleEndian) { Array($0) })  // Sample rate
    data.append(contentsOf: withUnsafeBytes(of: UInt32(48000).littleEndian) { Array($0) })  // Byte rate
    data.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) })  // Block align
    data.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) })  // Bits per sample
    data.append(contentsOf: "data".utf8)
    data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

    // Audio data
    for sample in samples {
        let i16 = Int16(max(-1, min(1, sample)) * 32767)
        data.append(contentsOf: withUnsafeBytes(of: i16.littleEndian) { Array($0) })
    }

    try? data.write(to: url)
}

// MARK: - Required Properties (add these to your view/controller)

// @StateObject private var kokoroTTSModel = KokoroTTSModel()
// @State private var status = ""
