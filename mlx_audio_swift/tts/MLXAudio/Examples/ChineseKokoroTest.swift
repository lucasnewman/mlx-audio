//
//  ChineseKokoroTest.swift
//  MLXAudio
//
//  Simple test to generate Chinese TTS audio and save to WAV
//

import Foundation
import MLX
import AVFoundation

/// Run this test to generate Chinese TTS audio
/// Usage: Call `ChineseKokoroTest.run()` from your app or test target
public class ChineseKokoroTest {

    public static func run() async {
        print("=== Chinese Kokoro TTS Test ===")

        do {
            // 1. Load Chinese model from HuggingFace
            print("[1/4] Downloading Chinese Kokoro model from HuggingFace...")
            let tts = try await KokoroTTS.fromHub { progress in
                print("  Download progress: \(Int(progress.fractionCompleted * 100))%")
            }
            print("  Model loaded successfully!")

            // 2. Generate audio for Chinese text
            let chineseText = "你好，世界！这是一个中文语音合成测试。"
            let voice: TTSVoice = .zfXiaoxiao  // Female Chinese voice

            print("[2/4] Generating audio for: \"\(chineseText)\"")
            print("  Using voice: \(voice)")

            var audioChunks: [MLXArray] = []

            try tts.generateAudio(voice: voice, text: chineseText, speed: 1.0) { chunk in
                audioChunks.append(chunk)
                print("  Received audio chunk: \(chunk.shape) samples")
            }

            // Wait for generation to complete
            try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds

            print("  Total chunks received: \(audioChunks.count)")

            // 3. Combine audio chunks
            print("[3/4] Combining audio chunks...")
            guard !audioChunks.isEmpty else {
                print("  ERROR: No audio chunks generated!")
                return
            }

            let combinedAudio = MLX.concatenated(audioChunks, axis: 0)
            combinedAudio.eval()

            let audioData: [Float] = combinedAudio.asArray(Float.self)
            print("  Combined audio: \(audioData.count) samples")

            // 4. Save to WAV file
            print("[4/4] Saving to WAV file...")
            let outputPath = "/tmp/chinese_kokoro_output.wav"
            try saveToWav(samples: audioData, sampleRate: 24000, path: outputPath)

            print("\n=== SUCCESS ===")
            print("Chinese TTS audio saved to: \(outputPath)")
            print("Duration: \(String(format: "%.2f", Double(audioData.count) / 24000.0)) seconds")

        } catch {
            print("\n=== ERROR ===")
            print("Failed to generate Chinese TTS: \(error)")
        }
    }

    private static func saveToWav(samples: [Float], sampleRate: Int, path: String) throws {
        let url = URL(fileURLWithPath: path)

        guard let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1) else {
            throw NSError(domain: "ChineseKokoroTest", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"])
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw NSError(domain: "ChineseKokoroTest", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
        }

        buffer.frameLength = buffer.frameCapacity

        guard let channelData = buffer.floatChannelData else {
            throw NSError(domain: "ChineseKokoroTest", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to get channel data"])
        }

        for i in 0..<samples.count {
            channelData[0][i] = samples[i]
        }

        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        try audioFile.write(from: buffer)
    }
}
