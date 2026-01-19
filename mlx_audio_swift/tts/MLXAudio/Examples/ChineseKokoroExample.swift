//
//  ChineseKokoroExample.swift
//  MLXAudio
//
//  Example showing how to use Chinese Kokoro TTS
//

import Foundation
import SwiftUI

/// Example view demonstrating Chinese TTS
struct ChineseKokoroExampleView: View {
    @StateObject private var kokoroTTSModel = KokoroTTSModel()
    @State private var status = "Ready"
    @State private var chineseTTS: KokoroTTS?
    @State private var isLoading = false

    var body: some View {
        VStack(spacing: 20) {
            Text("Chinese Kokoro TTS Example")
                .font(.title)

            Text(status)
                .foregroundColor(.secondary)

            if isLoading {
                ProgressView()
            }

            Button("Generate Chinese Speech") {
                runChineseTest()
            }
            .disabled(isLoading)
        }
        .padding()
    }

    private func runChineseTest() {
        Task {
            isLoading = true
            defer { isLoading = false }

            do {
                // 1. Load Chinese model from HuggingFace (includes model + G2P + voices)
                if chineseTTS == nil {
                    status = "Downloading Chinese model from HuggingFace..."
                    chineseTTS = try await KokoroTTS.fromHub { progress in
                        DispatchQueue.main.async {
                            status = "Downloading: \(Int(progress.fractionCompleted * 100))%"
                        }
                    }
                    status = "Model loaded!"
                }

                guard let tts = chineseTTS else {
                    status = "Failed to load model"
                    return
                }

                // 2. Generate audio
                status = "Generating Chinese speech..."
                let text = "你好，世界！这是一项测试。"

                var audioChunks: [[Float]] = []

                try tts.generateAudio(voice: .zfXiaoxiao, text: text, speed: 1.0) { chunk in
                    chunk.eval()
                    let samples: [Float]
                    if chunk.shape.count == 1 {
                        samples = chunk.asArray(Float.self)
                    } else {
                        let batch = chunk[0]
                        batch.eval()
                        samples = batch.asArray(Float.self)
                    }
                    audioChunks.append(samples)
                    print("Received chunk: \(samples.count) samples")
                }

                // Wait for async generation
                try await Task.sleep(nanoseconds: 2_000_000_000)

                // 3. Combine and save
                let allSamples = audioChunks.flatMap { $0 }
                if allSamples.isEmpty {
                    status = "No audio generated"
                    return
                }

                let outputPath = saveWAV(samples: allSamples, sampleRate: 24000)
                status = "Done! Saved to: \(outputPath)"
                print("📂 WAV saved: \(outputPath)")

                // Play with afplay
                #if os(macOS)
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
                process.arguments = [outputPath]
                try? process.run()
                #endif

            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }
    }

    private func saveWAV(samples: [Float], sampleRate: Int) -> String {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let path = docs.appendingPathComponent("chinese_tts_output.wav").path

        var data = Data()
        let dataSize = UInt32(samples.count * 2)

        // WAV header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: (36 + dataSize).littleEndian) { Array($0) })
        data.append(contentsOf: "WAVEfmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate * 2).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) })
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        for sample in samples {
            let i16 = Int16(max(-1, min(1, sample)) * 32767)
            data.append(contentsOf: withUnsafeBytes(of: i16.littleEndian) { Array($0) })
        }

        try? data.write(to: URL(fileURLWithPath: path))
        return path
    }
}
