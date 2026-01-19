import Foundation
import XCTest
@testable import MLXAudio

final class ChineseTTSGenerationTest: XCTestCase {

    /// Test Chinese TTS generation using Hub download
    /// This downloads the Chinese model from HuggingFace and generates speech
    func testGenerateChineseWAV() async throws {
        print("\n=== Generating Chinese TTS WAV ===\n")

        // 1. Download Chinese model from HuggingFace
        // This includes: model weights + G2P dictionaries + voice files
        print("[1/3] Downloading Chinese model from HuggingFace...")
        let tts = try await KokoroTTS.fromHub(
            repoId: "FluidInference/kokoro-82m-v1.1-zh-mlx"
        ) { progress in
            print("      Progress: \(Int(progress.fractionCompleted * 100))%")
        }
        print("      ✓ Model loaded!\n")

        // 2. Generate audio
        let text = "你好世界"
        print("[2/3] Generating: \(text)")

        var samples: [Float] = []
        let expectation = XCTestExpectation(description: "Audio generation")

        try tts.generateAudio(voice: .zfXiaoxiao, text: text, speed: 1.0) { chunk in
            chunk.eval()
            if chunk.shape.count == 1 {
                samples.append(contentsOf: chunk.asArray(Float.self))
            } else {
                let batch = chunk[0]
                batch.eval()
                samples.append(contentsOf: batch.asArray(Float.self))
            }
            print("      Chunk: \(samples.count) total samples")
        }

        // Wait for async generation to complete
        try await Task.sleep(nanoseconds: 3_000_000_000) // 3 seconds

        print("      ✓ Generated \(samples.count) samples\n")

        // 3. Save WAV
        print("[3/3] Saving WAV...")
        let path = saveWAV(samples: samples)
        print("      ✓ Saved: \(path)\n")
        print("=== Play with: afplay \(path) ===\n")

        XCTAssertGreaterThan(samples.count, 0, "Should generate audio samples")
    }

    /// Test using KokoroTTSModel directly (simpler API)
    func testChineseTTSWithModel() async throws {
        print("\n=== Testing Chinese TTS with KokoroTTSModel ===\n")

        let model = KokoroTTSModel()

        // Just call say() with a Chinese voice - it will auto-download the model
        print("Calling say() with Chinese voice...")
        model.say("你好世界", .zfXiaoxiao, speed: 1.0, autoPlay: false)

        // Wait for generation
        try await Task.sleep(nanoseconds: 10_000_000_000) // 10 seconds for download + generation

        // Check if audio was saved
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let audioPath = docs.appendingPathComponent("kokoro_output.wav")

        print("Audio path: \(audioPath.path)")
        XCTAssertTrue(FileManager.default.fileExists(atPath: audioPath.path), "Audio file should exist")
    }

    private func saveWAV(samples: [Float]) -> String {
        let path = "/tmp/chinese_tts_output.wav"
        var data = Data()
        let dataSize = UInt32(samples.count * 2)

        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: (36 + dataSize).littleEndian) { Array($0) })
        data.append(contentsOf: "WAVEfmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(24000).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(48000).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) })
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        for s in samples {
            let i16 = Int16(max(-1, min(1, s)) * 32767)
            data.append(contentsOf: withUnsafeBytes(of: i16.littleEndian) { Array($0) })
        }

        try? data.write(to: URL(fileURLWithPath: path))
        return path
    }
}
