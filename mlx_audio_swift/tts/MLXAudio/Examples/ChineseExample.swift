import SwiftUI

/// Example showing simplified Chinese TTS usage
/// With the auto-download fix, you no longer need to call initializeChineseSupport()
/// The Chinese model is automatically downloaded from HuggingFace Hub when a Chinese voice is used
struct ChineseExampleView: View {
    @StateObject private var kokoroTTSModel = KokoroTTSModel()
    @State private var status: String = "Ready"

    var body: some View {
        VStack(spacing: 20) {
            Text("Chinese TTS Example")
                .font(.title)

            Text(status)
                .foregroundColor(.secondary)

            // Show Chinese model loading progress
            if let progress = kokoroTTSModel.chineseModelLoadingProgress {
                Text(progress)
                    .foregroundColor(.blue)
            }

            Button("Test Chinese TTS") {
                runChineseTest()
            }
            .disabled(kokoroTTSModel.generationInProgress)
        }
        .padding()
    }

    /// Simplified Chinese TTS test - no initializeChineseSupport() needed!
    /// The Chinese model with G2P dictionaries is auto-downloaded from HuggingFace Hub
    private func runChineseTest() {
        status = "Generating Chinese speech..."

        // Just call say() with a Chinese voice - the model is auto-downloaded
        kokoroTTSModel.say(
            "你好，世界！这是一项测试。",
            .zfXiaoxiao,  // Chinese female voice
            speed: 1.0,
            autoPlay: true
        )

        // The chineseModelLoadingProgress property shows download progress
        // Once downloaded, audio generation starts automatically
    }

    /// Alternative test with different Chinese voice
    private func runChineseTestMale() {
        status = "Generating Chinese speech (male)..."

        kokoroTTSModel.say(
            "大家好，我是中文语音助手。",
            .zmYunxi,  // Chinese male voice
            speed: 1.0,
            autoPlay: true
        )
    }
}

// MARK: - Legacy Approach (No longer needed but still supported)

/// This is @shuhongwu's original approach using bundled G2P files
/// It still works but is NO LONGER NECESSARY because:
/// 1. The Chinese model now auto-downloads from HuggingFace Hub
/// 2. The downloaded model includes its own G2P dictionaries
/// 3. initializeChineseSupport() loads G2P into the ENGLISH engine, not the Chinese engine
///
/// Keeping this for reference only - use the simplified approach above
struct LegacyChineseExampleView: View {
    @StateObject private var kokoroTTSModel = KokoroTTSModel()
    @State private var status: String = "Ready"

    var body: some View {
        VStack(spacing: 20) {
            Text("Legacy Chinese TTS (Not Recommended)")
                .font(.title)

            Text(status)
                .foregroundColor(.secondary)

            Button("Test Chinese TTS (Legacy)") {
                runChineseTestLegacy()
            }
            .disabled(kokoroTTSModel.generationInProgress)
        }
        .padding()
    }

    /// Legacy approach - initializeChineseSupport() is now redundant
    /// The call is harmless but unnecessary - it loads G2P into the wrong engine
    private func runChineseTestLegacy() {
        Task {
            // NOTE: This G2P initialization is now UNNECESSARY
            // The Chinese model auto-downloads with its own G2P dictionaries
            // This code is kept only for backward compatibility

            guard let jiebaURL = Bundle.main.url(forResource: "jieba.bin", withExtension: "gz"),
                  let pinyinSingleURL = Bundle.main.url(forResource: "pinyin_single.bin", withExtension: "gz") else {
                status = "Error: Chinese dictionary files not found in Bundle."
                return
            }
            let pinyinPhrasesURL = Bundle.main.url(forResource: "pinyin_phrases.bin", withExtension: "gz")

            do {
                // This loads G2P into the ENGLISH engine (kokoroTTSEngine)
                // But Chinese voices now use a SEPARATE Chinese engine (chineseKokoroTTSEngine)
                // So this call has NO EFFECT on Chinese TTS
                try kokoroTTSModel.initializeChineseSupport(
                    jiebaURL: jiebaURL,
                    pinyinSingleURL: pinyinSingleURL,
                    pinyinPhrasesURL: pinyinPhrasesURL
                )
            } catch {
                status = "Failed to init Chinese G2P: \(error.localizedDescription)"
                return
            }

            status = "Generating Chinese..."

            // This is what actually triggers Chinese model download!
            // The say() method detects the Chinese voice and auto-downloads the model
            kokoroTTSModel.say("你好，世界！这是一项测试。", .zfXiaoxiao, speed: 1.0, autoPlay: true)
        }
    }
}
