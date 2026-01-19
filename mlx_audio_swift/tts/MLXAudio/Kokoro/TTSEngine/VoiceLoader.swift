//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Utility class for loading voices
class VoiceLoader {
  private init() {}

  static var availableVoices: [TTSVoice] {
    Array(Constants.voiceFiles.keys)
  }

  static func loadVoice(_ voice: TTSVoice) -> MLXArray {
    let (file, ext) = Constants.voiceFiles[voice]!
    let filePath = Bundle.main.path(forResource: file, ofType: ext)!
      print(filePath)
    return try! read3DArrayFromJson(file: filePath, shape: [510, 1, 256])!
  }

  /// Load a voice from an external directory (e.g., downloaded from Hub)
  /// Supports both JSON and NPY formats
  static func loadVoice(_ voice: TTSVoice, from directory: URL) -> MLXArray? {
    // Get the voice filename
    guard let (baseFilename, _) = Constants.voiceFiles[voice] else {
      return nil
    }

    // Try NPY first (used by Chinese model) - check both root and voices/ subdirectory
    let npyPath = directory.appending(path: "\(baseFilename).npy")
    let npyVoicesPath = directory.appending(path: "voices/\(baseFilename).npy")

    for path in [npyVoicesPath, npyPath] {
      if FileManager.default.fileExists(atPath: path.path) {
        print("[VoiceLoader] Loading NPY: \(path.path)")
        if let array = try? loadNPY(url: path) {
          return array
        }
      }
    }

    // Try JSON (used by English model)
    let jsonPath = directory.appending(path: "\(baseFilename).json")
    if FileManager.default.fileExists(atPath: jsonPath.path) {
      print("[VoiceLoader] Loading JSON: \(jsonPath.path)")
      return try? read3DArrayFromJson(file: jsonPath.path, shape: [510, 1, 256])
    }

    // For Chinese voices, try numbered format (zf_001.npy etc.) in voices/ subdirectory
    if let npyName = Constants.chineseVoiceNPYFiles[voice] {
      let chineseNpyVoicesPath = directory.appending(path: "voices/\(npyName).npy")
      let chineseNpyPath = directory.appending(path: "\(npyName).npy")

      for path in [chineseNpyVoicesPath, chineseNpyPath] {
        if FileManager.default.fileExists(atPath: path.path) {
          print("[VoiceLoader] Loading Chinese NPY: \(path.path)")
          if let array = try? loadNPY(url: path) {
            return array
          }
        }
      }
    }

    return nil
  }

  /// Load NPY file format (NumPy array format)
  private static func loadNPY(url: URL) throws -> MLXArray {
    let data = try Data(contentsOf: url)

    // NPY format:
    // - 6 bytes magic: \x93NUMPY
    // - 1 byte major version
    // - 1 byte minor version
    // - 2 bytes (v1) or 4 bytes (v2+) header length (little endian)
    // - Header (ASCII dict)
    // - Data

    guard data.count > 10 else {
      throw NSError(domain: "VoiceLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "NPY file too small"])
    }

    // Check magic
    let magic = [UInt8](data[0..<6])
    guard magic[0] == 0x93 && magic[1] == 0x4E && magic[2] == 0x55 &&
          magic[3] == 0x4D && magic[4] == 0x50 && magic[5] == 0x59 else {
      throw NSError(domain: "VoiceLoader", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid NPY magic"])
    }

    let majorVersion = data[6]
    let minorVersion = data[7]
    _ = minorVersion // Unused

    let headerLen: Int
    let headerStart: Int
    if majorVersion == 1 {
      headerLen = Int(data[8]) | (Int(data[9]) << 8)
      headerStart = 10
    } else {
      // Version 2+: 4-byte header length
      headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
      headerStart = 12
    }

    let headerEnd = headerStart + headerLen
    guard data.count > headerEnd else {
      throw NSError(domain: "VoiceLoader", code: 3, userInfo: [NSLocalizedDescriptionKey: "NPY header truncated"])
    }

    // Parse header to get shape and dtype
    let headerData = data[headerStart..<headerEnd]
    guard let headerStr = String(data: headerData, encoding: .ascii) else {
      throw NSError(domain: "VoiceLoader", code: 4, userInfo: [NSLocalizedDescriptionKey: "Cannot parse NPY header"])
    }

    // Extract shape from header (e.g., "'shape': (510, 1, 256)")
    let shape = parseNPYShape(headerStr)
    let dtype = parseNPYDtype(headerStr)

    // Read data
    let dataStart = headerEnd
    let rawData = data[dataStart...]

    // Create MLXArray based on dtype
    let array: MLXArray
    if dtype == "<f4" || dtype == "float32" {
      // Float32 little-endian
      let floats = rawData.withUnsafeBytes { ptr -> [Float] in
        let count = ptr.count / MemoryLayout<Float>.size
        return Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: count))
      }
      array = MLXArray(floats).reshaped(shape)
    } else if dtype == "<f8" || dtype == "float64" {
      // Float64 little-endian - convert to Float32
      let doubles = rawData.withUnsafeBytes { ptr -> [Double] in
        let count = ptr.count / MemoryLayout<Double>.size
        return Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Double.self).baseAddress, count: count))
      }
      let floats = doubles.map { Float($0) }
      array = MLXArray(floats).reshaped(shape)
    } else {
      throw NSError(domain: "VoiceLoader", code: 5, userInfo: [NSLocalizedDescriptionKey: "Unsupported NPY dtype: \(dtype)"])
    }

    return array
  }

  /// Parse shape from NPY header string
  private static func parseNPYShape(_ header: String) -> [Int] {
    // Look for 'shape': (x, y, z) or "shape": (x, y, z)
    let pattern = #"'shape'\s*:\s*\(([^)]+)\)"#
    guard let regex = try? NSRegularExpression(pattern: pattern),
          let match = regex.firstMatch(in: header, range: NSRange(header.startIndex..., in: header)),
          let range = Range(match.range(at: 1), in: header) else {
      return [510, 1, 256] // Default voice shape
    }

    let shapeStr = String(header[range])
    let components = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    return components.isEmpty ? [510, 1, 256] : components
  }

  /// Parse dtype from NPY header string
  private static func parseNPYDtype(_ header: String) -> String {
    // Look for 'descr': '<f4' or similar
    let pattern = #"'descr'\s*:\s*'([^']+)'"#
    guard let regex = try? NSRegularExpression(pattern: pattern),
          let match = regex.firstMatch(in: header, range: NSRange(header.startIndex..., in: header)),
          let range = Range(match.range(at: 1), in: header) else {
      return "<f4" // Default to float32
    }
    return String(header[range])
  }

  private static func read3DArrayFromJson(file: String, shape: [Int]) throws -> MLXArray? {
    guard shape.count == 3 else { return nil }

    let data = try Data(contentsOf: URL(fileURLWithPath: file))
    let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])

    var aa = Array(repeating: Float(0.0), count: shape[0] * shape[1] * shape[2])
    var aaIndex = 0

    if let nestedArray = jsonObject as? [[[Any]]] {
      guard nestedArray.count == shape[0] else { return nil }
      for a in 0 ..< nestedArray.count {
        guard nestedArray[a].count == shape[1] else { return nil }
        for b in 0 ..< nestedArray[a].count {
          guard nestedArray[a][b].count == shape[2] else { return nil }
          for c in 0 ..< nestedArray[a][b].count {
            if let n = nestedArray[a][b][c] as? Double {
              aa[aaIndex] = Float(n)
              aaIndex += 1
            } else {
              fatalError("Cannot load value \(a), \(b), \(c) as double")
            }
          }
        }
      }
    } else {
      return nil
    }

    guard aaIndex == shape[0] * shape[1] * shape[2] else {
      fatalError("Mismatch in array size: \(aaIndex) vs \(shape[0] * shape[1] * shape[2])")
    }

    return MLXArray(aa).reshaped(shape)
  }

  public enum Constants {
    /// Mapping from semantic Chinese voice names to numbered NPY files
    /// Based on FluidInference/kokoro-82m-v1.1-zh-mlx voice files
    static let chineseVoiceNPYFiles: [TTSVoice: String] = [
      .zfXiaobei: "zf_001",
      .zfXiaoni: "zf_002",
      .zfXiaoxiao: "zf_003",
      .zfXiaoyi: "zf_004",
      .zmYunjian: "zm_010",
      .zmYunxi: "zm_011",
      .zmYunxia: "zm_012",
      .zmYunyang: "zm_013"
    ]

    static let voiceFiles: [TTSVoice: (String, String)] = [
      .afAlloy: ("af_alloy", "json"),
      .afAoede: ("af_aoede", "json"),
      .afBella: ("af_bella", "json"),
      .afHeart: ("af_heart", "json"),
      .afJessica: ("af_jessica", "json"),
      .afKore: ("af_kore", "json"),
      .afNicole: ("af_nicole", "json"),
      .afNova: ("af_nova", "json"),
      .afRiver: ("af_river", "json"),
      .afSarah: ("af_sarah", "json"),
      .afSky: ("af_sky", "json"),
      .amAdam: ("am_adam", "json"),
      .amEcho: ("am_echo", "json"),
      .amEric: ("am_eric", "json"),
      .amFenrir: ("am_fenrir", "json"),
      .amLiam: ("am_liam", "json"),
      .amMichael: ("am_michael", "json"),
      .amOnyx: ("am_onyx", "json"),
      .amPuck: ("am_puck", "json"),
      .amSanta: ("am_santa", "json"),
      .bfAlice: ("bf_alice", "json"),
      .bfEmma: ("bf_emma", "json"),
      .bfIsabella: ("bf_isabella", "json"),
      .bfLily: ("bf_lily", "json"),
      .bmDaniel: ("bm_daniel", "json"),
      .bmFable: ("bm_fable", "json"),
      .bmGeorge: ("bm_george", "json"),
      .bmLewis: ("bm_lewis", "json"),
      .efDora: ("ef_dora", "json"),
      .emAlex: ("em_alex", "json"),
      .ffSiwis: ("ff_siwis", "json"),
      .hfAlpha: ("hf_alpha", "json"),
      .hfBeta: ("hf_beta", "json"),
      .hfOmega: ("hm_omega", "json"),
      .hmPsi: ("hm_psi", "json"),
      .ifSara: ("if_sara", "json"),
      .imNicola: ("im_nicola", "json"),
      .jfAlpha: ("jf_alpha", "json"),
      .jfGongitsune: ("jf_gongitsune", "json"),
      .jfNezumi: ("jf_nezumi", "json"),
      .jfTebukuro: ("jf_tebukuro", "json"),
      .jmKumo: ("jm_kumo", "json"),
      .pfDora: ("pf_dora", "json"),
      .pmSanta: ("pm_santa", "json"),
      .zfXiaobei: ("zf_xiaobei", "json"),
      .zfXiaoni: ("zf_xiaoni", "json"),
      .zfXiaoxiao: ("zf_xiaoxiao", "json"),
      .zfXiaoyi: ("zf_xiaoyi", "json"),
      .zmYunjian: ("zm_yunjian", "json"),
      .zmYunxi: ("zm_yunxi", "json"),
      .zmYunxia: ("zm_yunxia", "json"),
      .zmYunyang: ("zm_yunyang", "json")
    ]
  }
}

// Extension to add utility methods to TTSVoice
extension TTSVoice {
  static func fromIdentifier(_ identifier: String) -> TTSVoice? {
    let reverseMapping = Dictionary(
      VoiceLoader.Constants.voiceFiles.map { (voice, fileInfo) in
        (fileInfo.0, voice)
      },
      uniquingKeysWith: { first, _ in first } // In case of duplicates, keep the first one
    )
    return reverseMapping[identifier]
  }
}
