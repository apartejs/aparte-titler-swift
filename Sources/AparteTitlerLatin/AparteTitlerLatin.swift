// AparteTitlerLatin — the runtime and the 17-language model, ready to use.
//
//     import AparteTitlerLatin
//
//     let titler = try AparteTitlerLatin.load()
//     titler.title("Peux-tu m'expliquer la photosynthèse chez les plantes ?")
//
// English, French, Spanish, German, Portuguese, Italian, Dutch, Polish, Swedish,
// Danish, Finnish, Czech, Romanian, Norwegian, Hungarian, Croatian, Lithuanian.
// 133 KB, int3, version 1.2.0 of the weights.

import Foundation
@_exported import AparteTitler

public enum AparteTitlerLatin {
    /// The name of the bundled model file.
    public static let modelFile = "titler-v1.2-latin-int3.bin"

    /// Where the bundled model file is.
    public static var modelURL: URL? {
        Bundle.module.url(forResource: "titler-v1.2-latin-int3", withExtension: "bin")
    }

    /// Load the bundled 17-language model. Load it once and keep the `Titler`:
    /// it is immutable and safe to share across threads.
    public static func load() throws -> Titler {
        guard let url = modelURL else { throw TitlerError.malformed("the bundled model \(modelFile) is missing") }
        return try Titler(contentsOf: url)
    }
}
