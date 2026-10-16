// Equality test: the Swift runtime must give exactly the titles of the Python
// reference implementation and of the JavaScript runtime, message for message,
// on 1,496 messages in five languages (French, English, Spanish, Polish, Finnish).
//
// int3 runs everywhere. The three other precisions run when APARTE_TITLER_ROOT
// points at a checkout of the aparte-titler repositories (the model files are on
// Hugging Face, not in this package), and are reported as skipped otherwise.
//
//     swift test -c release
//     APARTE_TITLER_ROOT=../aparte-titler swift test -c release

import Foundation
import XCTest
import AparteTitler
import AparteTitlerLatin

private struct Reference: Decodable {
    let text: String
    let title: String
    let title_budget6: String?
}

private func references(_ url: URL) throws -> [Reference] {
    let text = try String(contentsOf: url, encoding: .utf8)
    return try text.split(whereSeparator: \.isNewline).map {
        try JSONDecoder().decode(Reference.self, from: Data($0.utf8))
    }
}

/// Every title that differs from the reference, hybrid and budget 6.
private func mismatches(_ titler: Titler, _ refs: [Reference]) -> [String] {
    var out: [String] = []
    for r in refs {
        let mine = titler.title(r.text)
        if mine != r.title { out.append("hybrid  expected «\(r.title)» got «\(mine)» — \(r.text.prefix(60))") }
        if let b6 = r.title_budget6 {
            let mine6 = titler.title(r.text, budget: 6)
            if mine6 != b6 { out.append("budget6 expected «\(b6)» got «\(mine6)» — \(r.text.prefix(60))") }
        }
    }
    return out
}

final class ParityTests: XCTestCase {
    func testReadsTheBundledModelHeader() throws {
        let titler = try AparteTitlerLatin.load()
        XCTAssertEqual(titler.header.version, "1.2.0")
        XCTAssertEqual(titler.header.model, "latin")
        XCTAssertEqual(titler.header.precision, "int3")
        XCTAssertEqual(titler.header.vocab, 12288)
        XCTAssertEqual(titler.header.languages?.count, 17)
    }

    func testGivesTheReferenceTitlesAtInt3() throws {
        let titler = try AparteTitlerLatin.load()
        let url = try XCTUnwrap(Bundle.module.url(forResource: "latin-int3", withExtension: "jsonl"))
        let refs = try references(url)
        XCTAssertEqual(refs.count, 1496)
        let bad = mismatches(titler, refs)
        XCTAssertTrue(bad.isEmpty, "\(bad.count) of \(refs.count * 2) titles differ at int3:\n" + bad.prefix(10).joined(separator: "\n"))
    }

    func testGivesTheReferenceTitlesAtTheOtherPrecisions() throws {
        guard let root = ProcessInfo.processInfo.environment["APARTE_TITLER_ROOT"] else {
            throw XCTSkip("set APARTE_TITLER_ROOT to the aparte-titler checkout to test fp32, int8 and int4")
        }
        let base = URL(fileURLWithPath: root)
        for precision in ["fp32", "int8", "int4"] {
            let bin = base.appendingPathComponent("hf-model/v1.2/titler-v1.2-latin-\(precision).bin")
            let refs = base.appendingPathComponent("model/tests/references/latin-\(precision).jsonl")
            guard FileManager.default.fileExists(atPath: bin.path), FileManager.default.fileExists(atPath: refs.path) else {
                XCTFail("\(precision): model or references not found under \(root)"); continue
            }
            let bad = mismatches(try Titler(contentsOf: bin), try references(refs))
            XCTAssertTrue(bad.isEmpty, "\(bad.count) titles differ at \(precision):\n" + bad.prefix(10).joined(separator: "\n"))
        }
    }

    func testAnEmptyOrPunctuationOnlyMessageGivesAnEmptyTitle() throws {
        let titler = try AparteTitlerLatin.load()
        XCTAssertEqual(titler.title(""), "")
        XCTAssertEqual(titler.title("   ...  !!  "), "")
    }

    func testStopWordsNeverLeaveFewerThanThreeWords() throws {
        let titler = try AparteTitlerLatin.load()
        let message = "Peux-tu m'expliquer la photosynthèse chez les plantes ?"
        let words = titler.title(message).split(separator: " ").map(String.init)
        let filtered = titler.title(message, stopWords: Set(words))
        XCTAssertGreaterThanOrEqual(filtered.split(separator: " ").count, 3)
    }

    func testRejectsAFileThatIsNotAModel() {
        XCTAssertThrowsError(try Titler(data: Data("not a model".utf8)))
    }
}
