// AparteTitler — a conversation title from its first message, on-device.
//
// One message in, a title of 3 to 6 words out — words copied verbatim from the
// message, never invented. Reads the `.bin` model files published at
// https://huggingface.co/apartejs/aparte-titler (format 1). No dependency.
//
//     let titler = try Titler(contentsOf: modelURL)
//     titler.title("Write me a cover letter for a junior data analyst position at a bank")
//     // "letter junior data analyst position bank"
//
// It gives exactly the titles of the JavaScript runtime (@aparte/titler on npm)
// and of the Python reference implementation; the tests check it on 1,496 messages.

import Foundation

/// Characters stripped from the edges of a word only: "-Don't" → "Don't", while
/// "rendez-vous" and "c++" are left whole.
private let punctuation: Set<Unicode.Scalar> = Set(".,;:!?\"'()[]{}<>*`“”‘’«»…-–—".unicodeScalars)

/// A hyphen or an apostrophe glued between two letters keeps a word whole
/// ("Pourrais-tu", "l'école"): a title must never start with a fragment like "-tu".
private let joiners: Set<Unicode.Scalar> = ["-", "'", "’"]

/// Strip Python whitespace from both ends.
private func trimSpaces(_ s: ArraySlice<Unicode.Scalar>) -> ArraySlice<Unicode.Scalar> {
    var a = s.startIndex, b = s.endIndex
    while a < b, isPythonSpace(s[a]) { a += 1 }
    while b > a, isPythonSpace(s[b - 1]) { b -= 1 }
    return s[a..<b]
}

private func trimPunctuation(_ s: ArraySlice<Unicode.Scalar>) -> String {
    var a = s.startIndex, b = s.endIndex
    while a < b, punctuation.contains(s[a]) { a += 1 }
    while b > a, punctuation.contains(s[b - 1]) { b -= 1 }
    var out = String.UnicodeScalarView()
    out.append(contentsOf: s[a..<b])
    return String(out)
}

/// A new word starts on a token whose first character is a space or not a letter or
/// digit — except a joiner glued between two letters, which keeps the word whole.
private func wordIds(_ text: [Unicode.Scalar], _ offsets: [Range<Int>]) -> [Int] {
    var out: [Int] = [], num = -1, previousEnd = -1
    for (i, r) in offsets.enumerated() {
        let a = r.lowerBound
        var starts = true
        if !r.isEmpty {
            let c = text[a]
            starts = i == 0 || isPythonSpace(c) || !isWordChar(c)
            if starts, i > 0, joiners.contains(c), a == previousEnd, a > 0, isWordChar(text[a - 1]),
               a + 1 < text.count, isWordChar(text[a + 1]) {
                starts = false
            }
        }
        if starts { num += 1 }
        out.append(num)
        previousEnd = r.upperBound
    }
    return out
}

/// A word of the message and how strongly the model wants it in the title.
public struct ScoredWord: Sendable, Equatable {
    /// The word as it appears in the message, edge punctuation removed.
    public let word: String
    /// The model's probability that this word belongs in the title, from 0 to 1.
    public let score: Double
    /// Where the word is, in Unicode scalars of the message after trimming and truncation.
    public let range: Range<Int>
}

/// The titler: load a model file once, then title as many messages as you like.
/// It holds no mutable state and can be shared across threads.
public final class Titler: Sendable {
    /// The model file's header: version, scope, languages, precision, dimensions.
    public let header: ModelHeader
    let tokenizer: Tokenizer
    let model: Model
    let threshold: Double, minWords: Int, maxWords: Int, keepAllUpTo: Int

    /// Load a model from the contents of a `.bin` file.
    public init(data: Data) throws {
        let (header, merges, weights) = try readModel(data)
        self.header = header
        tokenizer = Tokenizer(header: header, merges: merges)
        model = try Model(header: header, weights: weights)
        // a 1.0 file carries no `decoding` block and decodes with exactly these values
        let d = header.decoding
        threshold = d?.threshold ?? 0.5
        minWords = d?.minWords ?? 3
        maxWords = d?.maxWords ?? 6
        keepAllUpTo = d?.keepAllUpTo ?? 4
    }

    /// Load a model from a `.bin` file on disk or in a bundle.
    public convenience init(contentsOf url: URL) throws {
        try self.init(data: try Data(contentsOf: url))
    }

    /// Every word of the message with its score, in text order.
    public func words(_ message: String) -> [ScoredWord] {
        // Python's message.strip()[:maxChars]: Python's whitespace, cut in characters
        let all = Array(message.unicodeScalars)
        let text = Array(trimSpaces(all[...]).prefix(header.maxChars))
        let (ids, offsets) = tokenizer.encode(text)
        if ids.isEmpty { return [] }
        let z = model.logits(ids)
        var groups: [[Int]] = []
        for (i, wid) in wordIds(text, offsets).enumerated() {
            if wid == groups.count { groups.append([i]) } else { groups[wid].append(i) }
        }
        var words: [ScoredWord] = []
        for tokens in groups {
            let range = offsets[tokens[0]].lowerBound..<offsets[tokens[tokens.count - 1]].upperBound
            let slice = text[range]
            guard slice.contains(where: isWordChar) else { continue }  // punctuation-only groups are never a title word
            var sum = 0.0
            for i in tokens { sum += Double(z[i]) }                    // the decision is per word: mean of its token logits
            let score = 1 / (1 + exp(-sum / Double(tokens.count)))
            words.append(ScoredWord(word: trimPunctuation(trimSpaces(slice)), score: score, range: range))
        }
        return words
    }

    /// The title of a conversation, from its first message.
    ///
    /// By default the hybrid decoding: among the six best-scored words, those the
    /// model is at least half sure of, never fewer than three, and every word of a
    /// message that has four or fewer.
    ///
    /// - Parameters:
    ///   - message: the user's first message — not a system prompt, not the history.
    ///   - budget: keep exactly this many best-scored words instead, the decoding
    ///     version 1.0 shipped.
    ///   - stopWords: words never to keep (case-insensitive), as long as at least
    ///     three words remain.
    /// - Returns: the title, words in message order, or an empty string when the
    ///   message has no word at all.
    public func title(_ message: String, budget: Int? = nil, stopWords: Set<String> = []) -> String {
        var words = self.words(message)
        if words.isEmpty { return "" }
        if !stopWords.isEmpty {
            let banned = Set(stopWords.map { $0.lowercased() })
            let kept = words.filter { !banned.contains($0.word.lowercased()) }
            if kept.count >= minWords { words = kept }
        }
        // stable: ties keep text order
        let byScore = words.enumerated()
            .sorted { $0.element.score != $1.element.score ? $0.element.score > $1.element.score : $0.offset < $1.offset }
            .map { $0.element }
        var chosen: [ScoredWord]
        if let budget = budget {
            chosen = Array(byScore.prefix(max(0, budget)))
        } else if words.count <= keepAllUpTo {
            chosen = words
        } else {
            let top = Array(byScore.prefix(maxWords))
            chosen = top.filter { $0.score >= threshold }
            if chosen.count < minWords { chosen = Array(top.prefix(minWords)) }
        }
        return chosen.sorted { $0.range.lowerBound < $1.range.lowerBound }
            .map(\.word).filter { !$0.isEmpty }.joined(separator: " ")
    }
}
