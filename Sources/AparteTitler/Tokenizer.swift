// Byte-level BPE, as GPT-2 and the Hugging Face `tokenizers` ByteLevel model do it,
// reproducing the JavaScript runtime exactly. Positions are counted in Unicode
// scalars of the prepared text: the JavaScript runtime counts UTF-16 units, but
// every slice it takes falls on the same characters, so titles are identical.

// MARK: - character classes

/// `\p{L}`: a letter of any script.
@inline(__always)
func isLetter(_ s: Unicode.Scalar) -> Bool {
    switch s.properties.generalCategory {
    case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
    default: return false
    }
}

/// `\p{N}`: a number of any kind.
@inline(__always)
func isNumber(_ s: Unicode.Scalar) -> Bool {
    switch s.properties.generalCategory {
    case .decimalNumber, .letterNumber, .otherNumber: return true
    default: return false
    }
}

@inline(__always)
func isWordChar(_ s: Unicode.Scalar) -> Bool { isLetter(s) || isNumber(s) }

/// JavaScript's `\s` (WhiteSpace and LineTerminator), used by the pre-tokenizer only.
@inline(__always)
func isJSSpace(_ s: Unicode.Scalar) -> Bool {
    switch s.value {
    case 0x09...0x0D, 0x20, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF: return true
    default: return false
    }
}

/// Python's `str.isspace()`, used to trim a message and its words. It differs from
/// JavaScript's `\s`: it adds U+001C–U+001F and U+0085, and lacks U+FEFF.
@inline(__always)
func isPythonSpace(_ s: Unicode.Scalar) -> Bool {
    switch s.value {
    case 0x09...0x0D, 0x1C...0x1F, 0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
        return true
    default: return false
    }
}

// MARK: - pre-tokenization

/// GPT-2's split, the JavaScript regex
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
/// written out as a scanner. At each position the alternatives are tried in that
/// order and the first that matches wins, as a regex alternation does.
func pretokenize(_ s: [Unicode.Scalar]) -> [Range<Int>] {
    var pieces: [Range<Int>] = []
    let n = s.count
    let apostrophe: Unicode.Scalar = "'", space: Unicode.Scalar = " "
    let contractions: [[Unicode.Scalar]] = [["s"], ["t"], ["r", "e"], ["v", "e"], ["m"], ["l", "l"], ["d"]]

    // ` ?X+`: an optional ASCII space, then at least one character of the class
    func run(_ i: Int, _ inClass: (Unicode.Scalar) -> Bool) -> Int? {
        var j = i
        if s[j] == space, j + 1 < n, inClass(s[j + 1]) { j += 1 }
        guard inClass(s[j]) else { return nil }
        while j < n, inClass(s[j]) { j += 1 }
        return j
    }

    var i = 0
    outer: while i < n {
        if s[i] == apostrophe {
            for c in contractions where i + 1 + c.count <= n && Array(s[(i + 1)..<(i + 1 + c.count)]) == c {
                pieces.append(i..<(i + 1 + c.count)); i += 1 + c.count; continue outer
            }
        }
        if let j = run(i, isLetter) { pieces.append(i..<j); i = j; continue }
        if let j = run(i, isNumber) { pieces.append(i..<j); i = j; continue }
        if let j = run(i, { !isJSSpace($0) && !isLetter($0) && !isNumber($0) }) { pieces.append(i..<j); i = j; continue }
        // whitespace: `\s+(?!\S)` keeps the last space for the word that follows
        // when the run is longer than one; otherwise `\s+` takes the whole run.
        var j = i
        while j < n, isJSSpace(s[j]) { j += 1 }
        let length = j - i
        if j < n, length >= 2 { j -= 1 }
        pieces.append(i..<j)
        i = j
    }
    return pieces
}

// MARK: - BPE

@inline(__always)
func utf8(_ v: UInt32) -> [UInt8] {
    switch v {
    case 0..<0x80: return [UInt8(v)]
    case 0x80..<0x800: return [UInt8(0xC0 | v >> 6), UInt8(0x80 | v & 0x3F)]
    case 0x800..<0x10000: return [UInt8(0xE0 | v >> 12), UInt8(0x80 | (v >> 6) & 0x3F), UInt8(0x80 | v & 0x3F)]
    default: return [UInt8(0xF0 | v >> 18), UInt8(0x80 | (v >> 12) & 0x3F), UInt8(0x80 | (v >> 6) & 0x3F), UInt8(0x80 | v & 0x3F)]
    }
}

struct Tokenizer: Sendable {
    let byteIds: [Int]
    let base: Int
    let rank: [Int: Int]

    init(header: ModelHeader, merges: [UInt16]) {
        byteIds = header.byteIds
        base = header.mergesBase
        var r: [Int: Int] = [:]
        r.reserveCapacity(merges.count / 2)
        for k in 0..<(merges.count / 2) { r[Int(merges[2 * k]) * 65536 + Int(merges[2 * k + 1])] = k }
        rank = r
    }

    /// Token ids and, for each, the half-open range of characters it covers.
    func encode(_ text: [Unicode.Scalar]) -> (ids: [Int], offsets: [Range<Int>]) {
        var ids: [Int] = [], offsets: [Range<Int>] = []
        for piece in pretokenize(text) {
            var tokens: [Int] = [], lengths: [Int] = [], charOfByte: [Int] = []
            for k in piece {
                for b in utf8(text[k].value) {
                    tokens.append(byteIds[Int(b)]); lengths.append(1); charOfByte.append(k)
                }
            }
            while tokens.count > 1 {                 // merge the lowest-ranked pair first
                var best = -1, at = -1
                for j in 0..<(tokens.count - 1) {
                    if let r = rank[tokens[j] * 65536 + tokens[j + 1]], best < 0 || r < best { best = r; at = j }
                }
                if best < 0 { break }
                tokens[at] = base + best
                lengths[at] += lengths[at + 1]
                tokens.remove(at: at + 1); lengths.remove(at: at + 1)
            }
            var pos = 0
            for (id, length) in zip(tokens, lengths) {
                ids.append(id)
                offsets.append(charOfByte[pos]..<(charOfByte[pos + length - 1] + 1))
                pos += length
            }
        }
        return (ids, offsets)
    }
}
