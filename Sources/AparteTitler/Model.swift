import Foundation

// The forward pass, written as the same arithmetic as the JavaScript runtime,
// step for step: JavaScript computes in Double and rounds to Float32 each time it
// stores into a Float32Array, so this code accumulates in Double and rounds to
// Float at exactly the same places. Logits are not guaranteed identical to the
// bit — `exp` differs between V8 and each platform's math library — but titles
// are: the tests check 1,496 messages in four precisions.
//
// Measured on 16 October 2026: accumulating in Float instead changes none of the
// 2,992 int3 reference titles, so the titles are robust to that noise and the
// Double path is a precaution, not a necessity. A logic error is another matter:
// dropping the tokenizer's `\s+(?!\S)` rule alone changes 246 of them.

/// out[t] = layer norm of x[t], over D values per token.
private func layerNorm(_ x: [Float], _ L: Int, _ D: Int, _ g: [Float], _ b: [Float], _ out: inout [Float]) {
    x.withUnsafeBufferPointer { xp in
        out.withUnsafeMutableBufferPointer { op in
            for t in 0..<L {
                let r = t * D
                var mean = 0.0
                for d in 0..<D { mean += Double(xp[r + d]) }
                mean /= Double(D)
                var v = 0.0
                for d in 0..<D { let e = Double(xp[r + d]) - mean; v += e * e }
                v /= Double(D)
                let inv = 1 / (v + 1e-5).squareRoot()
                for d in 0..<D { op[r + d] = Float((Double(xp[r + d]) - mean) * inv * Double(g[d]) + Double(b[d])) }
            }
        }
    }
}

/// Abramowitz-Stegun 7.1.26, the same approximation as the JavaScript runtime.
@inline(__always)
private func erf(_ x0: Double) -> Double {
    let s: Double = x0 > 0 ? 1 : (x0 < 0 ? -1 : 0)
    let x = abs(x0)
    let t = 1 / (1 + 0.3275911 * x)
    let y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x)
    return s * y
}

private let sqrt2 = 2.0.squareRoot()

@inline(__always)
private func gelu(_ x: Double) -> Double { 0.5 * x * (1 + erf(x / sqrt2)) }

/// y[L x N] = x[L x K] · W[N x K]ᵀ + b[N], with the bias added first, as in JavaScript.
private func linear(_ x: [Float], _ L: Int, _ K: Int, _ W: [Float], _ b: [Float]?, _ N: Int, _ out: inout [Float]) {
    x.withUnsafeBufferPointer { xp in
        W.withUnsafeBufferPointer { wp in
            out.withUnsafeMutableBufferPointer { op in
                for t in 0..<L {
                    let xr = t * K
                    for n in 0..<N {
                        var s: Double = b.map { Double($0[n]) } ?? 0
                        let wr = n * K
                        for k in 0..<K { s += Double(xp[xr + k]) * Double(wp[wr + k]) }
                        op[t * N + n] = Float(s)
                    }
                }
            }
        }
    }
}

struct Model: Sendable {
    let w: [String: [Float]]
    let D: Int, H: Int, embDim: Int, layers: Int, maxPos: Int, sinus: Bool

    init(header: ModelHeader, weights: [String: [Float]]) throws {
        w = weights
        D = header.dim; H = header.heads; layers = header.layers; maxPos = header.maxPos
        embDim = header.dimEmb ?? 0
        sinus = header.positions == "sinus"
        guard D > 0, H > 0, D % H == 0, maxPos > 0 else { throw TitlerError.malformed("inconsistent dimensions") }
        var needed = ["emb_tok.weight", "norm.weight", "norm.bias", "head.weight", "head.bias"]
        if embDim > 0 { needed.append("proj_emb.weight") }
        if !sinus { needed.append("emb_pos.weight") }
        for c in 0..<layers {
            for part in ["n1", "att.qkv", "att.output", "n2", "ff.net.0", "ff.net.2"] {
                needed += ["blocks.\(c).\(part).weight", "blocks.\(c).\(part).bias"]
            }
        }
        for name in needed where weights[name] == nil { throw TitlerError.malformed("missing tensor \(name)") }
    }

    /// One logit per token: positive means keep.
    func logits(_ ids: [Int]) -> [Float] {
        let L = ids.count, Dh = D / H
        var x = [Float](repeating: 0, count: L * D)
        let embTok = w["emb_tok.weight"]!
        if embDim > 0 {                          // factorized embeddings (ALBERT style)
            var e = [Float](repeating: 0, count: L * embDim)
            for t in 0..<L { for d in 0..<embDim { e[t * embDim + d] = embTok[ids[t] * embDim + d] } }
            linear(e, L, embDim, w["proj_emb.weight"]!, nil, D, &x)
        } else {
            for t in 0..<L { for d in 0..<D { x[t * D + d] = embTok[ids[t] * D + d] } }
        }
        let embPos = sinus ? [] : w["emb_pos.weight"]!
        for t in 0..<L {
            let pos = min(t, maxPos - 1)
            if sinus {
                for d in stride(from: 0, to: D, by: 2) {
                    let f = exp(Double(d) * (-log(10000.0) / Double(D)))
                    x[t * D + d] = Float(Double(x[t * D + d]) + sin(Double(pos) * f))
                    x[t * D + d + 1] = Float(Double(x[t * D + d + 1]) + cos(Double(pos) * f))
                }
            } else {
                for d in 0..<D { x[t * D + d] = Float(Double(x[t * D + d]) + Double(embPos[pos * D + d])) }
            }
        }

        var y = [Float](repeating: 0, count: L * D), qkv = [Float](repeating: 0, count: L * 3 * D)
        var att = [Float](repeating: 0, count: L * D), f1 = [Float](repeating: 0, count: L * 4 * D)
        var f2 = [Float](repeating: 0, count: L * D), s = [Float](repeating: 0, count: L)
        let scale = Double(Dh).squareRoot()
        for c in 0..<layers {
            let k = "blocks.\(c)."
            layerNorm(x, L, D, w[k + "n1.weight"]!, w[k + "n1.bias"]!, &y)
            linear(y, L, D, w[k + "att.qkv.weight"]!, w[k + "att.qkv.bias"]!, 3 * D, &qkv)
            for i in 0..<(L * D) { att[i] = 0 }
            for head in 0..<H {                  // every token attends to every token (encoder: no causal mask)
                let o = head * Dh
                for i in 0..<L {
                    var mx = -Double.infinity
                    for j in 0..<L {
                        var d = 0.0
                        for e in 0..<Dh { d += Double(qkv[i * 3 * D + o + e]) * Double(qkv[j * 3 * D + D + o + e]) }
                        s[j] = Float(d / scale)
                        if Double(s[j]) > mx { mx = Double(s[j]) }
                    }
                    var sum = 0.0
                    for j in 0..<L { s[j] = Float(exp(Double(s[j]) - mx)); sum += Double(s[j]) }
                    for j in 0..<L {
                        let p = Double(s[j]) / sum
                        for e in 0..<Dh {
                            att[i * D + o + e] = Float(Double(att[i * D + o + e]) + p * Double(qkv[j * 3 * D + 2 * D + o + e]))
                        }
                    }
                }
            }
            linear(att, L, D, w[k + "att.output.weight"]!, w[k + "att.output.bias"]!, D, &f2)
            for i in 0..<(L * D) { x[i] = Float(Double(x[i]) + Double(f2[i])) }
            layerNorm(x, L, D, w[k + "n2.weight"]!, w[k + "n2.bias"]!, &y)
            linear(y, L, D, w[k + "ff.net.0.weight"]!, w[k + "ff.net.0.bias"]!, 4 * D, &f1)
            for i in 0..<(L * 4 * D) { f1[i] = Float(gelu(Double(f1[i]))) }
            linear(f1, L, 4 * D, w[k + "ff.net.2.weight"]!, w[k + "ff.net.2.bias"]!, D, &f2)
            for i in 0..<(L * D) { x[i] = Float(Double(x[i]) + Double(f2[i])) }
        }
        layerNorm(x, L, D, w["norm.weight"]!, w["norm.bias"]!, &y)
        var z = [Float](repeating: 0, count: L)
        linear(y, L, D, w["head.weight"]!, w["head.bias"]!, 1, &z)
        return z
    }
}
