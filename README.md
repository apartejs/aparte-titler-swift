# AparteTitler for Swift

[![test](https://github.com/apartejs/aparte-titler-swift/actions/workflows/test.yml/badge.svg)](https://github.com/apartejs/aparte-titler-swift/actions/workflows/test.yml)

**Name a chat conversation on the device — no API call, no network, 133 KB.**

Give it the first message of a conversation, get back a short title made of words from that message:

```swift
import AparteTitlerLatin

let titler = try AparteTitlerLatin.load()
titler.title("Write me a cover letter for a junior data analyst position at a bank")
// "letter junior data analyst position bank"
```

Built for chat apps, note-taking and voice-memo apps, and on-device LLM clients that need a conversation title without spending a request on it.

- **On-device and private** — the text never leaves the phone.
- **Works offline**, with no server and no key to manage.
- **Small** — 133 KB for 17 languages, or 41 KB for one.
- **Fast** — about 7 ms per title on one desktop CPU core, no GPU, no Core ML.
- **Safe from prompt injection by construction** — it only ever copies words from the message, so an instruction hidden in a message can at worst produce a poor title, never be followed.
- **Zero dependencies** — pure Swift and Foundation.

**17 languages**: English, French, Spanish, German, Portuguese, Italian, Dutch, Polish, Swedish, Danish, Finnish, Czech, Romanian, Norwegian, Hungarian, Croatian, Lithuanian.

- Presentation and live demo: **[apartejs.dev/models/titler](https://apartejs.dev/models/titler/)**
- Models, scores and benchmark: [huggingface.co/apartejs/aparte-titler](https://huggingface.co/apartejs/aparte-titler)
- Also for Android and the JVM: [aparte-titler-kotlin](https://github.com/apartejs/aparte-titler-kotlin) (`dev.apartejs:titler-latin` on Maven Central), and for JavaScript: [`@aparte/titler`](https://www.npmjs.com/package/@aparte/titler) on npm

## Install

Swift Package Manager. In Xcode: *File → Add Package Dependencies…* and paste the repository URL. In `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/apartejs/aparte-titler-swift.git", from: "1.2.0"),
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "AparteTitlerLatin", package: "aparte-titler-swift"),
    ]),
]
```

Two products:

| product | what it contains |
|---|---|
| `AparteTitlerLatin` | the runtime and the 17-language model (133 KB), ready to use |
| `AparteTitler` | the runtime alone — bring any model file from [Hugging Face](https://huggingface.co/apartejs/aparte-titler) |

iOS 13, macOS 10.15, tvOS 13, watchOS 6, visionOS 1, and Linux and Windows.

## Use

Load the model once and keep the `Titler`. It is immutable and safe to share across threads.

```swift
import AparteTitlerLatin

let titler = try AparteTitlerLatin.load()

titler.title("Peux-tu m'expliquer la photosynthèse chez les plantes ?")
titler.title("Kannst du mir helfen, einen Geburtstagsbrief für meine Oma zu schreiben?")
```

Give it **only the user's first message** — never a system prompt or the conversation history.

**One language, smaller.** Each language also has its own 41 KB model. Download `v1.2/titler-v1.2-<lang>-int3.bin` from [Hugging Face](https://huggingface.co/apartejs/aparte-titler/tree/main/v1.2), add it to your app's bundle, and load it with the runtime alone:

```swift
import AparteTitler

let url = Bundle.main.url(forResource: "titler-v1.2-fr-int3", withExtension: "bin")!
let titler = try Titler(contentsOf: url)
```

**Options.**

```swift
titler.title(message, budget: 6)              // exactly the six best words, as version 1.0 did
titler.title(message, stopWords: ["salut"])   // words never to keep, as long as three remain
titler.words(message)                         // every word with its score from 0 to 1
```

By default the title keeps, among the six best-scored words, those the model is at least half sure of — never fewer than three — and every word of a message that has four or fewer.

## How good is it

On a benchmark of real chat messages in 17 languages, the 17-language model scores nearly twice a naive baseline (the first five words of the message), and on the six languages where the comparison can be made fairly it reaches 94 % of the 2.4 GB language model it was distilled from — a model 18,000 times heavier. The full tables, the benchmark and its limits are on the [model card](https://huggingface.co/apartejs/aparte-titler).

## Same titles everywhere

This package gives **exactly** the titles of the JavaScript runtime, of the Kotlin library and of the Python reference implementation — not approximately: the tests run 1,496 messages in five languages, in both decodings, and compare every title character for character, at the four precisions of the model files (int3, int4, int8, fp32). A title computed on the server in Python, in the browser in JavaScript, on an iPhone in Swift or on an Android phone in Kotlin is the same title.

```bash
swift test -c release
APARTE_TITLER_ROOT=../aparte-titler swift test -c release   # also int4, int8 and fp32
```

## License

MIT © 2026 Paul Richez. The model was distilled from titles written by [gemma 4 e2b](https://huggingface.co/google/gemma-4-E2B-it); the training data and its licences are described in [the dataset](https://huggingface.co/datasets/apartejs/aparte-titler-data).
