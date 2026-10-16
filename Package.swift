// swift-tools-version: 5.9
//
// aparte-titler for Swift: a conversation title from its first message,
// on-device, offline, in a few milliseconds. Same model files and same titles
// as the JavaScript runtime (@aparte/titler on npm) and the Python reference.

import PackageDescription

let package = Package(
    name: "AparteTitler",
    platforms: [
        .iOS(.v13), .macOS(.v10_15), .tvOS(.v13), .watchOS(.v6), .visionOS(.v1),
    ],
    products: [
        // The runtime alone: bring your own model file (any language, any precision).
        .library(name: "AparteTitler", targets: ["AparteTitler"]),
        // The runtime and the 17-language model (133 KB), ready to use.
        .library(name: "AparteTitlerLatin", targets: ["AparteTitlerLatin"]),
    ],
    targets: [
        .target(name: "AparteTitler"),
        .target(
            name: "AparteTitlerLatin",
            dependencies: ["AparteTitler"],
            resources: [.copy("Resources/titler-v1.2-latin-int3.bin")]
        ),
        .testTarget(
            name: "AparteTitlerTests",
            dependencies: ["AparteTitler", "AparteTitlerLatin"],
            resources: [.copy("Resources/latin-int3.jsonl")]
        ),
    ]
)
