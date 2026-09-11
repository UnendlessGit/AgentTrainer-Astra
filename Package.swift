// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AgentTrainerAstra",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "AstraCore", targets: ["AstraCore"]),
        .executable(name: "AgentTrainerAstra", targets: ["AgentTrainerAstra"]),
        .executable(name: "AstraFixture", targets: ["AstraFixture"]),
        .executable(name: "AstraControl", targets: ["AstraControl"])
    ],
    targets: [
        .systemLibrary(name: "CSQLite"),
        .target(name: "AstraCore", dependencies: ["CSQLite"]),
        .target(name: "AstraPlatform", dependencies: ["AstraCore"]),
        .executableTarget(name: "AgentTrainerAstra", dependencies: ["AstraCore", "AstraPlatform"]),
        .executableTarget(name: "AstraFixture", dependencies: ["AstraCore"]),
        .executableTarget(name: "AstraControl", dependencies: ["AstraCore", "AstraPlatform"]),
        .testTarget(name: "AstraCoreTests", dependencies: ["AstraCore"]),
        .testTarget(name: "AstraPlatformTests", dependencies: ["AstraCore", "AstraPlatform"]),
        .testTarget(name: "AstraAppTests", dependencies: ["AgentTrainerAstra", "AstraCore", "AstraPlatform"])
    ]
)
