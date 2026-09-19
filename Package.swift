// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AgentTrainerAstra",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "AstraCore", targets: ["AstraCore"]),
        .executable(name: "AgentTrainerAstra", targets: ["AgentTrainerAstra"]),
        .executable(name: "AstraFixture", targets: ["AstraFixture"]),
        .executable(name: "AstraControl", targets: ["AstraControl"]),
        .executable(name: "AstraRecoveryFixture", targets: ["AstraRecoveryFixture"]),
        .executable(name: "AstraReceiptFixture", targets: ["AstraReceiptFixture"])
    ],
    targets: [
        .systemLibrary(name: "CSQLite"),
        .target(name: "CAstraRecovery", publicHeadersPath: "include"),
        .target(name: "AstraCore", dependencies: ["CSQLite", "CAstraRecovery"]),
        .target(name: "AstraPlatform", dependencies: ["AstraCore", "CAstraRecovery"]),
        .executableTarget(name: "AgentTrainerAstra", dependencies: ["AstraCore", "AstraPlatform"]),
        .executableTarget(name: "AstraFixture", dependencies: ["AstraCore"]),
        .executableTarget(name: "AstraControl", dependencies: ["AstraCore", "AstraPlatform"]),
        .executableTarget(name: "AstraRecoveryFixture", dependencies: ["AstraCore", "AstraPlatform"], path: "Tests/RecoveryFixture"),
        .executableTarget(name: "AstraReceiptFixture", dependencies: ["AstraCore", "AstraPlatform"], path: "Tests/ReceiptFixture"),
        .testTarget(name: "AstraCoreTests", dependencies: ["AstraCore"]),
        .testTarget(name: "AstraPlatformTests", dependencies: ["AstraCore", "AstraPlatform", "AstraRecoveryFixture"]),
        .testTarget(name: "AstraAppTests", dependencies: ["AgentTrainerAstra", "AstraCore", "AstraPlatform"])
    ]
)
