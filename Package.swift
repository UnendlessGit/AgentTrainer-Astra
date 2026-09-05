// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AgentTrainerAstra",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "AstraCore", targets: ["AstraCore"]),
        .executable(name: "AgentTrainerAstra", targets: ["AgentTrainerAstra"])
    ],
    targets: [
        .systemLibrary(name: "CSQLite"),
        .target(name: "AstraCore", dependencies: ["CSQLite"]),
        .executableTarget(name: "AgentTrainerAstra", dependencies: ["AstraCore"]),
        .testTarget(name: "AstraCoreTests", dependencies: ["AstraCore"])
    ]
)
