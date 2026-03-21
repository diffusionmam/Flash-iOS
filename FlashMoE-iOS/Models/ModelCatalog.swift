/*
 * ModelCatalog.swift — Curated registry of downloadable pre-packed Flash-MoE models
 *
 * Each entry maps to a public HuggingFace repo containing pre-packed weights
 * ready for the Flash-MoE engine (config.json, model_weights.bin, packed_experts/).
 *
 * Download URLs: https://huggingface.co/{repoId}/resolve/main/{filename}
 */

import Foundation

// MARK: - Data Types

struct RepoFile: Identifiable, Codable, Sendable {
    var id: String { filename }
    let filename: String
    let sizeBytes: UInt64
}

struct CatalogEntry: Identifiable, Codable, Sendable {
    let id: String
    let displayName: String
    let repoId: String
    let description: String
    let totalSizeBytes: UInt64
    let quantization: String
    let expertLayers: Int
    let files: [RepoFile]

    var totalSizeGB: Double {
        Double(totalSizeBytes) / (1024.0 * 1024.0 * 1024.0)
    }

    func downloadURL(for file: RepoFile) -> URL {
        URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(file.filename)")!
    }
}

// MARK: - Curated Model Catalog

enum ModelCatalog {

    /// Pre-packed models available for download.
    /// New models are added in app updates.
    static let models: [CatalogEntry] = [
        // -- Qwen 3.5 35B-A3B (small, fits on iPhone with 8GB+ RAM) --
        CatalogEntry(
            id: "qwen3.5-35b-a3b-q4",
            displayName: "Qwen 3.5 35B-A3B",
            repoId: "alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE",
            description: "Compact 35B MoE model. 3B active params per token. Good for 8GB devices.",
            totalSizeBytes: 19_500_000_000,
            quantization: "4-bit",
            expertLayers: 40,
            files: makeFileList(
                configFiles: [
                    ("config.json", 3_809),
                    ("model_weights.json", 251_539),
                    ("model_weights.bin", 1_378_869_376),
                    ("vocab.bin", 3_360_287),
                    ("tokenizer.json", 19_989_343),
                    ("tokenizer.bin", 8_201_040),
                ],
                expertLayers: 40,
                expertLayerSize: 452_984_832
            )
        ),
    ]

    // MARK: - Helpers

    private static func makeFileList(
        configFiles: [(String, UInt64)],
        expertLayers: Int,
        expertLayerSize: UInt64
    ) -> [RepoFile] {
        var files = configFiles.map { RepoFile(filename: $0.0, sizeBytes: $0.1) }
        for i in 0..<expertLayers {
            files.append(RepoFile(
                filename: String(format: "packed_experts/layer_%02d.bin", i),
                sizeBytes: expertLayerSize
            ))
        }
        return files
    }
}
