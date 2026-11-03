package io.improbable.keanu.golden;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Writes the golden test vectors that the Rust port is checked against. See golden-vectors/README.md.
 * <p>
 * Usage: GoldenVectorExporter OUTPUT_DIR
 */
public final class GoldenVectorExporter {

    /**
     * Increment when the file layout or encoding changes (not when values change).
     */
    static final int FORMAT_VERSION = 1;

    /**
     * Hand written files in the output directory that the exporter must never touch.
     */
    static final String README = "README.md";

    private GoldenVectorExporter() {
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: GoldenVectorExporter OUTPUT_DIR");
        }
        export(Paths.get(args[0]));
    }

    static void export(Path outputDir) throws IOException {
        Files.createDirectories(outputDir);
        clean(outputDir);

        Map<String, List<Map<String, Object>>> files = new LinkedHashMap<>();
        files.put("tensors.json", TensorCases.generate());
        files.put("distributions.json", DistributionCases.generate());
        files.put("vertices.json", VertexCases.generate());
        files.put("models.json", ModelCases.generate(outputDir.resolve("models")));
        files.put("algorithms.json", AlgorithmCases.generate());
        files.put("samplers.json", SamplerCases.generate());

        Map<String, Object> counts = new LinkedHashMap<>();
        for (Map.Entry<String, List<Map<String, Object>>> file : files.entrySet()) {
            Map<String, Object> contents = new LinkedHashMap<>();
            contents.put("formatVersion", FORMAT_VERSION);
            contents.put("cases", file.getValue());
            write(outputDir.resolve(file.getKey()), Json.write(contents));
            counts.put(file.getKey(), file.getValue().size());
        }

        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("formatVersion", FORMAT_VERSION);
        manifest.put("generator", "keanu-golden (" + GoldenVectorExporter.class.getName() + ")");
        manifest.put("caseCounts", counts);
        manifest.put("verticesWithoutAVertexMethod", VertexCases.skipped());
        manifest.put("knownIssues", KnownIssues.all());
        write(outputDir.resolve("manifest.json"), Json.write(manifest));
    }

    private static void write(Path path, String contents) throws IOException {
        Files.write(path, contents.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Removes previously generated files so that cases which no longer exist don't linger.
     */
    private static void clean(Path outputDir) throws IOException {
        try (Stream<Path> paths = Files.walk(outputDir)) {
            paths.sorted(Comparator.reverseOrder())
                .filter(p -> !p.equals(outputDir))
                .filter(p -> !outputDir.relativize(p).toString().equals(README))
                .forEach(p -> {
                    try {
                        Files.delete(p);
                    } catch (IOException e) {
                        throw new IllegalStateException(e);
                    }
                });
        }
    }
}
