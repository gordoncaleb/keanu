package io.improbable.keanu.golden;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Regenerates the golden vectors into a scratch directory and fails if they differ from the checked in copy.
 * <p>
 * Usage: GoldenVectorVerifier CHECKED_IN_DIR SCRATCH_DIR
 */
public final class GoldenVectorVerifier {

    private GoldenVectorVerifier() {
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: GoldenVectorVerifier CHECKED_IN_DIR SCRATCH_DIR");
        }
        Path checkedIn = Paths.get(args[0]);
        Path generated = Paths.get(args[1]);

        GoldenVectorExporter.export(generated);

        Set<String> expectedFiles = files(checkedIn);
        Set<String> actualFiles = files(generated);

        List<String> problems = new ArrayList<>();
        for (String file : expectedFiles) {
            if (!actualFiles.contains(file)) {
                problems.add("no longer generated: " + file);
            } else if (!Arrays.equals(Files.readAllBytes(checkedIn.resolve(file)), Files.readAllBytes(generated.resolve(file)))) {
                problems.add("changed: " + file);
            }
        }
        for (String file : actualFiles) {
            if (!expectedFiles.contains(file)) {
                problems.add("not checked in: " + file);
            }
        }

        if (!problems.isEmpty()) {
            System.err.println("Golden vectors differ from " + checkedIn + ":");
            problems.forEach(p -> System.err.println("  " + p));
            System.err.println("If this behaviour change is intended run ./gradlew :keanu-golden:generateGoldenVectors, review the diff and commit it.");
            System.err.println("Freshly generated vectors are in " + generated);
            System.exit(1);
        }
        System.out.println("Golden vectors match (" + expectedFiles.size() + " files)");
    }

    private static Set<String> files(Path root) throws IOException {
        if (!Files.isDirectory(root)) {
            return new TreeSet<>();
        }
        try (Stream<Path> paths = Files.walk(root)) {
            return paths.filter(Files::isRegularFile)
                .map(p -> root.relativize(p).toString())
                .filter(p -> !p.equals(GoldenVectorExporter.README))
                .collect(Collectors.toCollection(TreeSet::new));
        }
    }
}
