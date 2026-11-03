package io.improbable.keanu.golden;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Cases whose recorded Java result is a known defect rather than intended behaviour. The Rust port should not
 * reproduce these: the parity harness should skip the listed cases (or check them against the noted intended
 * behaviour) until the Java implementation is fixed and the vectors are regenerated.
 */
final class KnownIssues {

    private KnownIssues() {
    }

    static List<Map<String, Object>> all() {
        List<Map<String, Object>> issues = new ArrayList<>();

        // none at the moment: add entries here rather than letting the vectors encode a known defect

        return issues;
    }

    private static void issue(List<Map<String, Object>> issues, String file, String[] cases, String description, String intended) {
        Map<String, Object> issue = new LinkedHashMap<>();
        issue.put("file", file);
        List<String> caseNames = new ArrayList<>();
        for (String c : cases) {
            caseNames.add(c);
        }
        issue.put("cases", caseNames);
        issue.put("description", description);
        issue.put("intendedBehaviour", intended);
        issues.add(issue);
    }
}
