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

        issue(issues, "vertices.json",
            new String[]{"unary/sign", "unary/floor", "unary/ceil", "unary/round", "reduce/max", "reduce/min"},
            "Auto diff through sign, floor, ceil, round and the max/min reductions throws NullPointerException "
                + "(and for round, forward mode succeeds while reverse mode throws).",
            "sign/floor/ceil/round have a zero derivative almost everywhere; max/min reductions pass the gradient to the "
                + "selected element(s).");

        issue(issues, "vertices.json",
            new String[]{"binary/max/[2,3]x[3]", "binary/max/[]x[2,2]", "binary/min/[2,3]x[3]", "binary/min/[]x[2,2]"},
            "The binary max/min vertices reject broadcasting shapes at construction, although the tensor operations "
                + "(tensors.json binary/max and binary/min) broadcast them.",
            "Broadcast like the tensor operations.");

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
