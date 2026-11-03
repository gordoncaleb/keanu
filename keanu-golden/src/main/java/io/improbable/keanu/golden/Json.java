package io.improbable.keanu.golden;

import java.util.Collection;
import java.util.Map;

/**
 * A tiny deterministic JSON writer. Output depends only on the value written: maps are written in iteration order
 * (callers use LinkedHashMap) and doubles use Double.toString, which round trips exactly.
 * <p>
 * JSON has no NaN or infinity, so non-finite doubles are written as the strings "NaN", "Infinity" and "-Infinity".
 */
final class Json {

    private Json() {
    }

    static String write(Object value) {
        StringBuilder sb = new StringBuilder();
        write(value, sb, 0);
        sb.append('\n');
        return sb.toString();
    }

    private static void write(Object value, StringBuilder sb, int indent) {
        if (value == null) {
            sb.append("null");
        } else if (value instanceof String) {
            writeString((String) value, sb);
        } else if (value instanceof Boolean) {
            sb.append(value);
        } else if (value instanceof Double || value instanceof Float) {
            writeDouble(((Number) value).doubleValue(), sb);
        } else if (value instanceof Number) {
            sb.append(((Number) value).longValue());
        } else if (value instanceof Map) {
            writeMap((Map<?, ?>) value, sb, indent);
        } else if (value instanceof Collection) {
            writeList((Collection<?>) value, sb, indent);
        } else if (value instanceof double[]) {
            double[] array = (double[]) value;
            sb.append('[');
            for (int i = 0; i < array.length; i++) {
                if (i > 0) sb.append(", ");
                writeDouble(array[i], sb);
            }
            sb.append(']');
        } else if (value instanceof long[]) {
            long[] array = (long[]) value;
            sb.append('[');
            for (int i = 0; i < array.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(array[i]);
            }
            sb.append(']');
        } else if (value instanceof int[]) {
            int[] array = (int[]) value;
            sb.append('[');
            for (int i = 0; i < array.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(array[i]);
            }
            sb.append(']');
        } else if (value instanceof boolean[]) {
            boolean[] array = (boolean[]) value;
            sb.append('[');
            for (int i = 0; i < array.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(array[i]);
            }
            sb.append(']');
        } else {
            throw new IllegalArgumentException("Cannot write " + value.getClass() + " as JSON");
        }
    }

    static void writeDouble(double d, StringBuilder sb) {
        if (Double.isNaN(d)) {
            sb.append("\"NaN\"");
        } else if (d == Double.POSITIVE_INFINITY) {
            sb.append("\"Infinity\"");
        } else if (d == Double.NEGATIVE_INFINITY) {
            sb.append("\"-Infinity\"");
        } else {
            sb.append(Double.toString(d));
        }
    }

    private static void writeMap(Map<?, ?> map, StringBuilder sb, int indent) {
        if (map.isEmpty()) {
            sb.append("{}");
            return;
        }
        sb.append("{\n");
        int i = 0;
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            pad(sb, indent + 1);
            writeString(entry.getKey().toString(), sb);
            sb.append(": ");
            write(entry.getValue(), sb, indent + 1);
            if (++i < map.size()) sb.append(',');
            sb.append('\n');
        }
        pad(sb, indent);
        sb.append('}');
    }

    private static void writeList(Collection<?> list, StringBuilder sb, int indent) {
        if (list.isEmpty()) {
            sb.append("[]");
            return;
        }
        sb.append("[\n");
        int i = 0;
        for (Object item : list) {
            pad(sb, indent + 1);
            write(item, sb, indent + 1);
            if (++i < list.size()) sb.append(',');
            sb.append('\n');
        }
        pad(sb, indent);
        sb.append(']');
    }

    private static void pad(StringBuilder sb, int indent) {
        for (int i = 0; i < indent; i++) sb.append("  ");
    }

    private static void writeString(String s, StringBuilder sb) {
        sb.append('"');
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        sb.append('"');
    }
}
