package io.improbable.keanu.golden;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.lng.LongTensor;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Helpers shared by the case generators: encoding values as JSON and recording results or errors.
 */
final class Cases {

    private Cases() {
    }

    /**
     * {"dtype": "f64" | "i32" | "i64" | "bool", "shape": [...], "data": [...]} with data in row-major order.
     */
    static Map<String, Object> tensor(Tensor<?, ?> t) {
        Map<String, Object> m = new LinkedHashMap<>();
        if (t instanceof DoubleTensor) {
            m.put("dtype", "f64");
            m.put("shape", t.getShape());
            m.put("data", ((DoubleTensor) t).asFlatDoubleArray());
        } else if (t instanceof IntegerTensor) {
            m.put("dtype", "i32");
            m.put("shape", t.getShape());
            m.put("data", ((IntegerTensor) t).asFlatIntegerArray());
        } else if (t instanceof LongTensor) {
            m.put("dtype", "i64");
            m.put("shape", t.getShape());
            m.put("data", ((LongTensor) t).asFlatLongArray());
        } else if (t instanceof BooleanTensor) {
            m.put("dtype", "bool");
            m.put("shape", t.getShape());
            Boolean[] values = ((BooleanTensor) t).asFlatArray();
            boolean[] data = new boolean[values.length];
            for (int i = 0; i < values.length; i++) {
                data[i] = values[i];
            }
            m.put("data", data);
        } else {
            throw new IllegalArgumentException("Unsupported tensor type " + t.getClass());
        }
        return m;
    }

    static Object encode(Object value) {
        if (value instanceof Tensor) {
            return tensor((Tensor<?, ?>) value);
        } else if (value instanceof int[] || value instanceof long[] || value instanceof double[] || value instanceof boolean[]
            || value instanceof Number || value instanceof String || value instanceof Boolean || value == null) {
            return value;
        } else if (value instanceof Map) {
            Map<String, Object> m = new LinkedHashMap<>();
            ((Map<?, ?>) value).forEach((k, v) -> m.put(k.toString(), encode(v)));
            return m;
        } else if (value instanceof List) {
            List<Object> l = new ArrayList<>();
            ((List<?>) value).forEach(v -> l.add(encode(v)));
            return l;
        } else if (value instanceof Object[]) {
            List<Object> l = new ArrayList<>();
            for (Object v : (Object[]) value) {
                l.add(encode(v));
            }
            return l;
        }
        throw new IllegalArgumentException("Cannot encode " + value.getClass());
    }

    /**
     * Runs the computation and returns its encoded result, or {"error": "ExceptionSimpleName"} if it throws.
     * The Rust port is expected to fail in the same cases, not necessarily with the same error type name.
     */
    static Object result(Supplier<Object> computation) {
        try {
            return encode(computation.get());
        } catch (RuntimeException e) {
            Throwable cause = e instanceof ReflectiveCallException ? e.getCause() : e;
            Map<String, Object> error = new LinkedHashMap<>();
            error.put("error", cause.getClass().getSimpleName());
            return error;
        }
    }

    static Map<String, Object> testCase(String name, String op, List<Object> args, Map<String, Object> params, Object result) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", name);
        m.put("op", op);
        m.put("args", encode(args));
        if (params != null && !params.isEmpty()) {
            m.put("params", encode(params));
        }
        m.put("result", result);
        return m;
    }

    /**
     * Calls a public method by name, picking the overload whose parameter types accept the arguments
     * (with int/long/double/boolean arguments matching their primitive parameters).
     */
    static Object invoke(Object target, String method, Object... args) {
        Method chosen = resolve(target.getClass(), method, args);
        try {
            chosen.setAccessible(true);
            return chosen.invoke(target, args);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof RuntimeException) {
                throw new ReflectiveCallException(e.getCause());
            }
            throw new IllegalStateException(e.getCause());
        }
    }

    /**
     * Deterministic overload resolution: ignores bridge methods, then picks the candidate whose parameter types are
     * all assignable to every other candidate's (the most specific). Class.getMethods() has no defined order, so
     * picking the first match would make the output depend on the JVM run.
     */
    private static Method resolve(Class<?> type, String method, Object[] args) {
        List<Method> candidates = new ArrayList<>();
        for (Method m : type.getMethods()) {
            if (m.getName().equals(method) && !m.isBridge() && !m.isSynthetic()
                && m.getParameterCount() == args.length && accepts(m.getParameterTypes(), args)) {
                candidates.add(m);
            }
        }
        if (candidates.isEmpty()) {
            throw new IllegalStateException("No method " + method + " on " + type.getSimpleName() + " accepting " + args.length + " args");
        }

        List<Method> mostSpecific = new ArrayList<>();
        for (Method candidate : candidates) {
            boolean isMostSpecific = true;
            for (Method other : candidates) {
                if (!isAtLeastAsSpecific(candidate, other)) {
                    isMostSpecific = false;
                    break;
                }
            }
            if (isMostSpecific) {
                mostSpecific.add(candidate);
            }
        }

        // equally specific candidates are the same signature declared in several types; they behave identically
        mostSpecific.sort(java.util.Comparator.comparing(Method::toGenericString));
        if (mostSpecific.isEmpty()) {
            throw new IllegalStateException("Ambiguous call to " + method + " on " + type.getSimpleName() + ": " + candidates);
        }
        return mostSpecific.get(0);
    }

    private static boolean isAtLeastAsSpecific(Method a, Method b) {
        Class<?>[] aTypes = a.getParameterTypes();
        Class<?>[] bTypes = b.getParameterTypes();
        for (int i = 0; i < aTypes.length; i++) {
            if (!bTypes[i].isAssignableFrom(aTypes[i]) && !(aTypes[i].isPrimitive() && aTypes[i] == bTypes[i])) {
                return false;
            }
        }
        return true;
    }

    private static boolean accepts(Class<?>[] types, Object[] args) {
        for (int i = 0; i < types.length; i++) {
            Class<?> type = types[i];
            Object arg = args[i];
            if (type.isPrimitive()) {
                if (!(type == int.class && arg instanceof Integer
                    || type == long.class && arg instanceof Long
                    || type == double.class && arg instanceof Double
                    || type == boolean.class && arg instanceof Boolean)) {
                    return false;
                }
            } else if (arg != null && !type.isInstance(arg)) {
                return false;
            }
        }
        return true;
    }

    static final class ReflectiveCallException extends RuntimeException {
        ReflectiveCallException(Throwable cause) {
            super(cause);
        }
    }

    @SafeVarargs
    static <T> List<T> list(T... items) {
        List<T> l = new ArrayList<>();
        for (T item : items) {
            l.add(item);
        }
        return l;
    }

    static Map<String, Object> params(Object... keyValues) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < keyValues.length; i += 2) {
            m.put((String) keyValues[i], keyValues[i + 1]);
        }
        return m;
    }
}
