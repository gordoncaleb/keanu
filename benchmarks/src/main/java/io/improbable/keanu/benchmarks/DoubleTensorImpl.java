package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensorFactory;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensorFactory;

public enum DoubleTensorImpl {
    JVM {
        @Override
        DoubleTensorFactory getFactory() {
            return new JVMDoubleTensorFactory();
        }
    };

    abstract DoubleTensorFactory getFactory();
}
