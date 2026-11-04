# Third-party notices

`keanu-algorithms` is licensed under the MIT licence, except for the files in `src/optimizer/bobyqa/prima/`. Those
files are a Rust translation of the BOBYQA solver from PRIMA (https://github.com/libprima/prima), revision
43863c69ccd6408dedcd782d0dfdcbaad404dcde, and remain under PRIMA's BSD 3-Clause licence, reproduced below.
PRIMA is a modernised reference implementation of M. J. D. Powell's derivative-free optimisation solvers.

```
BSD 3-Clause License

Copyright (c) 2020--2026, Zaikun ZHANG ( https://www.zhangzk.net )

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Changes from PRIMA

- Translated from Fortran to Rust. Debugging assertions, printing and history recording are omitted, and indices
  are 0-based.
- In `rescue`, the term W1'·(B1·W1 + B·W) is computed with W2 = WMV(NPT+1:NPT+N), as in Powell's original
  BOBYQA and the derivation in PRIMA's own comments. The ported revision uses WMV(1:N).
- Temporary vectors in the linear algebra helpers and the main loops are stored inline (on the stack) for small
  problems instead of being heap allocated. This is a performance change only; the arithmetic is unchanged.
- The Lagrange function values (`calvlag`) and BETA (`calbeta`) for a step are computed once and shared by the
  rescue test, `setdrop_tr` and `updateh`, which PRIMA computes separately from the same inputs; `trsbox` uses the
  model without copying it unless it has to be scaled; `geostep` computes the line step's denominators only when
  it compares them with a Cauchy step. The arithmetic is unchanged, so results are bit-identical.
- PRIMA's callback termination is offered through the objective itself: `minimize_with_termination` takes an
  objective returning `ControlFlow`, and `Break` ends the solve after that evaluation with
  `ExitStatus::CallbackTerminate`.
