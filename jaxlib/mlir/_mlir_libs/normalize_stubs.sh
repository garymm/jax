#!/bin/bash
# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <FILES...>"
  exit 1
fi

if [ -n "${JAXLIB_BUILD:-}" ]; then
  # If we are building jaxlib, normalize `mlir.ir` to `jaxlib.mlir.ir`
  sed -i 's/\bmlir\.ir/jaxlib.mlir.ir/g' "$@"
fi

# Normalize `mlir.ir` imports:
#   1. Replace internal module paths with public ones.
#   2. Rewrite `import mlir.ir` to `from mlir import ir`.
#   3. Deduplicate the resulting `from mlir import ir` lines.
#   4. Shorten `mlir.ir.<NAME>` to `ir.<NAME>`.
sed -i -E \
  -e 's/mlir\._mlir_libs\._mlir\.ir/mlir.ir/g' \
  -e 's/import (jaxlib\.)?mlir\.ir/from \1mlir import ir/g' \
  -e '0,/^[[:space:]]*from (jaxlib\.)?mlir import ir[[:space:]]*$/b' \
  -e '/^[[:space:]]*from (jaxlib\.)?mlir import ir[[:space:]]*$/d' \
  -e 's/mlir\.ir\.([a-zA-Z0-9_]+)/ir.\1/g' \
  "$@"
