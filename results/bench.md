# Serving benchmark

CPU: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel

| Batch | PyTorch p50 (ms) | ONNX Runtime p50 (ms) | Speedup | ONNX rows/s |
|---|---|---|---|---|
| 1 | 0.145 | 0.024 | 6.0x | 41,322 |
| 64 | 0.326 | 0.160 | 2.0x | 399,002 |
| 1024 | 1.618 | 0.702 | 2.3x | 1,457,859 |
| 16384 | 9.355 | 9.915 | 0.9x | 1,652,437 |

End to end over HTTP

| Batch | p50 (ms) | p99 (ms) | rows/s |
|---|---|---|---|
| 1 | 1.00 | 1.57 | 999 |
| 1024 | 8.53 | 17.39 | 120,094 |
