# Serving benchmark

CPU: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel

| Batch | PyTorch p50 (ms) | ONNX Runtime p50 (ms) | Speedup | ONNX rows/s |
|---|---|---|---|---|
| 1 | 0.143 | 0.023 | 6.3x | 43,668 |
| 64 | 0.328 | 0.150 | 2.2x | 427,807 |
| 1024 | 1.285 | 0.718 | 1.8x | 1,426,979 |
| 16384 | 11.485 | 9.506 | 1.2x | 1,723,534 |

End to end over HTTP (one client, sequential requests)

| Batch | Round trip p50 (ms) | p99 (ms) | Server-side scoring p50 (ms) | rows/s |
|---|---|---|---|---|
| 1 | 1.08 | 2.57 | 0.08 | 922 |
| 1024 | 9.44 | 22.24 | 1.60 | 108,429 |
