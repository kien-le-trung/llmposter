# Batching Benchmarking

This benchmark measures how much `batch_prompting` reduces clue-generation latency.

Run a backend with batched prompting:

```powershell
backend\venv\Scripts\python.exe experiments\run_backend.py --config cpu_model_batched --port 8000
```

Then run:

```powershell
backend\venv\Scripts\python.exe experiments\batching_benchmarking\benchmark_batching.py --mode batched --repetitions 20
```

For per-case progress while the benchmark is running:

```powershell
backend\venv\Scripts\python.exe experiments\batching_benchmarking\benchmark_batching.py --mode batched --repetitions 20 --verbose --show-errors
```

For less noisy progress, print every 5 completed cases:

```powershell
backend\venv\Scripts\python.exe experiments\batching_benchmarking\benchmark_batching.py --mode batched --repetitions 20 --progress-every 5
```

Stop the backend, start the unbatched config:

```powershell
backend\venv\Scripts\python.exe experiments\run_backend.py --config cpu_model_unbatched --port 8000
```

Then run:

```powershell
backend\venv\Scripts\python.exe experiments\batching_benchmarking\benchmark_batching.py --mode unbatched --repetitions 20
```

Keep the model config, prompt technique, cases, and repetition count fixed when comparing modes.

Outputs are written to `experiments/batching_benchmarking/results/`.
