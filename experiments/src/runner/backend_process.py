from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import sys
import time

from backend_handler.api_client import BackendExperimentApiClient

from .config import BackendConfig, ComponentConfig


REPO_DIR = Path(__file__).resolve().parents[3]
START_BACKEND_PATH = REPO_DIR / "experiments" / "src" / "backend_handler" / "start_backend.py"


class BackendProcess:
    def __init__(
        self,
        backend_config: BackendConfig,
        components: ComponentConfig,
    ) -> None:
        self.backend_config = backend_config
        self.components = components
        self.process: subprocess.Popen | None = None

    @property
    def backend_url(self) -> str:
        return f"http://{self.backend_config.host}:{self.backend_config.port}"

    def start(self) -> None:
        command = [
            sys.executable,
            str(START_BACKEND_PATH),
            "--llm",
            self.components.llm,
            "--embedding",
            self.components.embedding,
            "--prompt",
            self.components.prompt,
            "--eval-dataset",
            self.components.eval_dataset,
            "--voting-algo",
            self.components.voting_algo,
            "--inference-mode",
            self.components.inference_mode,
            "--agent-config-source",
            self.components.agent_config_source,
            "--word-selection-mode",
            self.components.word_selection_mode,
            "--host",
            self.backend_config.host,
            "--port",
            str(self.backend_config.port),
        ]
        if self.backend_config.reload:
            command.append("--reload")

        self.process = subprocess.Popen(command, cwd=str(REPO_DIR))

    async def wait_until_healthy(self) -> None:
        if self.process is None:
            raise RuntimeError("Backend process has not been started")

        client = BackendExperimentApiClient(
            self.backend_url,
            timeout_seconds=self.backend_config.startup_timeout_seconds,
            request_timeout_seconds=2.0,
            poll_interval_seconds=0.25,
        )
        deadline = time.perf_counter() + self.backend_config.startup_timeout_seconds
        last_error: str | None = None

        while time.perf_counter() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(f"Backend process exited early with code {return_code}")

            artifact = await client.health_check()
            if artifact.success:
                return

            last_error = artifact.error or artifact.response_text
            await asyncio.sleep(0.25)

        raise TimeoutError(
            f"Backend did not become healthy within "
            f"{self.backend_config.startup_timeout_seconds}s: {last_error}"
        )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is not None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


class ManagedBackend:
    def __init__(
        self,
        backend_config: BackendConfig,
        components: ComponentConfig,
    ) -> None:
        self.backend = BackendProcess(backend_config, components)

    async def __aenter__(self) -> BackendProcess:
        self.backend.start()
        await self.backend.wait_until_healthy()
        return self.backend

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.backend.stop()
