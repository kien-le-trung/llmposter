from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.services.metrics import RequestMetric, list_recent_request_metrics

router = APIRouter(prefix="/admin", tags=["admin"])


class StageMetricResponse(BaseModel):
    name: str
    duration_ms: float
    metadata: dict[str, str]


class RequestMetricResponse(BaseModel):
    id: str
    method: str
    path: str
    status_code: int
    started_at: str
    duration_ms: float
    stages: list[StageMetricResponse]


class MetricsSummaryResponse(BaseModel):
    request_count: int
    average_latency_ms: float
    p95_latency_ms: float
    error_rate: float


class MetricsResponse(BaseModel):
    summary: MetricsSummaryResponse
    requests: list[RequestMetricResponse]


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    requests = list_recent_request_metrics()
    return MetricsResponse(
        summary=build_summary(requests),
        requests=[
            RequestMetricResponse(
                id=request.id,
                method=request.method,
                path=request.path,
                status_code=request.status_code,
                started_at=request.started_at.isoformat(),
                duration_ms=request.duration_ms,
                stages=[
                    StageMetricResponse(
                        name=stage.name,
                        duration_ms=stage.duration_ms,
                        metadata=stage.metadata,
                    )
                    for stage in request.stages
                ],
            )
            for request in requests
        ],
    )


@router.get("/dashboard", response_class=HTMLResponse)
def get_dashboard() -> str:
    return DASHBOARD_HTML


def build_summary(requests: list[RequestMetric]) -> MetricsSummaryResponse:
    if not requests:
        return MetricsSummaryResponse(
            request_count=0,
            average_latency_ms=0,
            p95_latency_ms=0,
            error_rate=0,
        )

    latencies = sorted(request.duration_ms for request in requests)
    p95_index = max(0, int((len(latencies) - 1) * 0.95))
    errors = [request for request in requests if request.status_code >= 500]
    return MetricsSummaryResponse(
        request_count=len(requests),
        average_latency_ms=sum(latencies) / len(latencies),
        p95_latency_ms=latencies[p95_index],
        error_rate=len(errors) / len(requests),
    )


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLMposter Metrics</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      color: #172026;
      background: #f6f7f9;
    }
    header {
      padding: 20px 28px;
      background: #ffffff;
      border-bottom: 1px solid #d9dee5;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    h1 {
      font-size: 20px;
      margin: 0;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }
    button {
      border: 1px solid #b7c0cc;
      background: #ffffff;
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .metric {
      background: #ffffff;
      border: 1px solid #d9dee5;
      border-radius: 8px;
      padding: 14px;
    }
    .label {
      font-size: 12px;
      color: #5c6670;
      margin-bottom: 8px;
    }
    .value {
      font-size: 24px;
      font-weight: 700;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #d9dee5;
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid #eef1f4;
      font-size: 13px;
      vertical-align: top;
    }
    th {
      color: #5c6670;
      background: #fbfcfd;
      font-weight: 600;
    }
    .stage {
      display: inline-block;
      margin: 0 6px 6px 0;
      padding: 4px 6px;
      border-radius: 5px;
      background: #eef4ff;
      color: #24466f;
      white-space: nowrap;
    }
    .status-error {
      color: #b42318;
      font-weight: 700;
    }
    .status-ok {
      color: #067647;
      font-weight: 700;
    }
    @media (max-width: 780px) {
      .grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      table {
        display: block;
        overflow-x: auto;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>LLMposter Metrics</h1>
    <button id="refresh">Refresh</button>
  </header>
  <main>
    <section class="grid">
      <div class="metric"><div class="label">Requests</div><div class="value" id="requestCount">0</div></div>
      <div class="metric"><div class="label">Avg Latency</div><div class="value" id="avgLatency">0 ms</div></div>
      <div class="metric"><div class="label">P95 Latency</div><div class="value" id="p95Latency">0 ms</div></div>
      <div class="metric"><div class="label">Error Rate</div><div class="value" id="errorRate">0%</div></div>
    </section>
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Route</th>
          <th>Status</th>
          <th>Total</th>
          <th>Stages</th>
        </tr>
      </thead>
      <tbody id="requestRows"></tbody>
    </table>
  </main>
  <script>
    const formatMs = (value) => `${Math.round(value)} ms`;
    const formatPercent = (value) => `${Math.round(value * 100)}%`;

    async function loadMetrics() {
      const response = await fetch('/admin/metrics', { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`Failed to load metrics: ${response.status}`);
      }
      const data = await response.json();
      document.getElementById('requestCount').textContent = data.summary.request_count;
      document.getElementById('avgLatency').textContent = formatMs(data.summary.average_latency_ms);
      document.getElementById('p95Latency').textContent = formatMs(data.summary.p95_latency_ms);
      document.getElementById('errorRate').textContent = formatPercent(data.summary.error_rate);

      const rows = document.getElementById('requestRows');
      rows.innerHTML = '';
      for (const request of data.requests) {
        const row = document.createElement('tr');
        const statusClass = request.status_code >= 500 ? 'status-error' : 'status-ok';
        const stages = request.stages.length
          ? request.stages.map((stage) => `<span class="stage">${stage.name}: ${formatMs(stage.duration_ms)}</span>`).join('')
          : '<span class="stage">no internal stages</span>';
        row.innerHTML = `
          <td>${new Date(request.started_at).toLocaleTimeString()}</td>
          <td>${request.method} ${request.path}</td>
          <td class="${statusClass}">${request.status_code}</td>
          <td>${formatMs(request.duration_ms)}</td>
          <td>${stages}</td>
        `;
        rows.appendChild(row);
      }
    }

    document.getElementById('refresh').addEventListener('click', loadMetrics);
    loadMetrics();
    setInterval(loadMetrics, 5000);
  </script>
</body>
</html>
"""
