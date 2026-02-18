"""Metrics monitoring for VoxCPM TTS Server."""

from typing import Optional
from .config import ServerConfig


class MetricsManager:
    """
    Manager for collecting and reporting metrics.

    Uses prometheus_client if available, otherwise provides stubs.
    """

    def __init__(self, config: ServerConfig):
        """Initialize metrics manager."""
        self.config = config
        self._enabled = config.metrics_enabled
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup metrics (if prometheus_client is available)."""
        if not self._enabled:
            return

        try:
            from prometheus_client import Counter, Histogram, Gauge

            # Request metrics
            self.request_counter = Counter(
                'voxcpm_tts_requests_total',
                'Total TTS requests',
                ['mode', 'status']
            )

            self.request_duration = Histogram(
                'voxcpm_tts_request_duration_seconds',
                'TTS request duration',
                ['mode']
            )

            # Connection metrics
            self.active_connections = Gauge(
                'voxcpm_tts_active_connections',
                'Active WebSocket connections'
            )

            # Queue metrics
            self.queue_length = Gauge(
                'voxcpm_tts_queue_length',
                'Pending requests in queue'
            )

            self.running_tasks = Gauge(
                'voxcpm_tts_running_tasks',
                'Currently running TTS tasks'
            )

            # Error metrics
            self.error_counter = Counter(
                'voxcpm_tts_errors_total',
                'Total errors',
                ['error_code']
            )

            # Audio metrics
            self.audio_duration = Histogram(
                'voxcpm_tts_audio_duration_seconds',
                'Generated audio duration',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
            )

        except ImportError:
            logger = __import__('logging').getLogger(__name__)
            logger.warning("prometheus_client not available, metrics disabled")
            self._enabled = False
            self._setup_stubs()

    def _setup_stubs(self):
        """Setup stub methods when metrics are disabled."""
        class StubMetric:
            def inc(self, amount=1): pass
            def dec(self, amount=1): pass
            def set(self, value): pass
            def observe(self, amount): pass
            def labels(self, **kwargs): return self
            def time(self): return self.__enter__
            def __enter__(self): return self
            def __exit__(self, *args): pass

        self.request_counter = StubMetric()
        self.request_duration = StubMetric()
        self.active_connections = StubMetric()
        self.queue_length = StubMetric()
        self.running_tasks = StubMetric()
        self.error_counter = StubMetric()
        self.audio_duration = StubMetric()


# Global metrics manager instance
metrics_manager: Optional[MetricsManager] = None


def init_metrics(config: ServerConfig) -> MetricsManager:
    """
    Initialize the global metrics manager.

    Args:
        config: Server configuration

    Returns:
        MetricsManager instance
    """
    global metrics_manager
    metrics_manager = MetricsManager(config)
    return metrics_manager
