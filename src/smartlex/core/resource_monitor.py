import threading
import time
import psutil
from smartlex.core.logger import setup_logger

logger = setup_logger("resource_monitor")

class ResourceMonitor(threading.Thread):
    def __init__(self, max_cpu_percent=75.0, check_interval_seconds=2):
        super().__init__(daemon=True)
        self.max_cpu_percent = max_cpu_percent
        self.check_interval_seconds = check_interval_seconds
        
        # This event dictates whether deep ML tasks are allowed to run
        self.can_run_heavy_tasks = threading.Event()
        self.can_run_heavy_tasks.set()  # Allow by default
        
        self._stop_event = threading.Event()

    def run(self):
        logger.info("ResourceMonitor started. Tracking CPU limits...")
        
        while not self._stop_event.is_set():
            # Get CPU usage over a short interval (blocking for 0.5s)
            cpu_usage = psutil.cpu_percent(interval=0.5)
            
            if cpu_usage > self.max_cpu_percent:
                if self.can_run_heavy_tasks.is_set():
                    logger.warning(f"High CPU usage ({cpu_usage}%). Pausing deep indexing tasks...")
                    self.can_run_heavy_tasks.clear()
            else:
                if not self.can_run_heavy_tasks.is_set():
                    logger.info(f"CPU usage normalized ({cpu_usage}%). Resuming deep indexing tasks...")
                    self.can_run_heavy_tasks.set()
            
            # Wait for next check interval
            self._stop_event.wait(self.check_interval_seconds - 0.5)

    def stop(self):
        self._stop_event.set()

# Global singleton-like instance for easy import
monitor = ResourceMonitor()
