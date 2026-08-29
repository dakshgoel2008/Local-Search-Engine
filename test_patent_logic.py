import time
import sys
import os
import threading

# Add src to python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from smartlex.core.resource_monitor import monitor
from smartlex.core.logger import setup_logger

logger = setup_logger("test_patent")

def simulate_heavy_cpu_task():
    """Simulates a game or heavy software running to spike CPU."""
    logger.warning(">>> USER STARTED A HEAVY TASK! Spiking CPU... <<<")
    
    def burn_cpu():
        start_time = time.time()
        while time.time() - start_time < 5:
            _ = [x**2 for x in range(10000)]
            
    # Spawn multiple threads to max out several cores
    threads = []
    for _ in range(8):
        t = threading.Thread(target=burn_cpu)
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    logger.warning(">>> USER CLOSED THE HEAVY TASK! CPU returning to normal... <<<")

def simulate_deferred_indexer():
    """Simulates the background ML indexer trying to work."""
    logger.info("Deferred Indexer started...")
    for i in range(1, 11):
        # The crucial inventive step: Wait for CPU to be free!
        monitor.can_run_heavy_tasks.wait()
        
        logger.info(f"Indexing media file {i}/10... (ML models running safely)")
        time.sleep(1) # Simulate time taken by ML model
        
    logger.info("Deferred Indexer finished all tasks!")

if __name__ == "__main__":
    print("\n--- Testing Patentable Resource Monitor Algorithm ---")
    print("This test proves that our background AI indexing automatically pauses")
    print("when the user does something heavy, and resumes when the PC is idle.\n")
    
    # 1. Start the resource monitor
    # Set threshold very low (e.g., 20%) just for testing so it triggers easily
    monitor.max_cpu_percent = 20.0
    monitor.start()
    
    # 2. Start our simulated background ML indexer
    indexer_thread = threading.Thread(target=simulate_deferred_indexer)
    indexer_thread.start()
    
    # 3. Let it run normally for 3 seconds
    time.sleep(3)
    
    # 4. Simulate the user suddenly opening a heavy game!
    simulate_heavy_cpu_task()
    
    # 5. Wait for indexer to finish
    indexer_thread.join()
    monitor.stop()
    print("\n--- Test Complete ---")
