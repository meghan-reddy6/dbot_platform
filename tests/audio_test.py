import time
import logging
from alerts.alert_manager import AlertManager

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing AlertManager...")
    manager = AlertManager(alert_cooldown=5.0)
    
    print("Dispatching test alert...")
    manager.dispatch(
        text="DeskBot audio system test successful",
        category="test",
        cooldown=0.0,
        identity="TestUser"
    )
    
    print("Waiting for audio to complete (max 10 seconds)...")
    
    # Wait until the queue is empty and the worker has finished processing
    manager.audio_queue.join()
    print("Audio processing complete!")
    
    # Give the thread a moment to print its completion logs
    time.sleep(0.5)

if __name__ == "__main__":
    main()
