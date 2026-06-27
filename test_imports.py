import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    import components.episode_buffer as ep
    print("SUCCESS: Imported components.episode_buffer from:", ep.__file__)
except Exception as e:
    print("FAILED to import components.episode_buffer:", str(e))

try:
    import sacred
    print("Sacred path:", sacred.__file__)
except Exception as e:
    print("FAILED to import sacred:", str(e))
