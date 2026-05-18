import sys
import os
from unittest.mock import MagicMock

# Add src/ to path so all source modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock all GUI / platform-specific dependencies before any source imports.
# This lets pytest run on any machine without Qt or a Raspberry Pi camera.
for _mod in [
    'PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui',
    'picamera2', 'picamera2.encoders', 'picamera2.outputs',
    'picamera2.previews', 'picamera2.previews.qt',
]:
    sys.modules[_mod] = MagicMock()

# Force matplotlib into non-interactive mode so process_main can save plots
# without a display server.
import matplotlib
matplotlib.use('Agg')
