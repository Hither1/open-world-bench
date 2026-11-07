"""``python -m owb.cli`` is the ``owb`` console script."""
import sys

from .main import main

sys.exit(main())
