"""Allow running as: python -m memnest_mcp [serve|config ...]"""
import sys

from .cli import main

sys.exit(main())
