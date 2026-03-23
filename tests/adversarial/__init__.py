"""
tests/adversarial/
===================
Adversarial testing suite — tries to break the system to prove it's robust.

If your system can't survive adversarial testing, it won't survive the market.
"""

from tests.adversarial.runner import AdversarialRunner

__all__ = ["AdversarialRunner"]
