"""Run a LeRobot policy in a separate process and query it from the simulator.

The policy often needs a Python environment that cannot be combined with Isaac Lab (for example GR00T needs
flash-attn and ``transformers<5``, Isaac Lab 3.0 pins ``transformers==5.10.4``). This package splits the two:

* ``protocol``: a tiny binary request / response protocol (JSON header + raw numpy bytes, no pickle).
* ``client``: used by ``scripts/rollout.py`` (inside the simulator container).
* ``server``: the generic serve loop used by ``scripts/policy_server.py`` (on the host).
* ``runner``: loads a LeRobot policy and turns observations into actions (host side only, needs lerobot).

``protocol``, ``client`` and ``server`` only need the standard library and numpy.
"""
