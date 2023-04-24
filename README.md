# Dulcet
A collision-aware simulator for continuous neighbor discovery protocols. 

This is a simulator in Python that uses the SimPy discrete event framework to simulate node mobility and neighbor discovery using neighbor discovery protocols. Each node has a position in 2D space and implements a neighbor discovery protocol that issues "beacon", "scan", or "sleep" event according to its schedule. Discoveries occur if exactly one beacon overlaps with at least one scan within communication range, and a collision (i.e., failed discovery) occurs if multiple beacon events within range overlap. Each node maintains a set of discovery events that include the time and the neighbor discovered.

Dulcet can be used for many different types of neighbor discovery research. It was chiefly developed for _adaptive_ neighbor discovery research, i.e., developing protocols that can adapt the behavior of neighbor discovery to different operating contexts. This is useful for saving energy and improving performance in heterogeneous environments.

## Getting Started
First, install the requirements:

``` bash
python -m pip install -r requirements.txt
```

Then, try running a simple sim configuration. This config creates two nodes, initializes them with slotted [Birthday](https://dl.acm.org/doi/10.1145/501431.501435) schedules, runs them for a little bit, then plots their schedules.

``` bash
cd simulator
python simulator.py ./evals/plot_bday_schedule.py
```

You can also plot [BLEnd](https://dl.acm.org/doi/abs/10.1145/3055031.3055086), a more recent slotless protocol.

``` bash
cd simulator
python simulator.py ./evals/plot_blend_schedule.py
```

There are many other configurations in ``./simulator/evals`` for simulating neighbor discovery in the presence of mobility, increasing/decreasing node density, adaptive vs. static protocol performance, energy usage, etc.

## Important Note
The original optimizer and model implementation for BLEnd is a dependency of this simulator, but is not currently included in this repository. We have, however, included a large table of pre-configured schedules that are sufficient for running many of the existing simulation configurations (provided the random seed is held constant). Please contact us if you wish to obtain these implementations.

### Contact
Evan King, UT Austin, e (dot) king (at) utexas (dot) edu