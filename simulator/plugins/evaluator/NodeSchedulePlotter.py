import logging

import matplotlib.pyplot as plt

from plugins.evaluator.EvaluatorPlugin import EvaluatorPlugin

class NodeSchedulePlotter(EvaluatorPlugin):
    """ Generates a plot of node schedule. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # which nodes to plot (by name)
        self.nodes = kwargs['nodes'].split(',')

    def update(self):
        if self.scene.time == self.trigger_step:
            self.run_evaluation()

    def run_evaluation(self):
        fig, ax = plt.subplots()

        collision_times = [t for t, c, n in self.scene.collisions if n > 0]
        
        for name in self.nodes:
            pts = list(map(list, zip(*self.scene.nodes[name].ndprotocol.schedule_plot)))
            ax.plot(pts[0], pts[1], marker='.', label=f'{name}')

            logging.info(f'[NodeSchedulePlotter] {name}: {pts[1].count(2)}ms scan, {pts[1].count(1)}ms beacon, {pts[1].count(-1)}ms advertisement intervals')

            collisions = [x for x in [t for t, b in self.scene.nodes[name].ndprotocol.schedule_plot if b == 1] if x in collision_times]

            ax.plot(collisions, [1] * len(collisions), linestyle='None', marker='x', color='r')

            # remove the collisions we already plotted (don't need to plot them more than once)
            collision_times = [x for x in collision_times if x not in collisions]

        plt.legend()
        plt.title('Node schedule')
        plt.xlabel('time (ms)')
        plt.ylabel('state')
        plt.show()