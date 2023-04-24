import configparser
import sys
import logging
import json
import random
import re
from os import path

import numpy.random

from plugins.mobility.MobilityPlugin import StaticMobility
from plugins.mobility.LevyWalkMobility import LevyWalkMobility
from plugins.context.ContextPlugin import StaticContext

from protocols.blend.BLEnd import BLEnd
from protocols.birthday.Birthday import Birthday

from protocols.adaptive.CANDor import CANDor
from protocols.adaptive.Optimal import Optimal

from plugins.evaluator.EvaluatorDriver import get_evaluators
from plugins.evaluator.NodeSchedulePlotter import NodeSchedulePlotter

from Scene import Scene
from Node import Node

def choose_offset(offsets, a, b):
    """ Randomly chooses an offset not already in the array of previously-chosen offests.
    """
    offset = random.randint(a, b)

    while offset in offsets:
        offset = choose_offset(offsets, a, b)

    return offset

def import_shared_config(config, section, path_key):
    dir_path = path.dirname(sys.argv[1])
    config_path = path.join(dir_path, config[section][path_key])

    shared_config = configparser.ConfigParser()
    shared_config.read(config_path)

    for k in shared_config[section]:
        config[section][k] = shared_config[section][k]

    return config

def parse_scene_configuration(config):
    if 'node_controller' in config['global']:
        node_controller = json.loads(config['global']['node_controller'])

        logging.info(f'Node controller: {node_controller}')
    else:
        node_controller = {}

    if 'communication_range' in config['global']:
        communication_range = int(config['global']['communication_range'])
    else:
        communication_range = 300

    if 'pickle_path' in config['global']:
        pickle_path = config['global']['pickle_path']
    else:
        pickle_path = None

    logging.info(f'Pickle: {f"cache/simulations/{pickle_path}" if pickle_path is not None else False}')

    if 'log_collisions' in config['global']:
        log_collisions = bool(int(config['global']['log_collisions']))
    else:
        log_collisions = False

    # default to ms if no units given
    try:
        stop_time = int(config['global']['stop_time'])
    except ValueError:
        stop_time = config['global']['stop_time'].split()
        units = stop_time[1]
        stop_time = int(stop_time[0])

        if units == 'sec':
            stop_time *= 1000
        elif units == 'min':
            stop_time *= 60000
        elif units == 'hr':
            stop_time *= 3600000
        else:
            print(config['global']['stop_time'])
            logging.warning('Unrecognized units given for stop time.')
            sys.exit()

    return {
        'node_controller': node_controller,
        'communication_range': communication_range,
        'pickle_path': pickle_path,
        'log_collisions': log_collisions,
        'stop_time': stop_time
    }

def parse_evaluators(config, scene):
    ## parse evaluators
    evaluators = {}

    # multiple evaluators
    if 'evaluators' in config:
        ls = json.loads(config['evaluators']['evaluators'])

        for evaluator in ls:
            evaluators[evaluator['type']] = evaluator['parameters']

        evaluators = get_evaluators(evaluators, scene=scene)
        logging.info(f'Evaluators: {evaluators}')

    # single evaluator
    elif 'evaluator' in config:
        evaluator = config['evaluator']['evaluator']
        parameters = json.loads(config['evaluator']['parameters'])

        evaluators = {
            evaluator: parameters
        }

        evaluators = get_evaluators(evaluators, scene=scene)
        logging.info(f'Evaluator: {evaluators}')

    return evaluators

def run_simulation(Ne, Na, config):
    scene = Scene(**parse_scene_configuration(config))
    scene.set_evaluators(parse_evaluators(config, scene))

    if 'mobility' in config:
        if 'shared' in config['mobility']:
            config = import_shared_config(config, 'mobility', 'shared')
            
        # figure out the class of the mobility plugin
        mobility_class = globals()[config['mobility']['mobility']]

        # instantiate the mobility plugin with the given parameters
        mobility_parameters = json.loads(config['mobility']['parameters'])

        logging.info(f'Mobility: {config["mobility"]["mobility"]}: {mobility_parameters}')
    else:
        mobility_class = globals()['StaticMobility']
        mobility_parameters = {'position': (0, 0)}

    # parse shared protocol configs first
    if 'shared' in config['protocol']:
        config = import_shared_config(config, 'protocol', 'shared')

    # parse protocol
    protocol = config['protocol']['protocol']

    logging.info(f'Protocol: {protocol}')

    if 'adaptation_protocol' in config['protocol']:
        adaptation_class = globals()[config['protocol']['adaptation_protocol']]

        if 'adaptation_parameters' in config['protocol']:
            adaptation_parameters = json.loads(config['protocol']['adaptation_parameters'])
        else:
            adaptation_parameters = {}
    else:
        adaptation_class = None
    
    logging.info(f'Adaptation Protocol: {adaptation_class}')

    # instruct the discovery protocols to keep track of their schedule state over time if using the schedule plotter
    if any(isinstance(e, NodeSchedulePlotter) for e in scene.evaluators):
        log_schedule = True
    else:
        log_schedule = False

    if protocol == 'blend':
        parameters = {
            'P': float(config['protocol']['P']),
            'Lambda': int(config['protocol']['Lambda']),
            'Ne': Ne
        }

        logging.info(f'Protocol Parameters: {parameters}')

        # stores random starting offsets
        offsets = []

        # make the nodes
        for i in range(0, Na):
            offsets.append(choose_offset(offsets, 0, parameters['Lambda']))
            name = f'{i}'
            mobility_parameters['parent_node'] = name

            if adaptation_class is not None:
                adaptation_protocol = adaptation_class(**adaptation_parameters)
            else:
                adaptation_protocol = None

            Node(scene, 
                name,
                mobility_class(**mobility_parameters),
                StaticContext(f'{i}'), 
                BLEnd(parameters=parameters, log_schedule=log_schedule), 
                start_offset=offsets[-1],
                adaptation_protocol=adaptation_protocol
            )

    elif protocol == 'birthday':
        if 'slot_duration' in config['protocol']:
            slot_duration = int(config['protocol']['slot_duration'])
        else:
            slot_duration = 1

        parameters = {
            'Ne': Ne,
            'P': float(config['protocol']['P']),
            'slot_duration': slot_duration,
            'mode': config['protocol']['mode']
        }

        # make a tmp protocol to calculate lambda (the window for adaptation)
        tmp = Birthday(**parameters)

        logging.info(f'CND Parameters: {parameters}')
        logging.info(f'w = {tmp.Lambda}')

        # make the nodes
        for i in range(0, Na):
            parameters['w'] = tmp.Lambda
            bday = Birthday(**parameters, log_schedule=log_schedule)
            name = f'{i}'
            mobility_parameters['parent_node'] = name

            if adaptation_class is not None:
                adaptation_protocol = adaptation_class(**adaptation_parameters)
                adaptation_protocol.w = tmp.Lambda
            else:
                adaptation_protocol = None

            Node(scene, 
                name,
                mobility_class(**mobility_parameters), 
                StaticContext(f'{i}'), 
                bday,
                adaptation_protocol=adaptation_protocol,
                start_offset=0
            )

    else:
        logging.warning(f'Protocol {protocol} is not a valid choice.')

    scene.env_run()

def parse_N_config(N):
    """ Parses a list of inputs for node density (actual or estimated)
    
    Valid Input Syntax: 
          a single value (e.g. 10), 
          a range (e.g. 10,100), 
          a range and a first value (e.g. 2,10,100)
          a list (e.g. 2,10,20,50)
    """
    if ',' in N:
        N = N.split(',')

        if len(N) == 2: # range given, e.g. 10,100 = [10, 20, 30, ..., 100]
            Ns = range(int(N[0]), int(N[1])+10, 10)
        if len(N) == 3: # first value given, e.g. 2,10,100 = [2, 10, 20, 30, ..., 100]
            Ns = list(range(int(N[1]), int(N[2])+10, 10))
            Ns.insert(0, int(N[0]))
        elif len(N) > 3: # list given, e.g. 2,10,30,40 = [2, 10, 30, 40]
            Ns = [int(x) for x in N]
    else:
        Ns = [int(N)]

    return Ns

def main():
    if len(sys.argv) < 2:
        logging.warning('No simulation config provided.')
        logging.warning('\tUsage: python simulator.py path/to/sim-config.ini')
        sys.exit()

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    if 'global' not in config:
        logging.warning('Either an invalid simulation config was provided, or the provided config is missing required global settings.')
        sys.exit()

    # parse the external global config, if given
    if 'shared' in config['global']:
        config = import_shared_config(config, 'global', 'shared')

    # get the log level, if given
    if 'log_level' in config['global']:
        try:
            log_level = getattr(logging, config['global']['log_level'])
        except AttributeError:
            log_level = logging.INFO
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level)

    # seed the rng for repeatability, if the seed is given
    if 'random_seed' in config['global']:
        random.seed(int(config['global']['random_seed']))
        numpy.random.seed(int(config['global']['random_seed']))

    # parse the number of simulation runs
    if 'runs' in config['global']:
        runs = int(config['global']['runs'])
    else:
        runs = 1

    terminal_width = 75

    logging.info('=' * terminal_width)
    logging.info(f'Config: {sys.argv[1]}')

    # print the given description of the simulation config we're about to run
    if 'description' in config['global']:
        logging.info(f'Description: {config["global"]["description"]}')

    logging.info('=' * terminal_width)

    Nas = parse_N_config(config['global']['Na'])
    Nes = parse_N_config(config['global']['Ne'])

    logging.info(f'Invoking simulator for {runs} run(s) per Ne in {Nes} and Na in {Nas}')

    for Ne in Nes:
        for Na in Nas:
            # gotta make sure the console output is neat :)
            head = f'== Ne = {Ne}, Na = {Na} '
            logging.info(head + '=' * (terminal_width - len(head)))
            
            # invoke the sim once for each run
            for i in range(0, runs):
                head = f'== Run {i} '
                logging.info(head + '=' * (terminal_width - len(head)))
                run_simulation(Ne, Na, config)

if __name__ == '__main__':
    if '--memuse' in sys.argv:
        # output memory usage statistics, from https://docs.python.org/3/library/tracemalloc.html
        import tracemalloc, linecache

        def display_top(snapshot, key_type='lineno', limit=10):
            snapshot = snapshot.filter_traces((
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            ))
            top_stats = snapshot.statistics(key_type)

            print("Top %s lines" % limit)
            for index, stat in enumerate(top_stats[:limit], 1):
                frame = stat.traceback[0]
                print("#%s: %s:%s: %.1f KiB"
                    % (index, frame.filename, frame.lineno, stat.size / 1024))
                line = linecache.getline(frame.filename, frame.lineno).strip()
                if line:
                    print('    %s' % line)

            other = top_stats[limit:]
            if other:
                size = sum(stat.size for stat in other)
                print("%s other: %.1f KiB" % (len(other), size / 1024))
            total = sum(stat.size for stat in top_stats)
            print("Total allocated size: %.1f KiB" % (total / 1024))

        tracemalloc.start()

    main()

    if '--memuse' in sys.argv:
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot)