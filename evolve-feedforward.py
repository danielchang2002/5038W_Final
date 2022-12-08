from __future__ import print_function
import multiprocessing
import pickle
from snake import *
import os
import neat
import visualize
import sys

def eval_genomes(genomes, config):
    best_genome = None
    best_fit = -1
    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness = simulate_headless(net)  

        if genome.fitness > best_fit:
            best_fit = genome.fitness
            best_genome = genome

    if best_fit >= 20:
        replay_genome(best_genome, config)

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = simulate_headless(net)  
    return fitness


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 500 generations.
    # winner = p.run(eval_genomes, 500)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, n=500)

    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

def replay_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    node_names = {
        -1 : "d_N_wall",
        -2 : "d_S_wall",
        -3 : "d_E_wall",
        -4 : "d_W_wall",
        -5 : "tail_N",
        -6 : "tail_S",
        -7 : "tail_E",
        -8 : "tail_W",
        -9 : "apple_N",
        -10 : "apple_S",
        -11 : "apple_E",
        -12 : "apple_W",
        0: 'up', 1 : "left", 2 : "down", 3 : "right"
    }

    visualize.draw_net(config, genome, False, node_names=node_names, filename="winner-feedforward.gv", prune_unused=True)

    simulate_animation(net)

def test_population(config_file, pop_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Checkpointer.restore_checkpoint(pop_file)
    p.run(eval_genomes, 1)

def test_winner(config_file, genome):
    with open(genome, "rb") as f:
        winner = pickle.load(f, encoding="latin-1")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    replay_genome(winner, config)



if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    if len(sys.argv) == 2:
        test_winner(config_path, sys.argv[1])
    else:
        run(config_path)
