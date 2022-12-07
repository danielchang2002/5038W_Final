from __future__ import print_function
import pickle
from snake import *
import os
import neat
import visualize
import sys

def eval_genomes(genomes, config):
    best_net = None
    best_genome = None
    best_fit = -1
    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness = simulate_headless(net)  

        if genome.fitness > best_fit:
            best_fit = genome.fitness
            best_net = net
            best_genome = genome

    # if best_fit >= 5:
    #     # d_N, d_S, d_E, d_W = y, numRows - y, numCols - x, x
    #     node_names = {
    #         -1 : "d_N_wall",
    #         -2 : "d_S_wall",
    #         -3 : "d_E_wall",
    #         -4 : "d_W_wall",
    #          0: 'up', 1 : "left", 2 : "down", 3 : "right"}
    #     visualize.draw_net(config, best_genome, True, node_names=node_names)
    #     simulate_animation(best_net)

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
    winner = p.run(eval_genomes, 500)

    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {
        -1 : "d_N_wall",
        -2 : "d_S_wall",
        -3 : "d_E_wall",
        -4 : "d_W_wall",
        -5 : "d_N_tail",
        -6 : "d_S_tail",
        -7 : "d_E_tail",
        -8 : "d_W_tail",
        -9 : "d_apple",
        -10 : "d_apple_x",
        -11 : "d_apple_y",
        0: 'up', 1 : "left", 2 : "down", 3 : "right"
    }
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)

def test(config_file, genome):
    print("Testing: ", genome)
    print(genome)
    net = neat.nn.FeedForwardNetwork.create(genome, config_file)
    simulate_animation(net)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    if len(sys.argv) == 2:
        test(config_path, sys.argv[1])
    else:
        run(config_path)
