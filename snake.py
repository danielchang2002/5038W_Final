from random import random
import random as random_module
import numpy as np
import pygame

random_module.seed(42)

# ----------------simulation-----------------
numRows, numCols = 10, 10
snake = [(3, 3)]
apple = (5, 5)
v_x, v_y = 1, 0 
dead = False
NUM_ITERS = 10
# MAX_TIME_STEPS = 1000
MIN_TIME_TO_EAT_APPLE = 100
# ----------------simulation-----------------

# ----------------animation stuff--------------
interval = 100
NODE_SIZE = 25
networkWidth, networkHeight = 700, 900
gameWidth, gameHeight = 900, 900
window_buffer = 25
screenWidth = window_buffer + networkWidth + window_buffer + gameWidth + window_buffer
screenHeight = networkHeight + 2 * window_buffer
blockWidth, blockHeight = gameWidth / numCols, gameHeight / numRows
gameTopLeft = (2 * window_buffer + networkWidth, window_buffer)
screen = None
RED = (255, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
BLUE = (70, 130, 180)
ORANGE = (255, 165, 13)
BUFFER = 8
font = None
# ----------------animation stuff--------------

def reset():
  global numRows
  global numCols
  global snake
  global apple
  global v_x
  global v_y
  global dead
  snake = [((int) (random() * numCols), (int) (random() * numRows))]
  apple = (int) (random() * numCols), (int) (random() * numRows)
  v_x, v_y = 1, 0 
  dead = False

def simulate_headless(net):
  scores = []
  for _ in range(NUM_ITERS):
    reset()
    last_ate_apple = 0
    t = 0
    while True:
      if dead:
        break
      if t - last_ate_apple > MIN_TIME_TO_EAT_APPLE:
        break
      
      sensory_vector = get_sensory()
      activation = net.activate(sensory_vector)
      action = np.argmax(activation)
      change_direction(action)
      apple = step()
      t += 1

      if apple:
        last_ate_apple = t

    scores.append(len(snake))

  return np.mean(scores)

def feed_forward_layers(inputs, outputs, connections, genome):
  """
  Modify neat-python's function to display more hidden nodes 
  """
  required = set(genome.nodes)

  layers = []
  s = set(inputs)
  while 1:
      # Find candidate nodes c for the next layer.  These nodes should connect
      # a node in s to a node not in s.
      c = set(b for (a, b) in connections if a in s and b not in s)
      # Keep only the used nodes whose entire input set is contained in s.
      t = set()
      for n in c:
          if n in required and all(a in s for (a, b) in connections if b == n):
              t.add(n)

      if not t:
          break

      layers.append(t)
      s = s.union(t)

  return layers

def modify_eval_functions(net, genome, config):
  """
  Modify neat-python's function to display more hidden nodes 
  """
  # Gather expressed connections.
  connections = [cg.key for cg in genome.connections.values() if cg.enabled]

  layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections, genome)
  node_evals = []
  for layer in layers:
      for node in layer:
          inputs = []
          for conn_key in connections:
              inode, onode = conn_key
              if onode == node:
                  cg = genome.connections[conn_key]
                  inputs.append((inode, cg.weight))

          ng = genome.nodes[node]
          aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
          activation_function = config.genome_config.activation_defs.get(ng.activation)
          node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))
  
  net.node_evals = node_evals

def simulate_animation(net, genome, config):
  global screen
  global font
  reset()

  modify_eval_functions(net, genome, config)
  has_eval = set(eval[0] for eval in net.node_evals)

  has_input = set(con[1] for con in genome.connections)

  hidden_nodes = [node for node in genome.nodes if not 0 <= node <= 3 and node in has_input and node in has_eval]

  node_centers = get_node_centers(net, genome, hidden_nodes)

  last_ate_apple = 0
  screen = pygame.display.set_mode((screenWidth, screenHeight))
  STEP = pygame.USEREVENT + 1
  pygame.time.set_timer(STEP, interval)

  pygame.init()
  font = pygame.font.Font("cmunbtl.otf", 24)
  running = True
  ts = 0
  while running:
    if dead:
      running = False
    if ts - last_ate_apple > MIN_TIME_TO_EAT_APPLE:
      running = False
    
    for event in pygame.event.get():
      if (event.type == pygame.QUIT):
        running = False
      elif (event.type == STEP):
        sensory_vector = get_sensory()
        activation = net.activate(sensory_vector)
        action = np.argmax(activation)
        change_direction(action)
        apple = step()
        if apple:
          last_ate_apple = ts

    screen.fill(BLACK)
    draw_square() 
    draw_snake() 
    draw_apple() 
    draw_network(net, genome, node_centers, hidden_nodes)
    pygame.display.flip()
  pygame.quit()

def get_node_centers(net, genome, hidden_nodes):
  
  node_centers = {}

  startY = window_buffer + NODE_SIZE
  startX = window_buffer

  for i, input_node in enumerate(net.input_nodes):
    center2 = startX + 5.5 * NODE_SIZE, startY + i * 3 * NODE_SIZE + 10
    node_centers[input_node] = center2

  startY = window_buffer + NODE_SIZE
  startX = window_buffer

  startX = window_buffer + 0.5 * networkWidth
  startY = window_buffer + NODE_SIZE * 6

  for i, hidden_node in enumerate(hidden_nodes):
    x = startX + 2 * NODE_SIZE if i % 2 == 0 else startX - 2 * NODE_SIZE
    if i == 2: x += NODE_SIZE * 3
    center2 = x, startY + i * 5 * NODE_SIZE + 10
    node_centers[hidden_node] = center2


  startY = window_buffer + 12 * NODE_SIZE
  startX = screenWidth - gameWidth - window_buffer * 3 - NODE_SIZE

  for i, output_node in enumerate(net.output_nodes):
    center2 = startX - 2 * NODE_SIZE, startY + i * 3 * NODE_SIZE + 10
    node_centers[output_node] = center2

  return node_centers

def draw_connections(first_set, second_set, net, genome, node_centers):
  for first in first_set:
    for second in second_set:
      if (first, second) in genome.connections:
        start = node_centers[first]
        stop = node_centers[second]
        weight = genome.connections[(first, second)].weight
        color = BLUE if weight >= 0 else ORANGE

        surf = pygame.Surface((screenWidth, screenHeight), pygame.SRCALPHA)
        alpha = 255 * (0.3 + net.values[first] * 0.7)
        pygame.draw.line(surf, color + (alpha,), start, stop, width=5)
        screen.blit(surf, (0, 0))

def draw_network(net, genome, node_centers, hidden_nodes):

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

  # draw connections between input and output nodes
  draw_connections(net.input_nodes, net.output_nodes, net, genome, node_centers)
  draw_connections(net.input_nodes, hidden_nodes, net, genome, node_centers)
  draw_connections(hidden_nodes, hidden_nodes, net, genome, node_centers)
  draw_connections(hidden_nodes, net.output_nodes, net, genome, node_centers)

  # draw input nodes
  for i, input_node in enumerate(net.input_nodes):
    center = node_centers[input_node]

    center2 = center[0] - 5.5 * NODE_SIZE, center[1] - 10
    img = font.render(node_names[input_node], True, WHITE)
    screen.blit(img, center2)

    color = (net.values[input_node] * 255, 0, 0)

    pygame.draw.circle(screen, color, center, NODE_SIZE)
    pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=5)

  # draw output nodes
  for i, output_node in enumerate(net.output_nodes):
    center = node_centers[output_node]
    color = (net.values[output_node] * 255, 0, 0)
    pygame.draw.circle(screen, color, center, NODE_SIZE)
    pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=5)

    center2 = center[0] + 1.5 * NODE_SIZE, center[1] - 10
    img = font.render(node_names[output_node], True, WHITE)
    screen.blit(img, center2)

  # draw hidden nodes

  for hidden in hidden_nodes:
    center = node_centers[hidden]
    color = (net.values[hidden] * 255, 0, 0)

    # center2 = center[0] - 5.5 * NODE_SIZE, center[1] - 10
    # img = font.render(str(hidden), True, WHITE)
    # screen.blit(img, center2)

    pygame.draw.circle(screen, color, center, NODE_SIZE)
    pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=5)

def draw_snake():
  for i, (x, y) in enumerate(snake):
      rect = pygame.Rect(getLeftTop(x, y), (blockWidth - BUFFER * 2, blockHeight - BUFFER * 2))
      pygame.draw.rect(screen, YELLOW if i == len(snake) - 1 else WHITE, rect)

def get_sensory():
  x, y = snake[-1]

  # inverted distance to wall
  # d_N, d_S, d_E, d_W
  dist_to_wall = [1 / (y + 1), 1 / (numRows - y), 1 / (numCols - x), 1 / (x + 1)]

  # flag for if will hit tail in this cardinal direction
  will_hit_tail = [0, 0, 0, 0]

  for (body_x, body_y) in snake[:-1]:
    if body_x == x:
      if body_y > y:
        will_hit_tail[1] = 1
      else:
        will_hit_tail[0] = 1
    elif body_y == y:
      if body_x > x:
        will_hit_tail[2] = 1
      else:
        will_hit_tail[3] = 1

  # apple
  a_x, a_y = apple

  apple_info = [
    x == a_x and a_y < y,
    x == a_x and a_y > y,
    y == a_y and a_x > x,
    y == a_y and a_x < x,
  ]

  # return 1.0 * np.array(dist_to_wall + will_hit_tail)
  return 1.0 * np.array(dist_to_wall + will_hit_tail + apple_info)

def change_direction(code):
  global v_x
  global v_y
  assert(0 <= code <= 3)

  # wasd
  if code == 0:
    v_x = 0
    v_y = -1
  elif code == 1:
    v_x = -1
    v_y = 0
  elif code == 2:
    v_x = 0
    v_y = 1
  else:
    v_x = 1
    v_y = 0 

def step():
  global apple
  global dead

  ate_apple = False

  x, y = snake[-1]
  snake.append((x + v_x, y + v_y))
  x, y = snake[-1]

  # hit wall
  if x < 0 or x >= numCols or y < 0 or y >= numRows:
    dead = True

  # hit body
  for s in snake[:-1]:
    if s == snake[-1]:
      dead = True
      break

  if not snake[-1] == apple:
    snake.pop(0)
  else:
    apple = (int) (random() * numCols), (int) (random() * numRows)
    ate_apple = True

  return ate_apple

def draw_square():
  draw = gameTopLeft[0] - BUFFER, gameTopLeft[1] - BUFFER
  rect = pygame.Rect(draw, (gameWidth + 2 * BUFFER, gameHeight + 2 * BUFFER))
  pygame.draw.rect(screen, WHITE, rect, width=BUFFER // 2)

def getLeftTop(x, y):
    return (x / numRows) * gameWidth + BUFFER + gameTopLeft[0], (y / numRows) * gameHeight + BUFFER + gameTopLeft[1]

def draw_apple():
  x, y = apple
  rect = pygame.Rect(getLeftTop(x, y), (blockWidth - BUFFER * 2, blockHeight - BUFFER * 2))
  pygame.draw.rect(screen, RED, rect)