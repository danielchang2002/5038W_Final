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
networkWidth, networkHeight = 500, 900
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
BUFFER = 8
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

def simulate_animation(net):
  global screen
  reset()
  last_ate_apple = 0
  screen = pygame.display.set_mode((screenWidth, screenHeight))
  STEP = pygame.USEREVENT + 1
  pygame.time.set_timer(STEP, interval)

  pygame.init()
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
    draw_network(net)
    pygame.display.flip()
  pygame.quit()

def draw_network(net):
  print(net.input_nodes)
  print(net.output_nodes)

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
  rect = pygame.Rect(gameTopLeft, (gameWidth, gameHeight))
  pygame.draw.rect(screen, WHITE, rect, width=BUFFER // 2)

def getLeftTop(x, y):
    return (x / numRows) * gameWidth + BUFFER + gameTopLeft[0], (y / numRows) * gameHeight + BUFFER + gameTopLeft[1]

def draw_apple():
  x, y = apple
  rect = pygame.Rect(getLeftTop(x, y), (blockWidth - BUFFER * 2, blockHeight - BUFFER * 2))
  pygame.draw.rect(screen, RED, rect)