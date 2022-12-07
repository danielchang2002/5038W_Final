from random import random
import numpy as np
import pygame

# ----------------simulation-----------------
numRows, numCols = 10, 10
snake = [(3, 3)]
apple = (5, 5)
v_x, v_y = 1, 0 
dead = False
NUM_ITERS = 10
MAX_TIME_STEPS = 1000
MIN_TIME_TO_EAT_APPLE = 200
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
    for t in range(MAX_TIME_STEPS):
      if dead:
        break
      if t - last_ate_apple > MIN_TIME_TO_EAT_APPLE:
        break
      
      sensory_vector = get_sensory()
      activation = net.activate(sensory_vector)
      action = np.argmax(activation)
      change_direction(action)
      apple = step()
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
    if ts == MAX_TIME_STEPS:
      running = False
    
    for event in pygame.event.get():
      if (event.type == pygame.QUIT):
        running = False
      elif (event.type == STEP):
        ts += 1
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
    pygame.display.flip()
  pygame.quit()

def draw_snake():
  for i, (x, y) in enumerate(snake):
      rect = pygame.Rect(getLeftTop(x, y), (blockWidth - BUFFER * 2, blockHeight - BUFFER * 2))
      pygame.draw.rect(screen, YELLOW if i == len(snake) - 1 else WHITE, rect)

def get_sensory():
  x, y = snake[-1]

  # distance to wall
  # d_N, d_S, d_E, d_W = y, numRows - y, numCols - x, x
  dist_to_wall = [y, numRows - y, numCols - x, x]

  # distance to tail (or just distance to wall if the cardinal dir doesn't hit tail)
  dist_to_tail = [y, numRows - y, numCols - x, x]

  for (body_x, body_y) in snake[:-1]:
    if body_x == x:
      if body_y > y:
        dist_to_tail[1] = min(dist_to_tail[1], body_y - y)
      else:
        dist_to_tail[0] = min(dist_to_tail[0], y - body_y)
    elif body_y == y:
      if body_x > x:
        dist_to_tail[2] = min(dist_to_tail[2], body_x - x)
      else:
        dist_to_tail[3] = min(dist_to_tail[3], x - body_x)

  return np.array(dist_to_wall + dist_to_tail) / numRows
  # # apple
  # a_x, a_y = apple

  # apple_info = [abs(a_x - x) + abs(a_y - y), a_x - x, y - a_y]

  # return np.array(dist_to_wall + dist_to_tail + apple_info)

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

  x, y = snake[-1]

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