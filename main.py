from random import random
import numpy as np
import pygame

# ----------------simulation-----------------
numRows, numCols = 20, 20
snake = [(10, 10), (10, 11)]
apple = (3, 3)
v_x, v_y = 1, 0 
dead = False
NUM_ITERS = 10
MAX_TIME_STEPS = 1000
# ----------------simulation-----------------

# ----------------animation stuff--------------
interval = 1000
animation = False
screenWidth, screenHeight = 800, 800
blockWidth, blockHeight = screenWidth / numRows, screenHeight / numRows
screen = None
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# ----------------animation stuff--------------

def reset():
  global numRows
  global numCols
  global snake
  global apple
  global v_x
  global v_y
  global dead
  numRows, numCols = 20, 20
  snake = [(10, 10), (10, 11)]
  apple = (3, 3)
  v_x, v_y = 1, 0 
  dead = False

def simulate_headless(net):
  scores = []
  for _ in range(NUM_ITERS):
    reset()
    penalty = 0
    for t in range(MAX_TIME_STEPS):
      if dead:
        break
      if penalty >= 10:
        break

      step()
      sensory_vector = get_sensory()

      activation = net.activate(sensory_vector)
      action = np.argmax(activation)
      change_direction(action)

    scores.append(len(snake) - penalty)
  return np.mean(scores)

def simulate_animation(net):
  scores = []
  global screen
  for _ in range(NUM_ITERS):
    reset()
    ts = 0
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    STEP = pygame.USEREVENT + 1
    pygame.time.set_timer(STEP, 100)

    pygame.init()
    running = True
    while running:
      print(get_sensory())
      if dead:
        running = False
      
      for event in pygame.event.get():
        if (event.type == pygame.QUIT):
          running = False
        elif (event.type == STEP):
          step()
          sensory_vector = get_sensory()
          activation = net.activate(sensory_vector)
          action = np.argmax(activation)
          change_direction(action)
          ts += 1
      if ts == MAX_TIME_STEPS:
        running = False
      screen.fill(BLACK)
      draw_snake() 
      draw_apple() 
      pygame.display.flip()
    pygame.quit()
    scores.append(len(snake))
  return np.mean(scores)

def simulate(net):
  """
  Returns fitness for network 
  """
  if animation: return simulate_animation(net)
  return simulate_headless(net)

def main():
  if animation:
    global screen
    global v_x
    global v_y

    screen = pygame.display.set_mode((screenWidth, screenHeight))
    # STEP = pygame.USEREVENT + 1
    # pygame.time.set_timer(STEP, interval)

    pygame.init()
    running = True
    while running:
      if dead:
        break
      
      for event in pygame.event.get():
        if (event.type == pygame.QUIT):
          running = False
        # if (event.type == STEP):
        #   step()
        #   sensory_vector = get_sensory()
        #   print(sensory_vector)
        elif (event.type == pygame.KEYDOWN):
          move = False
          if event.key == pygame.K_w:
              move = True
              change_direction(0)
          elif event.key == pygame.K_a:
              move = True
              change_direction(1)
          elif event.key == pygame.K_s:
              move = True
              change_direction(2)
          elif event.key == pygame.K_d:
              move = True
              change_direction(3)
          if move:
            step()
            sensory_vector = get_sensory()
            print(sensory_vector)

      screen.fill(BLACK)
      draw_snake() 
      draw_apple() 
      pygame.display.flip()
    pygame.quit()

def draw_snake():
  for (x, y) in snake:
      rect = pygame.Rect(getLeftTop(x, y), (blockWidth, blockHeight))
      pygame.draw.rect(screen, WHITE, rect)

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

  # apple
  a_x, a_y = apple

  apple_info = [abs(a_x - x) + abs(a_y - y), a_x - x, y - a_y]

  return np.array(dist_to_wall + dist_to_tail + apple_info)

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

  x, y = snake[-1]
  snake.append((x + v_x, y + v_y))

  if not snake[-1] == apple:
    snake.pop(0)
  else:
    apple = (int) (random() * numCols), (int) (random() * numRows)

  x, y = snake[-1]

  # hit wall
  if x < 0 or x >= numCols or y < 0 or y >= numRows:
    dead = True

  # hit body
  for s in snake[:-1]:
    if s == snake[-1]:
      dead = True
      break

  return snake[-1] == apple

  

def getLeftTop(x, y):
    return (x / numRows) * screenWidth, (y / numRows) * screenHeight

def draw_apple():
  x, y = apple
  rect = pygame.Rect(getLeftTop(x, y), (blockWidth, blockHeight))
  pygame.draw.rect(screen, RED, rect)

if __name__ == "__main__":
  main()