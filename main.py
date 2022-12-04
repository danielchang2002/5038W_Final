from random import random
import numpy as np
import pygame

# ----------------simulation-----------------
numRows, numCols = 10, 10
snake = [(3, 3), (3, 4)]
apple = (5, 5)
v_x, v_y = 1, 0 
dead = False
NUM_ITERS = 10
MAX_TIME_STEPS = 1000
MIN_TIME_TO_EAT_APPLE = 20
# ----------------simulation-----------------

# ----------------animation stuff--------------
interval = 100
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
  # snake = [(10, 10), (10, 11)]
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
  scores = []
  global screen
  for _ in range(1):
    reset()
    last_ate_apple = 0
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    STEP = pygame.USEREVENT + 1
    pygame.time.set_timer(STEP, interval)

    pygame.init()
    running = True
    ts = 0
    while running:
      ts += 1
      if dead:
        running = False
      # if ts - last_ate_apple > MIN_TIME_TO_EAT_APPLE:
      #   break
      
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

      # if ts == MAX_TIME_STEPS:
      #   running = False
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

def get_sensory_old():
  x, y = snake[-1]

  # N, S, E, W
  not_blocked_by_wall = [y > 0, y < numRows, x < numCols, x > 0]

  not_blocked_by_body = [1, 1, 1, 1]

  for (body_x, body_y) in snake[:-1]:
    if body_x == x and body_y == y - 1:
      not_blocked_by_body[1] = 0
    elif body_x == x and body_y == y + 1:
      not_blocked_by_body[0] = 0
    elif body_y == y and body_x == x + 1:
      not_blocked_by_body[2] = 0
    elif body_y == y and body_x == x - 1:
      not_blocked_by_body[3] = 0

  # fruit in direction
  a_x, a_y = apple
  fruit_in_direction = [
    a_y < y,
    a_y > y,
    a_x > x,
    a_x < x
  ]

  not_blocked = [a and b for a, b in zip(not_blocked_by_wall, not_blocked_by_body)]

  return np.array(not_blocked + fruit_in_direction) * 1
  # return np.array(fruit_in_direction) * 1

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

  ate_apple = False

  x, y = snake[-1]
  snake.append((x + v_x, y + v_y))

  if not snake[-1] == apple:
    snake.pop(0)
  else:
    apple = (int) (random() * numCols), (int) (random() * numRows)
    ate_apple = True

  x, y = snake[-1]

  # hit wall
  if x < 0 or x >= numCols or y < 0 or y >= numRows:
    dead = True

  # hit body
  for s in snake[:-1]:
    if s == snake[-1]:
      dead = True
      break



  return ate_apple

  

def getLeftTop(x, y):
    return (x / numRows) * screenWidth, (y / numRows) * screenHeight

def draw_apple():
  x, y = apple
  rect = pygame.Rect(getLeftTop(x, y), (blockWidth, blockHeight))
  pygame.draw.rect(screen, RED, rect)

if __name__ == "__main__":
  main()