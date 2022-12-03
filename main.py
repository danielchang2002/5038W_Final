import pygame
from random import random
import numpy as np

# ----------------simulation-----------------
numRows, numCols = 20, 20
snake = [(10, 10), (10, 11)]
apple = (3, 3)
v_x, v_y = 1, 0 
dead = False
# ----------------simulation-----------------

# ----------------animation stuff--------------
interval = 1000
animation = True
screenWidth, screenHeight = 800, 800
blockWidth, blockHeight = screenWidth / numRows, screenHeight / numRows
screen = None
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# ----------------animation stuff--------------

def main():

  if animation:
    global screen
    global v_x
    global v_y

    screen = pygame.display.set_mode((screenWidth, screenHeight))
    STEP = pygame.USEREVENT + 1
    pygame.time.set_timer(STEP, interval)

    pygame.init()
    running = True
    while running:
      if dead:
        break
      
      for event in pygame.event.get():
        if (event.type == pygame.QUIT):
          running = False
        if (event.type == STEP):
          step()
          get_sensory()
        elif (event.type == pygame.KEYDOWN):
          if event.key == pygame.K_w:
              v_y = -1
              v_x = 0
          elif event.key == pygame.K_a:
              v_y = 0
              v_x = -1
          elif event.key == pygame.K_s:
              v_y = 1
              v_x = 0
          elif event.key == pygame.K_d:
              v_y = 0
              v_x = 1
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

  d_N_wall, d_S_wall, d_E_wall, d_W_wall = y, numRows - y, numCols - x, x

  for (body_x, body_y) in snake[:-1]:
    if body_x == x:
      if body_y > y:
        d_S = min(d_S, body_y - y)
      else:
        d_N = min(d_N, y - body_y)
    elif body_y == y:
      if body_x > x:
        d_E = min(d_E, body_x - x)
      else:
        d_W = min(d_W, x - body_x)


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

  

def getLeftTop(x, y):
    return (x / numRows) * screenWidth, (y / numRows) * screenHeight

def draw_apple():
  x, y = apple
  rect = pygame.Rect(getLeftTop(x, y), (blockWidth, blockHeight))
  pygame.draw.rect(screen, RED, rect)

if __name__ == "__main__":
  main()