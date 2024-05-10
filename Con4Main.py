'''
Handles the setting up and client interaction as well as the main game loop of connect 4.
'''

import pygame as pg
from Con4Engine import GameState as gs
from Con4Engine import BasicAI

WIDTH = 700
HEIGHT = 600
ROWS = 6
COLUMNS = 7
SQ_SIZE = HEIGHT // ROWS
MAX_FPS = 15

EASY = 1
MEDIUM = 3
HARD = 5

'''
Main driver handles input and updating graphics
'''
def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()

    ai = BasicAI(HARD)

    running = True
    player_turn = True
    while running:
        if player_turn:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running = False
                elif e.type == pg.MOUSEBUTTONDOWN:
                    location = pg.mouse.get_pos()
                    col = location[0] // SQ_SIZE
                    if gs.is_valid_move(col):
                        gs.make_move(col, -1)
                        player_turn = False
        else:
            # Ai makes move
            ai.move()
            player_turn = True

        update_game_state(screen)

        # Check winners
        winner = gs.check_for_winner()
        if winner:
            pg.display.flip() # Display update
            if winner == 1:
                winner = "AI"
            else:
                winner = "Player"
            print("Winner: " + winner)

            pg.time.delay(3500) # Give player a chance to see winning move
            running = False

        # Check for draw
        if gs.is_finished():
            running = False
            
        clock.tick(MAX_FPS)
        pg.display.flip()

def update_game_state(screen): 
    draw_board(screen)
    draw_pieces(screen)

def draw_board(screen):
    for r in range(ROWS):
        for c in range(COLUMNS):
            pg.draw.rect(screen, pg.Color("Grey"), pg.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            pg.draw.rect(screen, pg.Color("Black"), pg.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE), 2)

def draw_pieces(screen):
    for r in range(ROWS):
        for c in range(COLUMNS):
            color = None
            cell_value = gs.board[r][c]
            if cell_value == -1:
                color = (255, 0, 0)
            elif cell_value == 1:
                color = (0, 0, 255)
            else:
                continue

            pg.draw.circle(screen, color, (c * SQ_SIZE + (SQ_SIZE / 2), r * SQ_SIZE + (SQ_SIZE / 2)), 25)
main()
