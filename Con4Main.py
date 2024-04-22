'''
Handles the setting up and client interaction as well as the main game loop of connect 4.
'''

import pygame as pg
import Con4Engine

WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15

'''
Main driver handles input and updating graphics
'''
def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()

    state = Con4Engine.GameState()
    ai = Con4Engine.BasicAI()

    running = True
    playerTurn = True
    while running:
        if playerTurn:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running = False
                elif e.type == pg.MOUSEBUTTONDOWN:
                    location = pg.mouse.get_pos()
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    if state.makeMove(row, col, "R"):
                        # If not valid move just repeats
                        playerTurn = False
        else:
            # Ai makes move
            ai.move(state)
            playerTurn = True

        updateGameState(screen, state.board)

        # Check winner
        winner = state.checkForWinner()
        if winner:
            print("Winner: " + winner)
            running = False

        clock.tick(MAX_FPS)
        pg.display.flip()

def updateGameState(screen, board): 
    drawBoard(screen)
    drawPieces(screen, board)

def drawBoard(screen):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            pg.draw.rect(screen, pg.Color("Grey"), pg.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
            pg.draw.rect(screen, pg.Color("Black"), pg.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE), 2)

def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = None
            cellValue = board[r][c]
            if cellValue == 'R':
                color = (255, 0, 0)
            elif cellValue == 'B':
                color = (0, 0, 255)
            else:
                continue

            pg.draw.circle(screen, color, (c * SQ_SIZE + (SQ_SIZE / 2), r * SQ_SIZE + (SQ_SIZE / 2)), 25)
main()


