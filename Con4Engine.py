import random

'''
Responsible for keeping track of the current games state. 
'''
class GameState():
    def __init__(self):
        self.board = [
            ["-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-"]]
        self.playerMove = True

    '''
    If move is invalid will return false.
    ''' 
    def makeMove(self, row, col, value):
        cellValue = self.board[row][col]
        if cellValue == '-':
            if row == len(self.board[0]) - 1:
                self.board[row][col] = value
                return True
            elif self.board[row + 1][col] != '-':
                self.board[row][col] = value
                return True
            
        return False
    
    '''
    Checks for winner, probably isnt the most optimal can revist later.
    '''
    def checkForWinner(self):
        rows, cols = len(self.board), len(self.board[0])

        # Function to check the sequence starting from (start_row, start_col) in direction (dx, dy)
        def checkDirection(start_row, start_col, dx, dy, player):
            count = 0
            row, col = start_row, start_col
            while 0 <= row < rows and 0 <= col < cols and self.board[row][col] == player:
                count += 1
                row += dx
                col += dy
            return count >= 4

        for r in range(rows):
            for c in range(cols):
                if self.board[r][c] != "-":  # If the cell is not empty
                    player = self.board[r][c]
                    # Check horizontal, vertical, and both diagonal directions
                    if checkDirection(r, c, 0, 1, player) or \
                    checkDirection(r, c, 1, 0, player) or \
                    checkDirection(r, c, 1, 1, player) or \
                    checkDirection(r, c, 1, -1, player):
                        return player  # Return the player ("X" or "O") who won

        return False  # If no winner, return None
        
    

class BasicAI():
    def __init__(self):
        self.value = "B"

    '''
    Uses a really inefficient way to get the move going, once AI is implemented it will be much more smoother.
    '''
    def move(self, state):
        row = random.randint(0, len(state.board) - 1)
        col = random.randint(0, len(state.board[0]) - 1)
        while not state.makeMove(row, col, self.value):
            row = random.randint(0, len(state.board) - 1)
            col = random.randint(0, len(state.board[0]) - 1)
