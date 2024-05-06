
from Con4Networks import AIEvaluator

'''
Responsible for keeping track of the current games state. 
To remain aligned with our dataset -1 denotes the player's pos 
0 denotes an empty space and 1 denotes the ai's pos.
'''
class GameState():

    board = [[0] * 7 for _ in range(6)]
    player_move = True

    def is_valid_move(col):
        return GameState.board[0][col] == 0

    def undo_move(row, col):
        GameState.board[row][col] = 0

    '''
    Assumes if validation of move is taken care of before call.
    Returns the index of the free row in which the piece is placed.
    ''' 
    def make_move(col, player):
        for row in reversed(range(6)):
            if GameState.board[row][col] == 0:
                GameState.board[row][col] = player
                return row
        return -1 # Theoretically will never be reached
    
    '''
    Checks for winner, probably isnt the most optimal can revist later.
    '''
    def check_for_winner():
        rows, cols = len(GameState.board), len(GameState.board[0])

        # Function to check the sequence starting from (start_row, start_col) in direction (dx, dy)
        def check_direction(start_row, start_col, dx, dy, player):
            count = 0
            row, col = start_row, start_col
            while 0 <= row < rows and 0 <= col < cols and GameState.board[row][col] == player:
                count += 1
                row += dx
                col += dy
            return count >= 4

        for r in range(rows):
            for c in range(cols):
                if GameState.board[r][c] != 0:  # If the cell is not empty
                    player = GameState.board[r][c]
                    # Check horizontal, vertical, and both diagonal directions
                    if check_direction(r, c, 0, 1, player) or \
                    check_direction(r, c, 1, 0, player) or \
                    check_direction(r, c, 1, 1, player) or \
                    check_direction(r, c, 1, -1, player):
                        return player  # Return the player (-1 or 1) who won

        return False  # If no winner, return None
    
    '''
    Checks if the game of connect 4 is over.
    '''
    def is_finished():
        # If winner then game over
        if GameState.check_for_winner():
            return True
        
        # Check if board is filled
        for col in range(7):
            if GameState.is_valid_move(col):
                return True
        return False
    
        
    

class BasicAI():

    def __init__(self, max_depth):
        self.value = 1
        self.max_depth = max_depth # main factor of how intelligent our AI is

        # Instantializing up NN
        eval = AIEvaluator()
        eval.load_data()
        eval.train()
        self.evaluator = eval

    '''
    Sends current board state to neural network and gets evaluation back.
    '''
    def evaluate_board(self, board):
        # Re-sizing board to match shape of Neural Network input
        flatten_board = []
        for sub_arr in board:
            for item in sub_arr:
                flatten_board.append(item)

        return self.evaluator.evaluate_pos(flatten_board)

    def minimax(self, depth, player):
        if depth == 0 or GameState.is_finished():
            return self.evaluate_board(GameState.board)

        # Maximizing
        if player == 1:
            score = float('-inf')
            # Going down each valid move
            for col in range(7):
                if GameState.is_valid_move(col):
                    row = GameState.make_move(col, player)
                    score = max(score, self.minimax(depth - 1, player * -1)) # Going down tree and getting max score
                    GameState.undo_move(row, col)
            return score
        # Minimizing
        else: 
            score = float('inf')
            for col in range(7):
                if GameState.is_valid_move(col):
                    row = GameState.make_move(col, player)
                    score = min(score, self.minimax(depth - 1, player * -1)) # Going down tree and getting min score
                    GameState.undo_move(row, col)
            return score
    

    '''
    Makes the best move (capped by NN) by sending each possible next position into a minmax
    and using the most favorable next move.
    '''
    def move(self):
        max_score = float('-inf')
        chosen_col = -1

        for col in range(7):
            # Sends each valid move into minimax
            if GameState.is_valid_move(col):
                row = GameState.make_move(col, self.value)
                score = self.minimax(self.max_depth - 1, self.value * -1)
                GameState.undo_move(row, col)
                # Keep track of best score and col
                if score > max_score:
                    chosen_col = col
                    max_score = score

        GameState.make_move(chosen_col, self.value)
