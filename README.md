# Connect-4 AI Game 
## Introduction
Connect-4 is a classic two-player strategy game where players alternate turns dropping colored discs into a seven-column, six-row vertically suspended grid. 

The objective is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs. 

Connect-4 is a solved game, with the first player being able to force a victory with perfect play. 

This project implements an AI to play Connect-4 using a neural network to evaluate game states, making it a challenging opponent for human players.

## Features

- AI that evaluates game states through a trained neural network.
- Strategic game play, requiring players to think both offensively and defensively.
- Graphical user interface using Pygame to play against the AI.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Pygame
- Pandas
- NumPy

### Installation

1. Clone the repository:

2. Install the required packages:
   ```bash
   pip install torch 
   pip install pygame
   pip install pandas
   pip install numpy
   ```

### Training the AI
Before playing against the AI, ensure that the neural network model is trained:

  1. Run Con4Train
     ```bash
     python3 Con4Train.py
     ```
This will train the neural network and save the model's state for use by the game AI.

### Launch the Game
To start the game, run the Con4Main.py script after training the AI:

  1. Run Con4Train
     ```bash
     python3 Con4Main.py
     ```
