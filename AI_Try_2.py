full_memory = []

import mss
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import random
import keyboard
import time
import pytesseract
import pyautogui
import screeninfo
import os
import pickle
import collections
from collections import deque

from PIL import Image

#Start Game
window = pyautogui.getActiveWindow()
if window:
    x, y, width, height = window.left, window.top, window.width, window.height
    pyautogui.click(x + width - 110, y + 15)

time.sleep(2)
pyautogui.doubleClick(50, 120)

time.sleep(10)
pyautogui.moveTo(600,420)

time.sleep(2)
pyautogui.click(600, 420)

def is_valid_move(grid, piece, x_offset, y_offset):
    for y in range(piece.shape[0]):
        for x in range(piece.shape[1]):
            if piece[y, x] == 1:
                new_x = x + x_offset
                new_y = y + y_offset
                if new_x < 0 or new_x >= grid.shape[1] or new_y >= grid.shape[0]:
                    return False
                if grid[new_y, new_x] == 1:
                    return False
    return True

def get_drop_position(grid, piece):
    for y in range(grid.shape[0] - 1, -1, -1):
        if is_valid_move(grid, piece, 0, y):
            return y
    return 0

referenceColor = None

def capture_screen():
    with mss.mss() as sct:
        try:
            screen = sct.grab(sct.monitors[1])  # Capture full screen
            img = np.array(screen)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            if img is None or img.size == 0:
                print("Error: Full image capture failed")
                return None, None

            board_x, board_y, board_w, board_h = 740, 90, 440, 900  # Adjusted for full capture
            img = img[board_y:board_y+board_h, board_x:board_x+board_w]

            piece_capture_y_offset = 60
            piece_capture_area = img[0:piece_capture_y_offset]
            grid_capture_area = img[piece_capture_y_offset:, :]

            # Debugging: Save screenshots to check if capturing correctly
            cv2.imwrite("debug_grid_capture.png", grid_capture_area)
            cv2.imwrite("debug_piece_capture.png", piece_capture_area)

            print("Captured screen successfully")

            return grid_capture_area, piece_capture_area

        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None, None

def get_background_color():
    grid_capture_area, _ = capture_screen()

    if grid_capture_area is None:
        print("Error capturing screen for background color.")
        return np.array([0, 0, 0])
    
    background_color = np.mean(grid_capture_area[0:10, 0:10], axis=(0, 1))
    return background_color

def detect_game_over(current_color, previous_color, threshold=10):
    color_diff = np.abs(current_color - previous_color)
    
    if np.all(color_diff > threshold):
        return True
    return False

def ocr_capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Capture the primary screen
        screenshot = np.array(sct.grab(monitor))

        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to improve OCR accuracy
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR
        text = pytesseract.image_to_string(thresh)

        return text.upper()  # Convert to uppercase for easier detection
    

def get_tetris_templates():
    templates = {
        'I': np.array([[1, 1, 1, 1]]),  # Horizontal 4-block line
        'O': np.array([[1, 1], [1, 1]]),  # 2x2 square
        'T': np.array([[0, 1, 0], [1, 1, 1]]),  # T-shape
        'S': np.array([[0, 1, 1], [1, 1, 0]]),  # S-shape
        'Z': np.array([[1, 1, 0], [0, 1, 1]]),  # Z-shape
        'J': np.array([[1, 0, 0], [1, 1, 1]]),  # J-shape
        'L': np.array([[0, 0, 1], [1, 1, 1]])   # L-shape
    }
    return templates

def get_piece_from_image(image):
    templates = get_tetris_templates()
    
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours detected.")
        return None
    
    best_match = None
    max_overlap = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        piece = thresh[y:y+h, x:x+w]
        
        for piece_name, template in templates.items():
            match = cv2.matchTemplate(piece, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(match)
            if max_val > max_overlap:
                max_overlap = max_val
                best_match = piece_name
    
    return best_match

def extract_board(img):
    if img is None:
        print("Error: Received None as input image.")
        return None

    if isinstance(img, tuple):
        img = img[0]  # Extract actual image if it's a tuple

    if not isinstance(img, np.ndarray):
        print("Error: Image is not a valid NumPy array.")
        return None

    if img.size == 0:
        print("Error: Image is empty.")
        return None

    # Convert to grayscale if it's a color image
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img  # Already grayscale

    # Ensure the grayscale image is in the correct format (8-bit)
    if gray_img.dtype != np.uint8:
        gray_img = np.uint8(gray_img)  # Convert to 8-bit if not already

    # Apply adaptive thresholding to make the board more distinguishable
    thresh_img = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    if thresh_img is None or thresh_img.size == 0:
        print("Error: Thresholded image is empty.")
        return None

    # Optionally, apply edge detection (Canny) for more clear distinction of pieces
    edges = cv2.Canny(thresh_img, 100, 200)

    return edges  # You can return thresh_img or edges depending on your needs

def preprocess_image(img):
    img = cv2.resize(img, (10, 20))
    _,img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return img.flatten() / 255.0

def get_state():
    grid_capture_area, _ = capture_screen()

    if grid_capture_area is None:
        print("Error: Failed to capture grid")
        return torch.zeros((10, 20))  # Return an empty grid instead of exiting

    grid = extract_board(grid_capture_area)

    if grid is None:
        print("Error: Extracted grid is None")
        return torch.zeros((10, 20))

    grid_tensor = torch.tensor(grid.flatten(), dtype=torch.float32)  # Flatten for NN

    # Debugging: Print state shape and some values
    print(f"State shape: {grid_tensor.shape}")
    print(f"State sample: {grid_tensor[:10]}")  # Show first 10 elements

    return grid_tensor

game_over_detected = False

def is_game_over():
    global game_over_detected

    if game_over_detected:
        return True
    
    grid_capture_area, _ = capture_screen()

    if grid_capture_area is None:
        print("Error: Screen Capture Failed")
        return False

    try:
        _, img = cv2.threshold(grid_capture_area, 150, 255, cv2.THRESH_BINARY)
        cv2.imwrite("game_over_debug.png", img)  # Save for debugging
        
        text = pytesseract.image_to_string(img).upper().strip()
        print(f"OCR Detected: {text}")

        if "YOU DIED" in text or "GAME OVER" in text:  # More robust detection
            print("Game Over Detected")
            if not game_over_detected:
                find_and_click_retry()
            game_over_detected = True
            return True
    except Exception as e:
        print(f"Error during OCR detection: {e}")
    
    return False

def find_and_click_retry():
    global game_over_detected

    img = capture_screen()
    
    results = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    for i in range(len(results["text"])):
        if "RETRY" in results["text"][i].upper():
            x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
            
            screen_x = 81 + x + w // 2  
            screen_y = 727 + y + h // 2
            
            print(f"Clicking Retry at ({screen_x}, {screen_y})")
            pyautogui.click(screen_x, screen_y)
            time.sleep(2)
            game_over_detected = False
            return True
    return False

class TetrisDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TetrisDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

actions = {
    0: "a", #Left
    1: "d", #Right
    2: "e", #Rotate Counter-CLockwise
    3: "q", #Rotate Clockwise
    4: "s", #Down
    5: "space" #Harddrop
    }

input_dim = 10 * 20
output_dim = len(actions)

policy_net = TetrisDQN(input_dim, output_dim)
target_net = TetrisDQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

def send_action(action):
    if action in actions:
        print(f"Executing action: {actions[action]}")  # Debugging
        keyboard.press_and_release(actions[action])
    else:
        print(f"Invalid action: {action}")

model = TetrisDQN(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
memory = deque(maxlen=5000)
epsilon = 1.0
gamma = 0.95
tau = 0.01

REPLAY_MEMORY_SIZE = 10000
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) 
criterion = nn.MSELoss()

def save_full_memory(filename="full_memory.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(full_memory, f)

def load_full_memory(filename="full_memory.pkl"):
    global full_memory
    with open(filename, "rb") as f:
        full_memory = pickle.load(f)

def store_experience(state, action, reward, next_state):
    experience = (state, action, reward, next_state)
    replay_memory.append(experience) 
    full_memory.append(experience)

def sample_batch(batch_size=32):
    if len(replay_memory) < batch_size:
        return []
    return random.sample(replay_memory, batch_size)

def train(batch_size=32):
    """Trains the model using experiences from the replay buffer."""
    # Sample a batch of experiences from the replay buffer
    batch = sample_batch(batch_size)
    
    if not batch:
        return  # Skip if there's not enough data to sample

    # Prepare the batch for training
    states, actions, rewards, next_states = zip(*batch)
    
    # Convert to tensors
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)

    # Get the current Q values and next Q values
    q_values = model(states_tensor).gather(1, actions_tensor.view(-1, 1))  # Q(s, a)
    next_q_values = model(next_states_tensor).max(1)[0].detach()  # Q(s', a') max over next state

    # Compute the target Q values using Bellman equation
    target_q_values = rewards_tensor + (gamma * next_q_values)  # Q(s, a) = r + gamma * max_a Q(s', a')

    # Compute the loss
    loss = criterion(q_values.squeeze(), target_q_values)

    # Backpropagate and update the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Optionally, print the loss for monitoring
    print(f"Loss: {loss.item()}")

def count_holes(grid):
    holes = 0
    for col in range(grid.shape[1]):
        for row in range(grid.shape[0] - 1, -1, -1):
            if grid[row, col] == 1:
                break  
            elif grid[row, col] == 0:
                holes += 1
    return holes

def calculate_reward(prev_state, next_state):
    prev_state = prev_state.numpy() if isinstance(prev_state, torch.Tensor) else prev_state  # Convert to numpy array if it's a tensor
    print(f"Previous state shape: {prev_state.shape}")  # Print the shape for debugging

    # Resize or reshape based on actual size
    if prev_state.size == 600:
        prev_grid = prev_state.reshape(30, 20)  # Or any size that matches your grid
    elif prev_state.size == 200:
        prev_grid = prev_state.reshape(20, 10)  # Original grid size
    else:
        print("Unexpected state size!")
        return 0  # Handle the case of unexpected grid sizes

    new_state = next_state if isinstance(next_state, np.ndarray) else next_state.numpy() 
    print(f"Next state shape: {new_state.shape}") 

    if new_state.size == 600:
        new_grid = new_state.reshape(30, 20)
    elif new_state.size == 200:
        new_grid = new_state.reshape(20, 10)
    else:
        print("Unexpected next state size!")
        return 0 

    cleared_lines = np.sum(np.all(new_grid == 1, axis=1).astype(int))

    stack_height = np.max(np.where(new_grid == 1)[0]) if np.any(new_grid == 1) else 0

    holes = count_holes(new_grid)

    smoothness = np.sum(np.abs(np.diff(new_grid.sum(axis=1))))

    reward = cleared_lines - 0.5 * holes - 0.1 * stack_height - 0.05 * smoothness

    prev_stack_height = np.max(np.where(prev_grid == 1)[0]) if np.any(prev_grid == 1) else 0
    prev_holes = count_holes(prev_grid)
    
    if prev_stack_height < stack_height: 
        reward -= 0.2 * (stack_height - prev_stack_height)
    
    if prev_holes < holes: 
        reward -= 0.3 * (holes - prev_holes)

    return reward

class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        return batch

    def clear(self):
        self.buffer.clear()

    def size(self):
        return len(self.buffer)
    

def get_piece_position(piece_image):
    contours, _ = cv2.findContours(piece_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours,key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        return (x, y)
    
    return (0, 0)

def detect_screen_text():
    screen_text = ocr_capture_screen() 
    return screen_text

def handle_screen_state():
    screen_text = detect_screen_text()
    
    if "YOU Died" in screen_text:
        print("Detected Game Over screen, clicking RETRY...")
        find_and_click_retry()
    elif "PLAY" in screen_text:
        print("Detected Main Menu, clicking PLAY...")
        pyautogui.click(600, 420)
        time.sleep(2)

def get_current_piece():
    print("Getting current piece and position...")

    img = capture_screen()
    if img is None:
        print("Error: Failed to capture screen.")
        return None, None

    board = extract_board(img)
    if board is None or board.size == 0:
        print("Error: Board extraction failed.")
        return None, None

    print(f"Board shape: {board.shape}, dtype: {board.dtype}")

    if len(board.shape) == 3:
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    else:
        gray = board

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = None
    max_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area and y < board.shape[0] // 2:
            max_area = area
            max_contour = contour

    if max_contour is None or max_area < 100:
        print("Error: No valid falling piece detected.")
        return None, None

    x, y, w, h = cv2.boundingRect(max_contour)
    piece_position = (x, y)

    print(f"Detected piece at {piece_position} with size ({w}, {h})")

    piece = board[y:y+h, x:x+w]

    return piece, piece_position

def get_locked_pieces(previous_state, current_state):
    previous_board = extract_board(previous_state)
    current_board = extract_board(current_state)

    if previous_board is None or current_board is None:
        print("Error: Board extraction failed.")
        return False

    if not isinstance(previous_board, np.ndarray) or not isinstance(current_board, np.ndarray):
        print(f"Error: Boards are not NumPy arrays. Types received: {type(previous_board)}, {type(current_board)}")
        return False

    # Convert to grayscale if the board is in color
    if previous_board.shape[-1] == 3:
        previous_board = cv2.cvtColor(previous_board, cv2.COLOR_BGR2GRAY)
    if current_board.shape[-1] == 3:
        current_board = cv2.cvtColor(current_board, cv2.COLOR_BGR2GRAY)

    # Detect edges in both frames
    previous_edges = cv2.Canny(previous_board, 50, 150)
    current_edges = cv2.Canny(current_board, 50, 150)

    # Compare the edges of the two boards
    difference = cv2.absdiff(previous_edges, current_edges)

    # If the difference is minimal, assume the piece has locked
    if np.count_nonzero(difference) == 0:
        return True

    return False

model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def simulate_move(grid, piece, x_offset, y_offset):
    piece = np.array(piece)
    piece_height, piece_width = piece.shape

    simulated_grid = grid.copy()

    for y in range(piece_height):
        for x in range(piece_width):
            if piece[y, x] == 1:
                new_x = x + x_offset
                new_y = y + y_offset
                if 0 <= new_x < simulated_grid.shape[1] and 0 <= new_y < simulated_grid.shape[0]:
                    simulated_grid[new_y, new_x] = 1

    if not is_valid_move(simulated_grid, piece, x_offset, y_offset):
        return simulated_grid, -1 
    
    return simulated_grid, 1  

def get_offsets_for_action(action, piece):
    offsets = {
        0: (-1, 0),  # move left
        1: ( 1, 0),  # move right
        2: ( 0, 0),  # rotate counter-clockwise
        3: ( 0, 0),  # rotate clockwise
        4: ( 0, 1),  # move down
        5: ( 0, 0)   # drop
    }
    
    if action == 2:
        piece = np.rot90(piece)
    elif action == 3:
        piece = np.rot90(piece, 3)
    
    return offsets[action], piece

piece = get_current_piece()

def simulate_actions(grid, piece):
    best_action = None
    best_reward = -float('inf')
    best_simulated_grid = None

    for action in actions:
        if action == 2:
            piece_rotated = np.rot90(piece)
        elif action == 3:
            piece_rotated = np.rot90(piece, 3)
        else:
            piece_rotated = piece
        
        x_offset, y_offset = get_offsets_for_action(action, piece_rotated)

        simulated_grid, reward = simulate_move(grid, piece_rotated, x_offset, y_offset)

        if reward > best_reward:
            best_reward = reward
            best_action = action
            best_simulated_grid = simulated_grid

    return best_action, best_simulated_grid

def select_action(state):
    global epsilon

    if random.random() < epsilon:
        action = random.choice([0, 1, 2, 3, 4, 5])
        print(f"Exploring: Selected random action {action}")
    else:
        with torch.no_grad():
            Q_values = policy_net(state.unsqueeze(0))
            action = torch.argmax(Q_values).item()
            print(f"Exploiting: Selected best action {action} with Q-values {Q_values.numpy()}")

    return action

def preform_best_action(prev_state, piece, piece_position):
    grid = get_state()
    piece = get_current_piece()

    best_action, _ = simulate_actions(prev_state, piece, piece_position)

    send_action(best_action)

def reset_game():
    print("Resetting game state...")
    ExperienceBuffer.clear()

def train_agent():
    global epsilon, previous_background_color
    previous_background_color = get_background_color()
    piece_limit = 5
    enable_game_over_detection = False

    experience_buffer = ExperienceBuffer()

    for episode in range(1000):
        state = get_state()
        done = False
        total_reward = 0
        piece_count = 0

        while not done and piece_count < piece_limit:
            if keyboard.is_pressed('esc'):
                print("Manual stop triggered. Stopping the training loop...")
                exit(0)

            action = select_action(state)
            send_action(action)
            time.sleep(0.1)

            grid_capture_area, _ = capture_screen()
            next_state = preprocess_image(grid_capture_area)
            reward = calculate_reward(state, next_state)
            experience_buffer.store(state, action, reward, next_state, done)
            
            if experience_buffer.size() >= 32:
                batch = experience_buffer.sample(32)
                train(batch)

            if get_locked_pieces(state, next_state):
                piece_count += 1

            state = next_state
            total_reward += reward

            if enable_game_over_detection:
                current_background_color = get_background_color()
                if detect_game_over(current_background_color, previous_background_color):
                    print("Game Over Detected by Background Color")
                    done = True
                previous_background_color = current_background_color

        epsilon = max(epsilon * 0.995, 0.01)
        print(f"Episode {episode}, Total Reward: {total_reward}")

        keyboard.press_and_release('esc')
        time.sleep(2)
        handle_screen_state()

        if piece_count >= piece_limit:
            piece_limit += 5
            if piece_limit >= 50:
                enable_game_over_detection = True

        experience_buffer.clear()

    model_path = os.path.join(model_dir, "tetris_ai.pth")
    torch.save(model.state_dict(), model_path)

train_agent()