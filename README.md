# Webcam Hand-Tracking Cursor
This was just a quick project I made over a week or so to learn the basics of AI. After taking images, labelling, and training, it is able to control your computer's cursor with your hand. Moving your open hand around will move the cursor, and closing your hand into a fist will click.

## How to Use
1. `1_take_images.py` takes training images. It will start by showing a still frame through your webcam, and pressing a key will update the frame. Pressing "s" will save the frame in `data/d{n}.png` where n starts at 0 and counts up. The images taken are 160x90.
2. `2_label_images.py` labels the images in `data.txt`. It will show the images in the `data/` directory.
    - Left click on open hands
    - Right click on closed fists
    - Press space if there is no hand
3. `3_dataset_checker.py` simulates the training data generation for validation. There are three models: the first determines whether there is a hand (`model_hand`), the second determines where the hand is (`model_xy`), and the third determines if the hand is a fist (`model_fist`).
    - The first model is given full images (all the images with no hands and 1/6 of the images with a hand, to balance out the data a bit) and labels for whether the hand is there
    - The second model is given the full images that have a hand and given labels with x-y coordinates.
    - The third model gets a 35x35 image cropped around the hand (and randomly offset to account for inaccuracies of model 2) and labels for whether the hand is a fist
    - The program just displays the results of the third model, there should just be many tiny images of hands.
4. `4_training.py` will train the 3 models using the same logic in `3_dataset_checker.py` to prepare the datset. This will create `model_fist.keras`, `model_hand.keras`, and `model_xy.keras`
5. `hand_tracker.py` will just show your webcam stream with a dot.
    - Red dot in middle: no hand
    - Green dot on hand: fist
    - Blue dot on hand: palm
6. `hand_cursor.py` is the same as `hand_tracker.py`, except it also controls your mouse!