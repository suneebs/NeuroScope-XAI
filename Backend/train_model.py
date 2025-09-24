"""
Run this separately to train your model
python train_model.py
"""
import tensorflow as tf
import os
# ... (include your training code here)

if __name__ == "__main__":
    # Your existing training code
    # Save the trained model
    model.save('models/brain_tumor_model.h5')
    print("Model saved to models/brain_tumor_model.h5")