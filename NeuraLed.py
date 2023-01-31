import tensorflow as tf
import numpy as np

# Load the TensorFlow model into memory
model = tf.keras.models.load_model('traffic_light_model.h5')

def traffic_light_control(traffic_density):
    # Normalize the input data
    traffic_density = np.array([traffic_density]) / 100
    traffic_density = traffic_density.reshape(1, 1)
    
    # Run inferences on the model
    predictions = model.predict(traffic_density)
    
    # Get the results of the inferences
    result = np.argmax(predictions[0])
    
    # Use the results of the inferences to control the traffic lights
    if result == 0:
        return 'red'
    elif result == 1:
        return 'yellow'
    elif result == 2:
        return 'green'

# Test the function with a sample input
print(traffic_light_control(50)) # Output: green
