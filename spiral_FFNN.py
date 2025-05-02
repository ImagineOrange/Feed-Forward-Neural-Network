import numpy as np
import matplotlib.pyplot as plt
import sys
import imageio.v2 as imageio # ImageIO for creating GIFs (using v2 API)
import os                    # For interacting with the operating system (file paths, directory handling)
import time                  # For timing the training process
import shutil                # For advanced file operations (removing directory trees)

# --- Global Setup ---

# Ensure the temporary directory for GIF frames exists
if not os.path.exists('decision_boundary_frames'):
    os.makedirs('decision_boundary_frames')

# Set a default plotting style (can be overridden locally)
plt.style.use('fast')

# --- Data Generation and Visualization ---

def gen_spiral_data(points, classes):
    """
    Generates a spiral dataset for classification.

    Based on the Stanford CS231n spiral dataset generator:
    https://cs231n.github.io/neural-networks-case-study/

    Args:
        points (int): Number of data points per class spiral.
        classes (int): Number of distinct classes (spirals) to generate.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Array of features (shape: [points*classes, 2]).
            - y (np.ndarray): Array of class labels (shape: [points*classes]).
    """
    X = np.zeros((points * classes, 2))         # Feature matrix (each row is a point [x, y])
    y = np.zeros(points * classes, dtype='uint8') # Class label vector
    for class_number in range(classes):
        # Calculate indices for the current class
        ix = range(points * class_number, points * (class_number + 1))
        # Generate radius values
        r = np.linspace(0.0, 1, points)
        # Generate angle values with spiral twist and some noise
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) \
            + np.random.randn(points) * 0.1
        # Convert polar coordinates (r, t) to Cartesian coordinates (x, y)
        X[ix] = np.c_[r * np.sin(t * 2), r * np.cos(t * 2)]
        # Assign the class label
        y[ix] = class_number
    print('\nData Generated.')
    # Note: Initial visualization call removed as per user request
    return X, y

def visualize_data(X, y, classes):
    """
    Visualizes the generated spiral dataset (Not called by default).

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Class labels.
        classes (int): Number of classes.
    """
    # Use a dark background for contrast
    with plt.style.context('dark_background'):
        plt.figure("Spiral Dataset", figsize=(10, 7))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', s=10) # Use 'jet' colormap
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"Spiral Dataset - Classes: {classes}, Features: 2")
        plt.grid(False)
        plt.show(block=False) # Show plot without blocking execution
        plt.pause(1)          # Pause briefly to allow plot rendering

# --- Utility Functions ---

def progress_bar(i, ceiling, loss, accuracy, learning_rate=None, momentum=None):
    """
    Displays a simple text-based progress bar in the console.

    Args:
        i (int): Current iteration (0-based).
        ceiling (int): Total number of iterations.
        loss (float): Current batch/epoch loss.
        accuracy (float): Current batch/epoch accuracy.
        learning_rate (float, optional): Current learning rate. Defaults to None.
        momentum (float, optional): Current momentum value. Defaults to None.
    """
    barlength = 25 # Length of the progress bar
    percentage = int((i + 1) / ceiling * 100) # Calculate percentage completion
    num_hash = int(percentage / 100 * barlength) # Number of '#' symbols
    num_space = int(barlength - num_hash)       # Number of '_' symbols
    tally = '#' * num_hash
    space = '_' * num_space

    # Build the status string
    status_str = (
        f"\r   Training Progress: [{tally}{space}] {percentage}% "
        f"- ({i+1}/{ceiling}) - "
        f"loss: {loss:.4f} --- "
        f"accuracy: {accuracy * 100:.2f} %"
    )
    # Append optional metrics
    if learning_rate is not None:
         status_str += f" --- LR: {learning_rate:.5f}"
    if momentum is not None and momentum > 0:
         status_str += f" --- Momentum: {momentum:.2f}"

    # Write to console, overwriting the previous line (\r)
    sys.stdout.write(status_str + "     ") # Add padding
    sys.stdout.flush() # Ensure immediate display

def plot_decision_boundary(X_orig, y_orig, model, epoch, classes, filename):
    """
    Plots the decision boundary of the current model state and saves it to a file.

    Args:
        X_orig (np.ndarray): Original feature data.
        y_orig (np.ndarray): Original class labels.
        model (dict): Dictionary containing model components ('layers', 'activations', 'softmax').
        epoch (int): Current epoch number (for title and filename).
        classes (int): Number of classes.
        filename (str): Path to save the plot image.
    """
    h = 0.02 # Step size in the mesh grid

    # Determine plot boundaries based on data range with some padding
    x_min, x_max = X_orig[:, 0].min() - 0.5, X_orig[:, 0].max() + 0.5
    y_min, y_max = X_orig[:, 1].min() - 0.5, X_orig[:, 1].max() + 0.5

    # Create a mesh grid of points spanning the plot boundaries
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Flatten the mesh grid into a list of points for prediction
    mesh_input = np.c_[xx.ravel(), yy.ravel()]

    # Get model components
    layers = model['layers']           # List of Layer_Dense objects
    relu_activations = model['activations'] # List of Activation_ReLU objects
    softmax_activation = model['softmax']   # Standalone Activation_Softmax object

    # --- Forward Pass for Mesh Grid ---
    # Propagate the mesh grid points through the network layers and activations
    current_output = mesh_input
    # Process input layer + its activation (first layer/activation)
    layers[0].forward_pass(current_output)
    relu_activations[0].forward_pass(layers[0].output)
    current_output = relu_activations[0].output

    # Process hidden layers + their activations
    for i in range(1, len(layers) - 1): # Iterate through hidden layers
        layers[i].forward_pass(current_output)
        if i < len(relu_activations): # Ensure there's a corresponding activation
             relu_activations[i].forward_pass(layers[i].output)
             current_output = relu_activations[i].output
        else: # Safety check, should not normally happen
             print(f"Warning: Activation missing for layer {i} in plot_decision_boundary.")
             current_output = layers[i].output # Pass through layer without activation if missing

    # Process the output layer
    output_layer = layers[-1]
    output_layer.forward_pass(current_output)

    # Apply standalone Softmax to get probabilities for the mesh points
    softmax_activation.forward_pass(output_layer.output)
    probabilities = softmax_activation.output
    # --- End Forward Pass ---

    # Get the predicted class (index with highest probability) for each mesh point
    Z = np.argmax(probabilities, axis=1)

    # Reshape the predictions back into the mesh grid shape for plotting
    if Z.size == xx.size:
         Z = Z.reshape(xx.shape)
    else: # Handle potential shape mismatch (should not happen with correct logic)
        print(f"\nError in plot_decision_boundary: Mismatch shapes. Z:{Z.size}, xx:{xx.size}. Skip frame {epoch}.")
        return # Skip plotting this frame

    # --- Plotting ---
    # Use dark background for contrast
    with plt.style.context('dark_background'):
        plt.figure(figsize=(10, 7))
        # Plot the decision regions using contourf
        plt.contourf(xx, yy, Z, cmap='jet', alpha=0.2)
        # Overlay the original data points
        plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y_orig, s=15, edgecolors='k', cmap='jet')
        # Set plot limits, title, and labels
        plt.xlim(xx.min(), xx.max()); plt.ylim(yy.min(), yy.max())
        plt.title(f'Decision Boundary - Epoch {epoch}')
        plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
        plt.grid(False)
        # Save the plot to the specified file and close the figure
        plt.savefig(filename); plt.close()

# --- Neural Network Components ---

class Layer_Dense:
    """Represents a fully connected (dense) layer in the neural network."""
    def __init__(self, n_inputs, n_neurons):
        """
        Initializes the dense layer.

        Args:
            n_inputs (int): Number of input features (size of the input vector).
                            Equal to the number of neurons in the previous layer.
            n_neurons (int): Number of neurons in this layer.
        """
        # Initialize weights with small random values (randn -> Gaussian distribution)
        # Shape: (n_inputs, n_neurons)
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases with zeros
        # Shape: (1, n_neurons) - a row vector for broadcasting
        self.biases = np.zeros((1, n_neurons))
        # Initialize momentum arrays for weights and biases (used by optimizer)
        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.biases)

    def forward_pass(self, inputs):
        """
        Performs the forward pass calculation for the layer.
        Output = inputs * weights + biases

        Args:
            inputs (np.ndarray): Output from the previous layer or initial input data.
                                 Shape: (batch_size, n_inputs).
        """
        # Store inputs for use in backpropagation
        self.inputs = inputs
        # Calculate the layer's output (linear transformation)
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward_pass(self, dvalues):
        """
        Performs the backward pass calculation (backpropagation) for the layer.
        Calculates gradients with respect to weights, biases, and inputs.

        Args:
            dvalues (np.ndarray): Gradient of the cost function with respect to the
                                  output of this layer (coming from the next layer).
                                  Shape: (batch_size, n_neurons).
        """
        # Gradient of the cost function with respect to weights (dCost/dWeights)
        # Uses the chain rule: dCost/dWeights = dCost/dOutput * dOutput/dWeights
        # dOutput/dWeights = inputs^T
        self.dweights = np.dot(self.inputs.T, dvalues)

        # Gradient of the cost function with respect to biases (dCost/dBiases)
        # Uses the chain rule: dCost/dBiases = dCost/dOutput * dOutput/dBiases
        # dOutput/dBiases = 1
        # Sum gradients across the batch
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient of the cost function with respect to inputs (dCost/dInputs)
        # Uses the chain rule: dCost/dInputs = dCost/dOutput * dOutput/dInputs
        # dOutput/dInputs = weights^T
        # This gradient is passed back to the previous layer.
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    """Rectified Linear Unit (ReLU) activation function."""
    def forward_pass(self, inputs):
        """
        Applies the ReLU function element-wise: output = max(0, input).

        Args:
            inputs (np.ndarray): Output from the preceding layer.
        """
        # Store inputs for backpropagation
        self.inputs = inputs
        # Apply ReLU
        self.output = np.maximum(0, inputs)

    def backward_pass(self, dvalues):
        """
        Calculates the gradient of the cost function w.r.t. the ReLU inputs.

        Args:
            dvalues (np.ndarray): Gradient of the cost function w.r.t. the
                                  output of the ReLU activation.
        """
        # Make a copy to avoid modifying the original gradient array
        self.dinputs = dvalues.copy()
        # Apply the ReLU derivative: 1 if input > 0, 0 otherwise.
        # Zero out gradients where the original input was non-positive.
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    """Softmax activation function for the output layer (for multi-class classification)."""
    def forward_pass(self, inputs):
        """
        Applies the Softmax function: Exponentiates and normalizes inputs
        to produce a probability distribution across classes.

        Args:
            inputs (np.ndarray): Output from the final dense layer.
                                 Shape: (batch_size, n_classes).
        """
        # Subtract max value for numerical stability (prevents large exponentials)
        # axis=1 ensures max is found per sample (row), keepdims preserves dimensions
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize to get probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities # Shape: (batch_size, n_classes)

    def backward_pass(self, dvalues):
        """
        Calculates the gradient of the cost function w.r.t. the Softmax inputs.
        Note: This is typically combined with the CCE loss derivative for efficiency.

        Args:
            dvalues (np.ndarray): Gradient of the cost function w.r.t. the
                                  Softmax output (probabilities).
        """
        # Initialize gradient array
        self.dinputs = np.empty_like(dvalues)
        # Calculate gradient for each sample using the Jacobian matrix of Softmax
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Reshape single_output to a column vector
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix (diag(softmax) - softmax * softmax^T)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient: Jacobian * dvalues
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    """Base class for loss functions."""
    def calculate(self, output, y, *, include_regularization=False):
        """
        Calculates the average loss for a batch.

        Args:
            output (np.ndarray): Model's predicted output (e.g., probabilities).
            y (np.ndarray): True target labels.
            include_regularization (bool): Flag to include regularization loss (not implemented here).

        Returns:
            float: Average data loss for the batch.
        """
        # Calculate sample losses using the specific loss function's forward pass
        sample_losses = self.forward_pass(output, y)
        # Calculate the mean loss over the batch
        batch_loss = np.mean(sample_losses)
        # Placeholder for adding regularization loss if implemented
        # if include_regularization: batch_loss += self.regularization_loss(layer1) + ...
        return batch_loss

class Loss_Categorical_Cross_Entropy(Loss):
    """
    Categorical Cross-Entropy (CCE) loss function.
    Suitable for multi-class classification with Softmax output.
    Measures the difference between the predicted probability distribution
    and the true distribution (one-hot encoded or sparse labels).
    """
    def forward_pass(self, y_pred, y_true):
        """
        Calculates CCE loss for each sample.

        Args:
            y_pred (np.ndarray): Predicted probability distribution from Softmax.
                                 Shape: (batch_size, n_classes).
            y_true (np.ndarray): True class labels (either sparse indices or one-hot encoded).
                                 Shape: (batch_size,) or (batch_size, n_classes).

        Returns:
            np.ndarray: Array of CCE loss values for each sample in the batch.
        """
        samples = len(y_pred)
        # Clip predicted probabilities to avoid log(0) -> infinity
        # Values smaller than 1e-7 become 1e-7, larger than 1-1e-7 become 1-1e-7
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Determine how to get the predicted probability for the correct class
        if len(y_true.shape) == 1: # Sparse labels (e.g., [0, 2, 1, ...])
            # Select the probability corresponding to the true class index for each sample
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # One-hot encoded labels (e.g., [[1,0,0], [0,0,1], ...])
            # Multiply predicted probabilities by the one-hot matrix and sum rows
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
             raise ValueError("Invalid shape for y_true in CCE loss calculation.")

        # Calculate the negative log likelihood (the CCE loss)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward_pass(self, dvalues, y_true):
        """
        Calculates the gradient of the CCE loss w.r.t. its input (y_pred).
        Note: This is often combined with Softmax backward pass for efficiency.

        Args:
            dvalues (np.ndarray): Gradient w.r.t. the output of this loss function
                                  (usually the combined Softmax+CCE gradient).
            y_true (np.ndarray): True class labels.
        """
        samples = len(dvalues)
        labels = len(dvalues[0]) # Number of classes

        # If labels are one-hot encoded, convert to sparse indices for easier calculation
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Calculate gradient: -y_true / y_pred (where y_true is one-hot)
        # However, the combined Softmax+CCE gradient simplifies this:
        self.dinputs = dvalues.copy() # Start with the gradient passed from the combined class
        # Subtract 1 from the gradient corresponding to the correct class index
        self.dinputs[range(samples), y_true] -= 1
        # Normalize the gradient by the number of samples
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Combines Softmax activation and CCE loss for efficient backpropagation.
    Calculates the gradient in a single, simplified step.
    """
    def __init__(self):
        """Initializes the constituent activation and loss objects."""
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Cross_Entropy()

    def forward_pass(self, inputs, y_true):
        """
        Performs forward pass through Softmax and calculates CCE loss.

        Args:
            inputs (np.ndarray): Output from the final dense layer.
            y_true (np.ndarray): True class labels.

        Returns:
            float: The calculated batch loss.
        """
        # Perform Softmax activation
        self.activation.forward_pass(inputs)
        # Store the Softmax output (probabilities)
        self.output = self.activation.output
        # Calculate and return the CCE loss using the Softmax output
        return self.loss.calculate(self.output, y_true)

    def backward_pass(self, dvalues, y_true):
        """
        Calculates the combined gradient of Softmax+CCE loss w.r.t. the
        input of the Softmax layer (output of the previous dense layer).
        The formula simplifies to: (y_pred - y_true) / n_samples

        Args:
            dvalues (np.ndarray): Output probabilities from the forward pass (self.output).
            y_true (np.ndarray): True class labels.
        """
        samples = len(dvalues)
        # If labels are one-hot, convert to sparse indices
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Calculate the gradient directly: y_pred - y_true (where y_true is one-hot)
        self.dinputs = dvalues.copy() # Start with y_pred
        # Subtract 1 from the prediction corresponding to the true class
        self.dinputs[range(samples), y_true] -= 1
        # Normalize the gradient by the number of samples
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum and learning rate decay.
    """
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        """
        Initializes the optimizer.

        Args:
            learning_rate (float): Initial learning rate (step size).
            decay (float): Learning rate decay factor applied per iteration.
            momentum (float): Momentum factor (typically 0.9).
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate # Track potentially decayed LR
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0 # Counter for decay calculation

    def pre_update_parameters(self):
        """Calculates the current learning rate if decay is enabled."""
        if self.decay:
            # Decay formula: lr / (1 + decay * iteration)
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_parameters(self, layer):
        """
        Updates the weights and biases of a given layer using SGD with momentum.

        Args:
            layer (Layer_Dense): The layer whose parameters need updating.
        """
        # Calculate updates using momentum
        # Update = momentum * previous_update - learning_rate * gradient
        weight_updates = self.momentum * layer.weight_momentums - \
                         self.current_learning_rate * layer.dweights
        # Store the new update for the next iteration
        layer.weight_momentums = weight_updates

        bias_updates = self.momentum * layer.bias_momentums - \
                       self.current_learning_rate * layer.dbiases
        layer.bias_momentums = bias_updates

        # Apply the calculated updates to the layer's parameters
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_parameters(self):
        """Increments the iteration counter (used for decay)."""
        self.iterations += 1

# --- Training Loop ---

def train_spiral_model(
    points, classes, epochs,
    hidden_neurons=(64, 64),
    learning_rate=1.0, decay=1e-4, momentum=0.9,
    plot_every=100
):
    """
    Trains the Feed-Forward Neural Network on the spiral dataset.

    Args:
        points (int): Number of data points per class.
        classes (int): Number of classes.
        epochs (int): Number of training iterations.
        hidden_neurons (tuple): Tuple defining the number of neurons in each hidden layer.
        learning_rate (float): Initial learning rate for SGD.
        decay (float): Learning rate decay factor for SGD.
        momentum (float): Momentum factor for SGD.
        plot_every (int): Frequency (in epochs) for saving decision boundary plots.
    """
    start_time = time.time()
    # Generate the dataset
    X, y = gen_spiral_data(points, classes)

    # --- Initialize Network Components ---
    layers = []
    n_inputs = X.shape[1] # Number of features (2 for spiral data)
    # Create hidden layers based on the hidden_neurons tuple
    for n_neurons in hidden_neurons:
        layers.append(Layer_Dense(n_inputs, n_neurons))
        n_inputs = n_neurons # Output of current layer is input to next
    # Create the output layer
    layers.append(Layer_Dense(n_inputs, classes))

    # Create ReLU activation functions for all layers except the output layer
    relu_activations = [Activation_ReLU() for _ in range(len(layers) - 1)]
    # Use the combined Softmax+CCE class for efficient training
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    # Need a separate Softmax for plotting decision boundaries
    softmax_activation = Activation_Softmax()

    # Initialize the optimizer
    optimizer = Optimizer_SGD(learning_rate=learning_rate, decay=decay, momentum=momentum)
    # --- End Initialization ---

    # --- Print Model Configuration ---
    shape_parts = [str(X.shape[1])] + [str(n) for n in hidden_neurons] + [str(classes)]
    network_shape_str = " x ".join(shape_parts)
    print(f"\n   Network Shape (Inputs x Hidden... x Outputs): {network_shape_str}")
    print(f"""\
   Classes: {classes}
   Training Epochs = {epochs}
   Learning Rate = {optimizer.learning_rate}{f" (decay={optimizer.decay})" if optimizer.decay > 0 else ""}
   Momentum = {optimizer.momentum}{" (applied)" if optimizer.momentum > 0 else ""}
   Plotting Decision Boundary Every {plot_every} Epochs
""")
    # --- End Print ---

    # Store components needed by the plotting function
    model_components = {
        'layers': layers, 'activations': relu_activations, 'softmax': softmax_activation
    }

    # Lists to store history for final plotting
    loss_history, accuracy_history, epoch_history, frame_files = [], [], [], []

    # --- Main Training Loop ---
    for epoch in range(epochs):
        # --- Forward pass ---
        current_output = X # Start with the input data
        # Pass through input layer and hidden layers with ReLU activations
        for i, layer in enumerate(layers[:-1]): # Iterate through all but output layer
            layer.forward_pass(current_output)
            relu_activations[i].forward_pass(layer.output)
            current_output = relu_activations[i].output # Output becomes input for next layer
        # Pass through the output layer
        output_layer = layers[-1]
        output_layer.forward_pass(current_output)
        # Perform forward pass of the combined Softmax+Loss function
        loss = loss_activation.forward_pass(output_layer.output, y)
        # Get predictions from the combined activation's output
        predictions = np.argmax(loss_activation.output, axis=1)
        # --- End Forward Pass ---

        # --- Calculate Accuracy ---
        if len(y.shape) == 2: y_targets = np.argmax(y, axis=1) # Handle one-hot if necessary
        else: y_targets = y
        accuracy = np.mean(predictions == y_targets)
        # Store metrics for plotting
        loss_history.append(loss); accuracy_history.append(accuracy); epoch_history.append(epoch)

        # --- Backward pass ---
        # Start backpropagation with the combined Softmax+Loss gradient
        loss_activation.backward_pass(loss_activation.output, y)
        current_dvalues = loss_activation.dinputs
        # Backpropagate through the output layer
        output_layer.backward_pass(current_dvalues)
        current_dvalues = output_layer.dinputs
        # Backpropagate through hidden layers and ReLU activations (in reverse order)
        for i in range(len(layers) - 2, -1, -1): # From second-to-last down to first layer
            relu_activations[i].backward_pass(current_dvalues)
            layers[i].backward_pass(relu_activations[i].dinputs)
            current_dvalues = layers[i].dinputs # Gradient w.r.t. layer input
        # --- End Backward pass ---

        # --- Update Parameters ---
        optimizer.pre_update_parameters() # Update learning rate if decay is enabled
        for layer in layers: # Update weights and biases for all layers
            optimizer.update_parameters(layer)
        optimizer.post_update_parameters() # Increment iteration counter
        # --- End Update ---

        # Display progress
        progress_bar(epoch, epochs, loss, accuracy, optimizer.current_learning_rate, momentum)

        # Plot decision boundary periodically
        if epoch % plot_every == 0 or epoch == epochs - 1:
            frame_filename = f"decision_boundary_frames/frame_{epoch:05d}.png"
            plot_decision_boundary(X, y, model_components, epoch, classes, frame_filename)
            # Add filename to list for GIF creation if the plot was saved successfully
            if os.path.exists(frame_filename): frame_files.append(frame_filename)
    # --- End Training Loop ---

    print("\n\nTraining finished.")
    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

    # --- Create GIF ---
    if frame_files:
        print("Creating decision boundary GIF..."); gif_path = 'decision_boundary.gif'
        try:
            # Use imageio writer to append frames
            with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer: # duration controls frame speed
                for filename in sorted(frame_files): # Sort frames ensures correct order
                    if os.path.exists(filename): writer.append_data(imageio.imread(filename))
            print(f"GIF saved as {gif_path}")
        except Exception as e: print(f"Error creating GIF: {e}")

        # --- Cleanup using shutil.rmtree ---
        print("Cleaning up temporary frame directory...")
        try:
            shutil.rmtree('decision_boundary_frames') # Force remove directory and its contents
            print("Temporary directory removed.")
        except OSError as e: print(f"Error removing directory 'decision_boundary_frames': {e}")
    else: print("No frames generated, skipping GIF creation.")

    # --- Plot final loss/accuracy (Normalized, Dark Style) ---
    if loss_history and accuracy_history: # Check if lists are not empty
        # --- Normalization ---
        loss_arr = np.array(loss_history)
        acc_arr = np.array(accuracy_history)
        loss_min, loss_max = loss_arr.min(), loss_arr.max()
        acc_min, acc_max = acc_arr.min(), acc_arr.max()
        # Min-Max scaling: (value - min) / (max - min)
        # Add small epsilon to denominator to avoid division by zero if max == min
        epsilon = 1e-9
        norm_loss = (loss_arr - loss_min) / (loss_max - loss_min + epsilon)
        norm_acc = (acc_arr - acc_min) / (acc_max - acc_min + epsilon)
        # --- End Normalization ---

        # Apply dark style locally for this plot
        with plt.style.context('dark_background'):
            plt.figure("Normalized Loss and Accuracy", figsize=(12, 7))
            plt.plot(epoch_history, norm_loss, label='Normalized Loss', color='red', alpha=0.8, linewidth=1)
            plt.plot(epoch_history, norm_acc, label='Normalized Accuracy', color='cyan', alpha=0.8, linewidth=1)
            plt.title("Normalized Loss and Accuracy over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Normalized Value (0 to 1)")
            plt.legend()
            plt.grid(False)
            plt.ylim(-0.05, 1.05) # Set Y axis limits
            plt.tight_layout() # Adjust plot to prevent labels overlapping
    else:
        print("No data to plot for Loss and Accuracy.")

# --- Main Execution Block ---
def main():
    """Sets hyperparameters and initiates the model training."""
    # --- Hyperparameters (Based on original naive_backprop) ---
    TOTAL_POINTS = 2000       # Total number of data points
    CLASSES = 6               # Number of distinct classes (spirals)
    EPOCHS = 15000            # Number of training iterations
    POINTS_PER_CLASS = TOTAL_POINTS // CLASSES # Points per class for generation
    HIDDEN_NEURONS = (32, 32) # Defines network architecture: 2 -> 32 -> 16 -> 5
    LEARNING_RATE = 0.25      # Initial step size for optimizer
    DECAY = 0.0               # Learning rate decay (0 = no decay)
    MOMENTUM = 0.0            # Momentum factor (0 = no momentum)
    PLOT_EVERY = 50          # Frequency to plot decision boundary for GIF
    # --- End Hyperparameters ---

    print(f"Note: Using parameters based on original naive_backprop setup:")
    print(f"Total Points={TOTAL_POINTS}, Classes={CLASSES}, Epochs={EPOCHS}")
    print(f"Hidden Layers={HIDDEN_NEURONS}, LR={LEARNING_RATE}, Decay={DECAY}, Momentum={MOMENTUM}")

    # Start the training process
    train_spiral_model(
        points=POINTS_PER_CLASS, classes=CLASSES, epochs=EPOCHS,
        hidden_neurons=HIDDEN_NEURONS, learning_rate=LEARNING_RATE,
        decay=DECAY, momentum=MOMENTUM, plot_every=PLOT_EVERY
    )

if __name__ == "__main__":
    # This block executes only when the script is run directly
    main()
    plt.show() # Display the final plot(s) after training