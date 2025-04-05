import torch
import numpy as np
import os
import cv2

def set_seed(seed):
    # Set the seed for pytorch (CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic results
    torch.backends.cudnn.benchmark = False

def compute_lrp(model, x, epsilon=1e-5, alpha=2, beta=-1):
    model.eval()

    # Define hooks to capture activations and gradients for the layers
    activations = []
    gradients = []

    # Hook to capture activations
    def save_activation(module, input, output):
        activations.append(output)

    # Hook to capture gradients
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks for the layers you want to analyze (e.g., conv1, conv2)
    hook_activation_1 = model.conv1.register_forward_hook(save_activation)
    hook_gradient_1 = model.conv1.register_full_backward_hook(save_gradient)

    # Forward pass
    output = model(x)
    model.zero_grad()

    # Backpropagation for the class with maximum probability
    target_class = output.argmax(dim=1)
    target = output[range(output.size(0)), target_class]
    target.backward(torch.ones_like(target))  # Backpropagation w.r.t. the target class

    # Check if we have activations from the right layers
    if len(activations) < 1:
        print("Activation list does not have expected number of elements.")
        return None

    # Get activations and gradients for the layers
    activation = activations[0].detach()  # Get activations from the first convolution layer (conv1)
    gradient = gradients[0].detach()     # Get gradients for that layer

    # Apply the filtering rule for LRP
    z_ij = activation * gradient
    z_ij_pos = torch.clamp(z_ij, min=0)  # Positive part of z_ij
    z_ij_neg = torch.clamp(z_ij, max=0)  # Negative part of z_ij

    # Normalize the activations
    sum_z_pos = z_ij_pos.sum(dim=1, keepdim=True)
    sum_z_neg = z_ij_neg.sum(dim=1, keepdim=True)

    # Compute relevance using LRP formula
    relevance = (alpha * z_ij_pos / (sum_z_pos + epsilon) + beta * z_ij_neg / (sum_z_neg + epsilon)) * activation

    # Return the relevance map for the layer
    return relevance.sum(dim=1).cpu().numpy()

def normalize_relevance_map(heatmap):
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    normalized_map = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    return normalized_map


def plot_relevance(image, heatmap, idx):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    combined = np.hstack((image, heatmap[0]))

    output_folder = 'code/lrp/results'
    output_path = os.path.join(output_folder, 'relevance_map_' + str(idx) + '.png')
    cv2.imwrite(output_path, combined)