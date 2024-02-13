import torch
import torch.nn.functional as F

def predict_with_probs(recognizer, img):
    """
    Predict the label of an image along with the probabilities of each character.
    
    Args:
    recognizer: An instance of the recognizer class with a defined model, converter, and necessary methods.
    img: The input image for prediction.
    
    Returns:
    tuple: A tuple containing the predicted label as a string and the probabilities of each character.
    """
    # Preprocess the image to match the model's input requirements
    image_tensors = recognizer.align_val([img])

    # Ensure the model is in evaluation mode
    recognizer.model.eval()

    with torch.no_grad():
        # Make predictions
        preds = recognizer.model(image_tensors.to(recognizer.device), '')
        # Calculate probabilities
        preds_text_prob = F.softmax(preds, dim=2)
        preds_max_prob, preds_index = preds_text_prob.max(dim=2)

        # Decode predictions into strings
        batch_size = image_tensors.size(0)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(recognizer.device)
        preds_str = recognizer.converter.decode(preds_index, preds_size)

    # Return the first (and only) prediction and its character probabilities
    return preds_str[0], preds_text_prob.squeeze(0)  # Assuming batch size of 1 for simplicity

# Usage example:
# recognizer = Recognizer(recog_config, device=device)  # Ensure the Recognizer class is correctly defined
image = dl.dataset[0][0]  # Assuming dl is your DataLoader
pred_label, pred_probs = predict_with_probs(model, image)
