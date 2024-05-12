# Import the necessary libraries
import os
import pickle
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tensorflow.keras.preprocessing import image
from Foodimg2Ing.args import get_parser
from Foodimg2Ing.model import get_model
from Foodimg2Ing.utils.output_utils import prepare_output
from Foodimg2Ing import app

def output(uploadedfile):
    # Keep all the codes and pre-trained weights in data directory
    data_dir = os.path.join(app.root_path, 'data')

    # GPU or CPU configuration
    use_gpu = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    map_loc = device  # Use device as the map_location directly

    # Load vocabulary files
    ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
    vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))
    ingr_vocab_size = len(ingrs_vocab)
    instrs_vocab_size = len(vocab)

    # Initialize and load the model
    args = get_parser()
    args.maxseqlen = 15
    args.ingrs_only = False
    model = get_model(args, ingr_vocab_size, instrs_vocab_size)
    model_path = os.path.join(data_dir, 'modelbest.ckpt')
    model.load_state_dict(torch.load(model_path, map_location=map_loc))
    model.to(device)
    model.eval()

    # Image transformation pipeline
    to_input_transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load and transform the image
    img = image.load_img(uploadedfile)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    image_transf = transform(img)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

    # Generation settings
    greedy = [True, False]
    beam = [-1, -1]
    temperature = 1.0
    numgens = len(greedy)

    # Generate recipes
    title = []
    ingredients = []
    recipe = []
    for i in range(numgens):
        with torch.no_grad():
            outputs = model.sample(image_tensor, greedy=greedy[i], temperature=temperature, beam=beam[i], true_ingrs=None)
        
        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()
        
        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
        
        if valid['is_valid']:
            title.append(outs['title'])
            ingredients.append(outs['ingrs'])
            recipe.append(outs['recipe'])
        else:
            title.append("Not a valid recipe")
            recipe.append("Reason: " + valid['reason'])

    return title, ingredients, recipe
