from flask import Flask, render_template, request, jsonify
import os
import json
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import requests

app = Flask(__name__)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def identify_fruit_or_vegetable(image_path):
    model_name = "dima806/fruit_vegetable_image_detection"
    model = ViTForImageClassification.from_pretrained(model_name)
    image_processor = ViTImageProcessor.from_pretrained(model_name)

    image = Image.open(image_path)
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = logits.argmax(-1).item()
    labels = model.config.id2label

    return labels[predicted_label]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    ingredients = []

    for file in files:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        ingredient = identify_fruit_or_vegetable(file_path)
        ingredients.append(ingredient)
    
    meat = request.form.get('meat')
    if meat.lower() != 'no':
        ingredients.append(meat)

    output_path = 'ingrediant.json'
    with open(output_path, 'w') as json_file:
        json.dump(ingredients, json_file)

    return jsonify(ingredients)

@app.route('/recipes', methods=['GET'])
def recipes():
    api_key = 'b6763dfbbb944766b20a8c6fe6987e90'  
    with open('ingrediant.json', 'r') as json_file:
        ingredients = json.load(json_file)
    
    ingredients_string = ','.join(ingredients)
    url = f'https://api.spoonacular.com/recipes/findByIngredients'
    params = {
        'ingredients': ingredients_string,
        'number': 10,
        'ranking': 1,
        'ignorePantry': True,
        'apiKey': api_key
    }

    response = requests.get(url, params=params)
    print(f"Request URL: {response.url}")
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")

    if response.status_code == 200:
        recipes = response.json()
        return jsonify(recipes)
    else:
        return jsonify({'error': 'Error fetching recipes', 'details': response.text})

@app.route('/recipe_details/<int:recipe_id>', methods=['GET'])
def recipe_details(recipe_id):
    api_key = 'b6763dfbbb944766b20a8c6fe6987e90' 
    url = f'https://api.spoonacular.com/recipes/{recipe_id}/information'
    params = {'apiKey': api_key}

    response = requests.get(url, params=params)
    print(f"Request URL: {response.url}")
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")

    if response.status_code == 200:
        recipe = response.json()
        return render_template('recipe_details.html', recipe=recipe)
    else:
        return render_template('error.html', error=f'Error fetching recipe details: {response.status_code}')


if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
