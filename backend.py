from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = load_model('my_model.h5')

# Define the class labels (15 classes)
class_names = ['Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 
               'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 
               'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

# Water footprint data and alternative descriptions
water_footprint_data = {
    "Bean": "Beans have a water footprint of 1500 liters/kg. An alternative is Lentils with 600 liters/kg.",
    "Bitter Gourd": "Bitter gourd has a water footprint of 2000 liters/kg. An alternative is Cucumber with 500 liters/kg.",
    "Bottle Gourd": "Bottle gourd has a water footprint of 1800 liters/kg. An alternative is Zucchini with 700 liters/kg.",
    "Brinjal": "Brinjal has a water footprint of 1800 liters/kg. An alternative is Bell Pepper with 500 liters/kg.",
    "Broccoli": "Broccoli has a water footprint of 1000 liters/kg. An alternative is Kale with 600 liters/kg.",
    "Cabbage": "Cabbage has a water footprint of 700 liters/kg. An alternative is Swiss Chard with 500 liters/kg.",
    "Capsicum": "Capsicum has a water footprint of 500 liters/kg. An alternative is Chili Peppers with 400 liters/kg.",
    "Carrot": "Carrot has a water footprint of 600 liters/kg. An alternative is Beetroot with 500 liters/kg.",
    "Cauliflower": "Cauliflower has a water footprint of 1200 liters/kg. An alternative is Romanesco with 1000 liters/kg.",
    "Cucumber": "Cucumber has a water footprint of 500 liters/kg. An alternative is Celery with 300 liters/kg.",
    "Papaya": "Papaya has a water footprint of 2500 liters/kg. An alternative is Melon with 1000 liters/kg.",
    "Potato": "Potato has a water footprint of 500 liters/kg. An alternative is Sweet Potato with 400 liters/kg.",
    "Pumpkin": "Pumpkin has a water footprint of 1200 liters/kg. An alternative is Butternut Squash with 1000 liters/kg.",
    "Radish": "Radish has a water footprint of 300 liters/kg. An alternative is Turnip with 250 liters/kg.",
    "Tomato": "Tomato has a water footprint of 1000 liters/kg. An alternative is Cherry Tomato with 800 liters/kg."
}

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to the same size as your model's input shape
    img = image.img_to_array(img) / 255.0  # Convert to array and normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the image class and get water footprint data
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    img_file = request.files['image']
    img = Image.open(img_file.stream)
    img_preprocessed = preprocess_image(img)

    prediction = model.predict(img_preprocessed)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_class = class_names[predicted_class_index[0]]
    footprint_info = water_footprint_data.get(predicted_class,{})

    return jsonify(footprint_info)

# Fetch water footprint info by product name
@app.route('/getFootprint', methods=['GET'])
def get_footprint():
    product_name = request.args.get('product')
    footprint_info = water_footprint_data.get(product_name.capitalize(), {})
    return jsonify(footprint_info)

if __name__ == '__main__':
    app.run(debug=True)

# Example of usage
# Assuming you have an image to predict
# img = Image.open('path_to_image.jpg')  # Load your image here
# predicted_class, footprint_info = predict_class(img)
# print(f"Predicted class: {predicted_class}")
# print(f"Water Footprint Info: {footprint_info}")




# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np

# app = Flask(__name__)
# model = load_model('my_model.h5')

# # Water footprint data with alternatives and descriptions
# water_footprint_data = {
#     "Bean": "Beans have a water footprint of 1500 liters/kg. They are high in protein and fiber and are used in a variety of dishes, from soups to salads. An alternative to beans is Lentils, which have a water footprint of 600 liters/kg. Lentils are a high-protein legume with a lower water footprint compared to beans. They are used in soups, stews, and salads.",
    
#     "Bitter Gourd": "Bitter gourd has a water footprint of 2000 liters/kg. It is known for its bitter taste and is often used in Asian cuisine. It is believed to have health benefits for blood sugar control. An alternative to bitter gourd is Cucumber, with a water footprint of 500 liters/kg. Cucumber can be an alternative to bitter gourd, offering a similar refreshing quality with a lower water footprint.",
    
#     "Bottle Gourd": "Bottle gourd has a water footprint of 1800 liters/kg. It is a versatile vegetable used in many Indian and Asian dishes. It is low in calories and high in water content. An alternative to bottle gourd is Zucchini, which has a water footprint of 700 liters/kg. Zucchini is a versatile squash used in cooking and baking. It has a lower water footprint and similar culinary uses to bottle gourd.",
    
#     "Brinjal": "Brinjal, also known as eggplant, has a water footprint of 1800 liters/kg. It is a staple in many cuisines with a mild flavor, often used in curries and stews. An alternative to brinjal is Bell Pepper, with a water footprint of 500 liters/kg. Bell pepper is a crunchy and sweet alternative to brinjal, used in similar dishes and with a lower water footprint.",
    
#     "Broccoli": "Broccoli has a water footprint of 1000 liters/kg. It is a nutrient-dense vegetable known for its high vitamin C and fiber content. It is commonly used in salads and stir-fries. An alternative to broccoli is Kale, with a water footprint of 600 liters/kg. Kale is a leafy green with a lower water footprint compared to broccoli. It is rich in vitamins and used in salads and smoothies.",
    
#     "Cabbage": "Cabbage has a water footprint of 700 liters/kg. It is a leafy vegetable used in salads, soups, and as a side dish. It is rich in vitamins K and C. An alternative to cabbage is Swiss Chard, which has a water footprint of 500 liters/kg. Swiss chard is a leafy green vegetable that can replace cabbage in recipes, offering similar nutritional benefits with less water use.",
    
#     "Capsicum": "Capsicum, or bell pepper, has a water footprint of 500 liters/kg. It comes in various colors and is used for its sweet, crunchy texture in salads and stir-fries. An alternative to capsicum is Chili Peppers, with a water footprint of 400 liters/kg. Chili peppers are a spicier alternative to capsicum, providing a similar flavor profile with a lower water footprint.",
    
#     "Carrot": "Carrot has a water footprint of 600 liters/kg. Carrots are root vegetables known for their high vitamin A content. They are commonly eaten raw, cooked, or in juices. An alternative to carrot is Beetroot, with a water footprint of 500 liters/kg. Beetroot is a root vegetable with a lower water footprint than carrots. It is used in salads, juices, and as a side dish.",
    
#     "Cauliflower": "Cauliflower has a water footprint of 1200 liters/kg. It is a cruciferous vegetable that can be used in a variety of dishes, including soups, salads, and as a rice substitute. An alternative to cauliflower is Romanesco, with a water footprint of 1000 liters/kg. Romanesco is a cruciferous vegetable that has a similar taste and texture to cauliflower with slightly less water use.",
    
#     "Cucumber": "Cucumber has a water footprint of 500 liters/kg. It is a refreshing vegetable with high water content, often used in salads and sandwiches. An alternative to cucumber is Celery, with a water footprint of 300 liters/kg. Celery is a crisp vegetable with a lower water footprint than cucumber, used in salads, soups, and as a crunchy snack.",
    
#     "Papaya": "Papaya has a water footprint of 2500 liters/kg. It is a tropical fruit known for its sweet taste and digestive benefits. It is used in both sweet and savory dishes. An alternative to papaya is Melon, with a water footprint of 1000 liters/kg. Melon is a sweet and refreshing fruit that serves as an alternative to papaya, with a lower water footprint.",
    
#     "Potato": "Potato has a water footprint of 500 liters/kg. Potatoes are versatile tubers used in many dishes worldwide, from fries to mashed potatoes. They are high in carbohydrates. An alternative to potato is Sweet Potato, with a water footprint of 400 liters/kg. Sweet potato is a root vegetable with a lower water footprint compared to regular potatoes. It is used in a variety of dishes and has a sweet flavor.",
    
#     "Pumpkin": "Pumpkin has a water footprint of 1200 liters/kg. It is used in soups, pies, and as a side dish. It is rich in vitamins A and C and has a slightly sweet flavor. An alternative to pumpkin is Butternut Squash, with a water footprint of 1000 liters/kg. Butternut squash is a good alternative to pumpkin with a similar taste and texture but slightly less water usage.",
    
#     "Radish": "Radish has a water footprint of 300 liters/kg. Radishes are crisp, peppery root vegetables often used in salads and as a garnish. They have a high water content and low calories. An alternative to radish is Turnip, with a water footprint of 250 liters/kg. Turnip is a root vegetable with a lower water footprint compared to radish, used in similar ways in salads and cooking.",
    
#     "Tomato": "Tomato has a water footprint of 1000 liters/kg. Tomatoes are widely used in cooking for their tangy flavor. They are essential in sauces, soups, and salads. An alternative to tomato is Cherry Tomato, with a water footprint of 800 liters/kg. Cherry tomatoes are a smaller, sweeter alternative to regular tomatoes, with a lower water footprint and similar culinary uses."
# }

# @app.route('/predict', methods=['POST'])
# def predict():
#     file = request.files['file']
#     image = Image.open(file)
#     image = image.resize((224, 224))
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)

#     predictions = model.predict(image)
#     predicted_class = np.argmax(predictions, axis=1)
#     class_name = list(water_footprint_data.keys())[predicted_class[0]]
#     description = water_footprint_data.get(class_name, "Description not available")

#     return jsonify({"product_name": class_name, "description": description})

# if __name__ == '__main__':
#     app.run(debug=True)
