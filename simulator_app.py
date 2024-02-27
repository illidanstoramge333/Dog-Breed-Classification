import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("ResNet50V2_DataAug.h5")

# Define class names
class_names = {
        "0": "Chihuahua",
        "1": "Japanese Spaniel",
        "2": "Maltese Dog",
        "3": "Pekinese",
        "4": "Shih-Tzu",
        "5": "Blenheim Spaniel",
        "6": "Papillon",
        "7": "Toy Terrier",
        "8": "Rhodesian Ridgeback",
        "9": "Afghan Hound",
        "10": "Basset",
        "11": "Beagle",
        "12": "Bloodhound",
        "13": "Bluetick",
        "14": "Black and Tan Coonhound",
        "15": "Walker Hound",
        "16": "English Foxhound",
        "17": "Redbone",
        "18": "Borzoi",
        "19": "Irish Wolfhound",
        "20": "Italian Greyhound",
        "21": "Whippet",
        "22": "Ibizan Hound",
        "23": "Norweigian Elkhound",
        "24": "Otterhound",
        "25": "Saluki",
        "26": "Scottish Deerhound",
        "27": "Weimaraner",
        "28": "Staffordshire Bullterrier",
        "29": "American Staffordshire Bullterrier",
        "30": "Bedlington Terrier",
        "31": "Border Terrier",
        "32": "Kerry Blue Terrier",
        "33": "Irish Terrier",
        "34": "Norflok Terrier",
        "35": "Norwich Terrier",
        "36": "Yorkshire terrier",
        "37": "Wire haired fox terrier",
        "38": "Lakeland Terrier",
        "39": "Sealyham Terrier",
        "40": "Airedale",
        "41": "Cairn",
        "42": "Australian Terrier",
        "43": "Dandie Dinmont",
        "44": "Boston bull",
        "45": "Miniature Schnauzer",
        "46": "Giant Schnauzer",
        "47": "Standard Schnauzer",
        "48": "Scotch Terrier",
        "49": "Tibetan Terrier",
        "50": "Silky Terrier",
        "51": "Soft Coated Wheaten Terrier",
        "52": "West Highland White Terrier",
        "53": "Lhasa",
        "54": "Flat Coated Retriever",
        "55": "Curly Coated Retriever",
        "56": "Golden retriever", 
        "57": "Labrador retriever",
        "58": "Chesapeake bay retriever",
        "59": "German short haired pointer",
        "60": "Vizsla",
        "61": "English Setter",
        "62": "Irish Setter",
        "63": "Gordon Setter",
        "64": "Brittany Spaniel",
        "65": "Clumber",
        "66": "English Springer",
        "67": "Welsh springer spaniel",
        "68": "Cocker Spaniel",
        "69": "Sussex Spaniel",
        "70": "Irish Water Spaniel",
        "71": "Kuvasz",
        "72": "Schipperke",
        "73": "Groenendael",
        "74": "Malinois",
        "75": "Briard",
        "76": "Kelpie",
        "77": "Komondor",
        "78": "Old English Sheepdog",
        "79": "Shetland Sheepdog",
        "80": "Collie",
        "81":  "Border Collie",
        "82": "Bouvier Des Flandres",
        "83": "Rottweiler",
        "84": "German Shepherd",
        "85": "Doberman",
        "86": "Miniature Pinscher",
        "87": "Greater Swiss Mountain Dog",
        "88": "Bernese Mountain Dog",
        "89": "Appenzeller",
        "90": "EntleBucher",
        "91":  "Boxer",
        "92": "Bull Mastiff",
        "93": "Tibetan MAstiff",
        "94": "French Bulldog",
        "95": "Great Dane",
        "96": "Saint Bernard",
        "97": "Eskimo Dog",
        "98": "Malamute",
        "99": "Siberian Husky",
        "100": "Affenpinscher",
        "101":  "Basenji",
        "102": "Pug",
        "103": "Leonberg",
        "104": "Newfoundland",
        "105": "Great Pyreness",
        "106": "Samoyed",
        "107": "Pomeranian",
        "108": "Chow",
        "109": "Keeshond",
        "110": "Brabancon Griffon",
        "111":  "Pembroke",
        "112": "Cardigan",
        "113": "Toy Poodle",
        "114": "Miniature Poodle",
        "115": "Standard Poodle",
        "116": "Mexican Hairless",
        "117": "Dingo",
        "118": "Dhole",
        "119": "African Hunting Dog",


       

}
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
   
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Dog Breed Classification Simulator")

# Upload an image for classification
uploaded_image = st.file_uploader("Upload a dog image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = tf.image.decode_image(uploaded_image.read(), channels=3)
    image = tf.image.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0) / 255.0

    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    if st.button("Classify"):
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=-1)
        st.write(f"Predicted Breed: {class_names[str(predicted_class[0])]}")
