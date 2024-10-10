from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import re
import torch
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration for SQLAlchemy
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/userdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Image Classifier Model
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = load_model('image_classifier_model.h5')

# Define class labels for image classification
class_labels = {
    0: "Acne", 1: "Lichen", 2: "Psoriasis", 3: "Melanoma", 4: "Basal Cell Carcinoma",
    5: "Squamous Cell Carcinoma", 6: "Fungal Infections", 7: "Impetigo", 8: "Dermatitis", 9: "Urticaria",
    10: "Tinea", 11: "Vitiligo", 12: "Actinic Keratosis", 13: "Folliculitis", 14: "Hives",
    15: "Cellulitis", 16: "Lichen Planus", 17: "Contact Dermatitis", 18: "Seborrheic Dermatitis", 19: "Rosacea",
    20: "Atopic Dermatitis", 21: "Warts", 22: "Melanocytic Nevus", 23: "Benign Keratosis", 24: "Eczema",
    25: "AIDS", 26: "Reiter Syndrome", 27: "Pityriasis rosea", 28: "Seborrheic Keratosis", 29: "Benign Tumours",
    30: "Chondro dermatitis", 31: "Cylindroma", 32: "Dermatofibroma", 33: "Epidermal-cyst", 34: "Keloids",
    35: "Kerato acantho", 36: "Neuro fibroma", 37: "Nevus-sebaceo", 38: "Pilar-cyst", 39: "Poro keratosis",
    40: "Skin-tags-polyps", 41: "Syringoma", 42: "Sebaceous hyperplasia", 43: "Tinea Ringworm", 44: "Perleche",
    45: "Herpes", 46: "Molluscum", 47: "Hidradentitis", 48: "Milia", 49: "Perioral Dermatitis",
    50: "Sebaceous gland", 51: "Hyperhidrosis", 52: "Rhinophyma", 53: "Actinic-cheilitis", 54: "Bowens disease",
    55: "Cutaneous horn", 56: "CTCL", 57: "Granulation Tissue", 58: "Leukoplakia", 59: "Lymphomatoid",
    60: "Metastasis", 61: "Verrucous Carcinoma", 62: "Ichthyosis", 63: "Bullous-perphigoid", 64: "Benign familial chronic pemphigus",
    65: "Darier's Disease", 66: "Grover's disease", 67: "Diabetic Bullae", 68: "Epidermolysis bullosa", 69: "Dermatitis herpetiformis",
    70: "Leprosy", 71: "Sycosis Barbae", 72: "Fissure", 73: "Staphylococcal", 74: "Pseudomonas folliculitis",
    75: "Stasis dermatitis", 76: "Neurotic excoriation", 77: "Lichen simplex chronicus", 78: "Dyshidrosis", 79: "Desquamation",
    80: "Chapped-fissured-feet", 81: "Viral-exanthems", 82: "Scarlet-fever", 83: "Minocycline-pigmentation", 84: "Fixed-drug-eruption",
    85: "Erythema-infectiosum", 86: "Enterovirus", 87: "Drug-eruptions", 88: "Acne-keloidalis", 89: "Alopecia-areata",
    90: "Folliculitis-decalvans", 91: "Hot-comb-alopecia", 92: "Pseudopelade", 93: "Trichotillomania", 94: "Eruptive-folliculitis",
    95: "Genital-warts", 96: "Syphilis", 97: "Sun-damaged", 98: "Pseudo-porphyria", 99: "Sunburn",
    100: "Porphyria", 101: "Phototoxic-reactions", 102: "Polymorphous-light-eruption", 103: "Lupus", 104: "Acrocyanosis",
    105: "Chilblains-perniosis", 106: "Dermatomyositis", 107: "Morphea", 108: "Scleroderma", 109: "Atypical-nevi",
    110: "Blue-nevus", 111: "Congenital-nevus", 112: "Lentigo-maligna", 113: "Malignant-melanoma", 114: "Nevus-veg-pigment",
    115: "Nevus-spilus", 116: "Alopecia areata", 117: "Chronic paronychia", 118: "Distal subungual onychomycosis", 119: "Habit-tic deformity",
    120: "Ingrown nail", 121: "Koilonychia", 122: "Nail fungus", 123: "Allergic contact dermatitis", 124: "Cosmetic fragrance allergy",
    125: "Irritant contact dermatitis", 126: "Metal dermatitis", 127: "Rhus dermatitis", 128: "Shoe allergy", 129: "Lichen sclerosus",
    130: "Lichen striatus", 131: "Biting infection", 132: "Biting insects", 133: "Acute paronychia", 134: "Duck itch",
    135: "Lyme disease", 136: "Leishmaniasis", 137: "Pubic lice", 138: "Scabies", 139: "Tick bite",
    140: "Trichodysplasia spinulosa", 141: "Seborrheic keratoses", 142: "Angioedema", 143: "Cholinergic urticaria", 144: "PUPPP",
    145: "Urticaria vasculitis", 146: "Angiokeratoma", 147: "Hemangioma", 148: "Pyogenic granuloma", 149: "Telangiectasia",
    150: "Venous malformations", 151: "Atrophie blanche", 152: "Erythema multiforme", 153: "Henoch-Schonlein purpura", 154: "Schamberg's disease",
    155: "Vasculitis", 156: "Smallpox", 157: "Varicella", 158: "Warts (common)", 159: "Chickenpox",
    160: "Balanitis", 161: "Candidal vulvitis", 162: "Intertrigo penis"

}

# Global variables for the QA chain and other components
qa_chain = None
question_count = 0  # Counter for the number of questions asked

# Prompt template for QA bot
custom_prompt_template = """Answer the following question using the given context.
Context: {context}
Question: {question}
Helpful answer:
"""

def set_custom_prompt():
    """Prompt template for QA retrieval."""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    """Load the language model."""
    llm = CTransformers(
        model="TheBloke/llama-2-7b-chat-GGML",
        model_type="llama",
        max_new_tokens=256,
        temperature=0.7,
        n_gpu_layers=8,
        n_threads=24,
        n_batch=1000,
        load_in_8bit=True,
        num_beams=1,
        max_length=256,
        clean_up_tokenization_spaces=False
    )
    return llm

from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.cached(timeout=300, key_prefix='faq_cache')
def get_faq_response(question):
    return qa_chain({'query': question})

def retrieval_qa_chain(llm, prompt, db):
    """Create a RetrievalQA chain."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

def initialize_qa_bot():
    """Initialize the QA bot and store it in a global variable."""
    global qa_chain
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})

    try:
        print("Loading FAISS database...")
        faiss_path = os.getenv('FAISS_DB_PATH', 'vectorstores/db_faiss')
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        print("FAISS database loaded successfully.")
    except FileNotFoundError:
        print("FAISS index not found. Please create the FAISS index first.")
        return None
    except Exception as e:
        print(f"Error loading FAISS database: {e}")
        return None

    try:
        print("Loading LLM...")
        llm = load_llm()
        print("LLM loaded successfully.")
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return None

    try:
        print("Setting custom prompt...")
        qa_prompt = set_custom_prompt()
        print("Creating QA chain...")
        qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
        print("QA chain created successfully.")
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html exists in the 'templates' folder

@app.route('/ask', methods=['POST'])
def ask_question():
    global question_count
    user_input = request.form['query']
    
    if not is_valid_query(user_input):
        return jsonify({"response": "Nothing matched. Please enter a valid query."})
    
    if qa_chain is None:
        return jsonify({"response": "Failed to initialize QA bot."})
    
    try:
        res = qa_chain.invoke({'query': user_input})
        answer = res.get("result", "No answer found.")
        question_count += 1
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"response": f"Error processing the query: {e}"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_probability = np.max(predictions)

        threshold = 0.6

        if predicted_probability < threshold:
            predicted_label = 'Healthy Skin or Not a Valid Disease Image'
        else:
            predicted_label = class_labels.get(predicted_class, 'Unknown')

        return jsonify({'predicted_class': predicted_label}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def is_valid_query(query):
    """Check if the query is valid."""
    if not query or query.isspace():
        return False
    if not re.search(r'[a-zA-Z0-9]', query):
        return False
    return True

if __name__ == '__main__':
    initialize_qa_bot()  # Initialize the QA bot when the app starts
    app.run(host='0.0.0.0', debug=True, port=5000)  #app.run(host='0.0.0.0', port=5000)

