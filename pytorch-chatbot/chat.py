import random
import json
import logging
import torch
from datetime import datetime
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# ===== log settings=====
logging.basicConfig(
    level=logging.INFO,
    filename="chatbot.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ChatBot")

# bot start log
logger.info("===== BOT STARTED =====")
print("Logging initialized. All conversations will be saved to chatbot.log")

# ===== model init =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

try:
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)
    logger.info("Intents loaded successfully")
except Exception as e:
    logger.error(f"Error loading intents: {str(e)}")
    raise

try:
    FILE = "data.pth"
    data = torch.load(FILE)
    
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]
    
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# ===== chat bot  =====
bot_name = "Sam"
print(f"Let's chat! (type 'quit' to exit) - Bot: {bot_name}")

while True:
    try:
        user_input = input("You: ")
        logger.info(f"USER: {user_input}")
        
        if user_input.lower() == "quit":
            logger.info("User requested quit")
            break

       
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        
        logger.info(f"PREDICTED: tag='{tag}', probability={prob.item():.4f}")

        response = ""
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    print(f"{bot_name}: {response}")
        else:
            response = "I do not understand..."
            print(f"{bot_name}: {response}")
        
        logger.info(f"BOT: {response}")

    except Exception as e:
        logger.error(f"Error during conversation: {str(e)}", exc_info=True)
        print(f"{bot_name}: Sorry, I encountered an error. Please try again.")

logger.info("===== BOT STOPPED =====")
