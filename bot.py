import telebot
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error 
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
API_KEY = os.environ.get("api_key")

data = pd.read_csv('students.csv')

# Preprocessing
data['Subjects'] = data['Subjects'].apply(lambda x: x.split(', '))
data['Interest Score'] = data['Interest Score'].apply(lambda x: [int(i) for i in x.split(', ')])
data['Performance Score'] = data['Performance Score'].apply(lambda x: [int(i) for i in x.split(', ')])

# one-hot encoding
mlb = MultiLabelBinarizer()
subjects_encoded = pd.DataFrame(mlb.fit_transform(data['Subjects']), columns=mlb.classes_)

data = pd.concat([data, subjects_encoded], axis=1)

features = ['Age'] + list(mlb.classes_)

#avg score performance 
data['Avg_Performance_Score'] = data['Performance Score'].apply(lambda x: np.mean(x))

# Model training
X = data[features].values
y = data['Avg_Performance_Score'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pytorch tensors convertion
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

#model
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = RegressionModel(X_train.shape[1])


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training process
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

#evaluation step
with torch.no_grad():
    predicted = model(X_test_tensor)
    mse = mean_squared_error(y_test, predicted.numpy())
    #print('Mean Squared Error:', mse)


torch.save(model.state_dict(), 'performance_model.pth')


#telegram bot for interacting with user

bot = telebot.TeleBot(TELEGRAM_TOKEN)
client = OpenAI(api_key=API_KEY)

#openai for chatting
def generate_response(message):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "O‘zbekiston tarixi va madaniyatini o‘rgatish va Kimyo, matematikam fizika va astronomiya xaqida ham malumot berish uchun AI chatbot san. Sen faqat O'zbekiston tarixi xaqida va Kimyo, matematikam fizika va astronomiya xaqida ham javob berishing kerak. Agar boshqa mavzudan savol berilsa men unga javob bera olmayman deb javob qaytarishing kerak!"},
            {"role": "user", "content": message} 
        ]
    )
    return response.choices[0].message.content


def predict_performance_score(name, age, subjects):
    input_data = np.zeros((1, len(features)))
    input_data[0, 0] = age
    for subject in subjects:
        if subject in mlb.classes_:
            input_data[0, features.index(subject)] = 1

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    performance_score = model(input_tensor).item()
    return performance_score


@bot.message_handler(commands=['start'])
def handle_start(message):
    markup = telebot.types.ReplyKeyboardMarkup(row_width=2)
    chat_btn = telebot.types.KeyboardButton('/chat 💬')
    predict_btn = telebot.types.KeyboardButton('/predict_score 🎱')
    markup.add(chat_btn, predict_btn)
    bot.reply_to(message, "Assalomu alaykum. Mening Ismim Zulfiya👩‍🦱 Men sizga:\n ✅ O'zbekiston tarixi\n ✅ Fizika\n ✅ Matematika\n ✅ Astronomiya\n ✅ Kimyo\n xaqida ma'lumot berishga xarakat qilaman. Istalgan savolingizni bering....", reply_markup=markup)

# Function to handle /chat command
@bot.message_handler(commands=['chat'])
def handle_chat_command(message):
    bot.reply_to(message, "Siz chat bo'limidasiz. Istalgan savollaringizni berishingiz mumkin...")

# Function to handle /predict_score command
@bot.message_handler(commands=['predict_score'])
def handle_predict_score_command(message):
    bot.reply_to(message, "Siz o'qish darajasini bilish bo'limidasiz...")
    bot.reply_to(message, "Iltimos, Ismingizni kiriting:")
    bot.register_next_step_handler(message, ask_age)

def ask_age(message):
    chat_id = message.chat.id
    name = message.text
    bot.send_message(chat_id, "Iltimos, Yoshingizni kiriting:")
    bot.register_next_step_handler(message, lambda msg: ask_subjects(msg, name))

def ask_subjects(message, name):
    chat_id = message.chat.id
    age = message.text
    bot.send_message(chat_id, "Iltimos, fanlar nomini vergul bilan ajratilgan holda kiriting (masalan, matematika, tarix, kimyo):")
    bot.register_next_step_handler(message, lambda msg: make_prediction(msg, name, age))

def make_prediction(message, name, age):
    chat_id = message.chat.id
    subjects = message.text.split(", ")
    prediction = predict_performance_score(name, int(age), subjects)
    prediction = round(prediction)
    # You can customize this recommendation based on your specific criteria
    if prediction < 6:
        bot.send_message(chat_id, f"Xurmatli {name}, Sizning o'rtacha bahoingiz: {prediction}.  \nSiz o'qishga ko'proq e'tibor qaratishingiz kerak. Repetitordan yordam so'rashni o'ylab ko'ring.")
    elif prediction < 8:
        bot.send_message(chat_id, f"Xurmatli {name}, Sizning o'rtacha bahoingiz: {prediction}. \nSiz yaxshi harakat qilyapsiz, lekin yaxshilanish uchun xarakat qilib ko'ring. Ko'proq o'rganishndan to'xtamang")
    else:
        bot.send_message(chat_id, f"Xurmatli {name}, Sizning o'rtacha bahoingiz: {prediction}.\nSiz a'lo harakat qilyapsiz. Ishlashda davom eting!" )
  

# Function to handle other messages
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    loading_message = bot.send_animation(message.chat.id, 'https://media.giphy.com/media/xFmuT64Jto3mRO4w3G/giphy.gif')
    user_message = message.text
    response = generate_response(user_message)
    bot.delete_message(message.chat.id, loading_message.message_id)
    bot.reply_to(message, response)

# Start the bot
bot.polling()
