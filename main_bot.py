"""
О, великий Бог кода и строк,
Храни наш скрипт от багов и сбоев!
Пусть компилятор будет нам друг,
А память — без утечек и горя.
Даруй нам логику чистую, свет,
Чтоб код исполнялся без всяких бед.
Аминь
"""
# 25.02.59

import json
import string
import asyncio
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nest_asyncio
import logging
import re

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Список стоп-слов для удаления из запросов
STOP_WORDS = {
    "какой", "есть", "колледжа", "колледже", "колледж", "каким", "какoм", "когда", "какого", "долго", "помогает", "поможет",
    "сколько", "что", "в", "к", "а", "можно", "после", "при", "часто", "подробнее", "окажет", "возможность",
    "как", "какие", "кто", "дают", "ли", "где", "оказывают", "работает", "доступен", "нужно", "нужна", "с", "срок",
    "проводится", "проходит", "этого", "почему", "какие", "найти", "какая", "есть", "записаться", "расскажи", "информация",
    "о", "до", "что", "за", "я", "на", "для", "по", "чем", "будет", "где", "скажи", "узнать", "кому", "или", "этом", "это",
    "его", "как", "от", "ты", "чем"
}

# Функция предварительной обработки текста
def preprocess_text(text):
    # Приводим текст к нижнему регистру
    text = text.lower()
    # Удаляем знаки препинания
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Разбиваем текст на слова
    words = text.split()
    # Удаляем стоп-слова
    filtered_words = [word for word in words if word not in STOP_WORDS]
    # Собираем обратно в строку
    return " ".join(filtered_words)

try:
    with open("intent.json", "r", encoding="utf-8") as file:
        data = json.load(file)
except FileNotFoundError:
    logger.error("Файл intent.json не найден. Пожалуйста, загрузите его.")
    raise

# Подготовка данных для сравнения
examples = []
responses = {}

# Обработка структуры JSON
for item in data["intents"]:
    intent = item["intent"].strip()
    responses[intent] = item["answer"]
    for example in item["examples"]:
        examples.append((preprocess_text(example), intent))

# Создаем TF-IDF векторизатор
vectorizer = TfidfVectorizer()
example_texts = [text for text, _ in examples]
example_vectors = vectorizer.fit_transform(example_texts)

# Функция для нахождения максимально похожего примера
def get_best_match(user_input, examples, vectors, vectorizer, threshold=0.5):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, vectors).flatten()
    best_index = similarities.argmax()
    best_score = similarities[best_index]
    return examples[best_index][1] if best_score >= threshold else None  # Возвращаем интент

# Функция получения ответа
def get_response(user_input):
    # Предварительная обработка запроса
    user_input = preprocess_text(user_input)
    # Нахождение наиболее похожего примера
    predicted_intent = get_best_match(user_input, examples, example_vectors, vectorizer)
    if predicted_intent is None:
        return "Извините я не могу ответить на ваш вопрос но вы можете связаться с приемной комиссией по телефону +7-707-846-44-27 / +7-727-302-23-03."
    # Возвращаем соответствующий ответ
    return responses.get(predicted_intent, "Извините, я не нашел ответ на ваш вопрос.")

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Основное меню с новыми кнопками
    reply_keyboard = [["Общая информация", "Учебный процесс"],
                      ["Поддержка студентов", "Поступление и оплата"],
                      ["Карьера и трудоустройство", "Дополнительные возможности"],
                      ["Задать вопрос", "Актуальные новости"]]  # Добавлены новые кнопки
    markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)

    await update.message.reply_text(
        "Привет! Я чат-бот колледжа МАБ. Выберите категорию для получения информации.",
        reply_markup=markup
    )

# Команда /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Вы можете задать мне любой вопрос, просто написав его в сообщении. "
        "Также вы можете использовать кнопки меню для навигации."
    )

# Обработчик текстовых сообщений
async def bot_reaction_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text  # Что написал пользователь
        logger.info(f"[user]: {text}")

        # Обработка основного меню
        if text == "Общая информация":
            reply_keyboard = [["Меню вопросов", "Преимущества"], ["Адрес", "Назад"]]
            markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)
            await update.message.reply_text("Выберите раздел:", reply_markup=markup)

        elif text == "Учебный процесс":
            reply_keyboard = [["Система оценивания", "Практические занятия"],
                              ["Учебная программа", "График обучения"], ["Назад"]]
            markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)
            await update.message.reply_text("Выберите раздел:", reply_markup=markup)

        elif text == "Поддержка студентов":
            reply_keyboard = [["Кураторы", "Психолог"], ["Медпункт", "Студенческие клубы"], ["Назад"]]
            markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)
            await update.message.reply_text("Выберите раздел:", reply_markup=markup)

        elif text == "Поступление и оплата":
            reply_keyboard = [["Экзамены при поступлении", "Цена/оплата"],
                              ["Сроки подачи документов", "Льготы/скидки"], ["Назад"]]
            markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)
            await update.message.reply_text("Выберите раздел:", reply_markup=markup)

        elif text == "Карьера и трудоустройство":
            reply_keyboard = [["Трудоустройство", "Сотрудничество с организациями"], ["Вуз", "Назад"]]
            markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)
            await update.message.reply_text("Выберите раздел:", reply_markup=markup)

        elif text == "Дополнительные возможности":
            reply_keyboard = [["Онлайн-платформа", "Стажировки за границей"],
                              ["Подготовительные курсы", "Назад"]]
            markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)
            await update.message.reply_text("Выберите раздел:", reply_markup=markup)

        # Обработка новых кнопок
        elif text == "Задать вопрос":
            await update.message.reply_text(
                "Вы можете задать любой вопрос прямо здесь. Я постараюсь помочь!"
            )

        elif text == "Актуальные новости":
            await update.message.reply_text(
                "Вот последние новости:\n\n"
                "Следите за новостями: [Актуальные новости](https://www.instagram.com/college_mab)",  # Гиперссылка
                parse_mode="Markdown"  # Указываем формат Markdown
            )

        # Обработка подменю
        elif text in responses:
            reply = responses[text]
            await update.message.reply_text(reply)

        # Возврат к основному меню
        elif text == "Назад":
            reply_keyboard = [["Общая информация", "Учебный процесс"],
                              ["Поддержка студентов", "Поступление и оплата"],
                              ["Карьера и трудоустройство", "Дополнительные возможности"],
                              ["Задать вопрос", "Актуальные новости"]]  # Добавлены новые кнопки
            markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)
            await update.message.reply_text("Вы вернулись в главное меню.", reply_markup=markup)

        else:
            reply = get_response(text)  # Получаем ответ
            await update.message.reply_text(reply)

    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте позже.")

# Основная функция запуска бота
async def main():
    bot_key = '7577210942:AAFMb8XLoxQCsfmuwKLyOJZLn8DXTeKCZSk'
    app = Application.builder().token(bot_key).build()

    # Добавляем обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    # Добавляем обработчик текстовых сообщений
    handler = MessageHandler(filters.TEXT & ~filters.COMMAND, bot_reaction_message)
    app.add_handler(handler)

    # Запуск бота
    await app.run_polling()

# Разрешаем использование асинхронного кода в Colab
nest_asyncio.apply()

# Запуск бота
try:
    loop = asyncio.get_running_loop()
    loop.create_task(main())  # Запускаем main() в существующем event loop
    logger.info("Бот запущен!")
except RuntimeError:
    asyncio.run(main())  # Запуск в обычном Python