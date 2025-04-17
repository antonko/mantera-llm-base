import asyncio
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_message_histories.file import FileChatMessageHistory

from rag import RAGAssistant

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем переменные из .env файла
load_dotenv()

# Получаем токен бота из переменных окружения
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Токен бота Telegram не найден. Укажите его в .env файле.")

# Директория для хранения истории диалогов
HISTORY_DIR = Path("chat_histories")
HISTORY_DIR.mkdir(exist_ok=True)


class RagBot:
    def __init__(self):
        """Инициализация бота с RAG системой."""
        self.bot = Bot(token=BOT_TOKEN)
        self.dp = Dispatcher()
        self.rag = RAGAssistant()

        # Регистрация обработчиков сообщений
        self.register_handlers()

    def register_handlers(self):
        """Регистрация обработчиков сообщений."""
        self.dp.message(CommandStart())(self.start_command)
        self.dp.message()(self.process_message)

    async def start_command(self, message: types.Message):
        """Обработчик команды /start."""
        await message.answer(
            "Привет! Я бот, использующий RAG для ответов на вопросы. "
            "Просто напишите свой вопрос, и я постараюсь на него ответить.",
        )

    async def process_message(self, message: types.Message):
        """Обработка входящих сообщений от пользователя."""
        user_id = message.chat.id
        user_question = message.text or ""

        # Показываем индикатор набора текста
        await self.bot.send_chat_action(chat_id=user_id, action="typing")

        # Получаем историю диалога для этого пользователя
        history = self.get_chat_history(user_id)

        # Сохраняем сообщение пользователя в историю
        history.add_message(HumanMessage(content=user_question))

        try:
            # Формируем контекст из предыдущих сообщений
            context = self.get_conversation_context(history)
            enhanced_question = f"{context}\n\nТекущий вопрос: {user_question}"

            # Генерируем ответ с помощью RAG
            answer = await self.rag.generate_answer(
                question=enhanced_question,
                n_results=5,
            )

            # Проверяем, что ответ не None
            if answer is None:
                answer = "Извините, не удалось сгенерировать ответ. Пожалуйста, попробуйте переформулировать вопрос."

            # Сохраняем ответ в историю
            history.add_message(AIMessage(content=answer))

            # Отправляем ответ пользователю
            await message.answer(answer)

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            await message.answer(
                "Извините, произошла ошибка при обработке вашего запроса. "
                "Пожалуйста, попробуйте позже.",
            )

    def get_chat_history(self, user_id: int) -> FileChatMessageHistory:
        """Получает объект истории диалога для указанного пользователя.

        Args:
            user_id: ID пользователя/чата

        Returns:
            Объект истории диалога

        """
        history_file = HISTORY_DIR / f"{user_id}.json"
        return FileChatMessageHistory(file_path=str(history_file))

    def get_conversation_context(self, history: FileChatMessageHistory) -> str:
        """Формирует контекст беседы из истории сообщений.

        Args:
            history: Объект истории диалога

        Returns:
            Строка с контекстом беседы

        """
        messages = history.messages

        # Ограничиваем количество сообщений для контекста (последние 10)
        recent_messages = messages[-10:] if len(messages) > 10 else messages

        context_parts = []
        for msg in recent_messages:
            role = "Пользователь" if isinstance(msg, HumanMessage) else "Ассистент"
            context_parts.append(f"{role}: {msg.content}")

        if context_parts:
            return "История диалога:\n" + "\n".join(context_parts)
        return ""

    async def start(self):
        """Запуск бота."""
        logger.info("Запуск бота...")
        await self.dp.start_polling(self.bot)


async def main():
    """Точка входа в программу."""
    rag_bot = RagBot()
    await rag_bot.start()


if __name__ == "__main__":
    asyncio.run(main())
