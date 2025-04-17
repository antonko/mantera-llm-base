import argparse
import asyncio
import os
from datetime import datetime
from typing import Any

import openai
from dotenv import load_dotenv

from crawler import WebsiteCrawler
from search import format_search_results

# Загружаем переменные из .env файла
load_dotenv()


class RAGAssistant:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "website_data",
        openai_api_key: str | None = None,
        model: str = "gpt-4o",
    ):
        """Инициализация RAG ассистента.

        Args:
            db_path: Путь к директории для хранения базы данных ChromaDB
            collection_name: Имя коллекции для хранения данных
            openai_api_key: API ключ OpenAI
            model: Модель OpenAI для генерации ответов

        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = model

        # Используем ключ из .env файла, если не передан явно
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "API ключ OpenAI не найден. Укажите его в .env файле или передайте явно.",
                )

        # Инициализируем клиент OpenAI
        self.client = openai.AsyncOpenAI(api_key=openai_api_key)

        # Инициализируем crawler для работы с базой данных
        self.crawler = WebsiteCrawler(
            db_path=db_path,
            collection_name=collection_name,
            openai_api_key=openai_api_key,
        )

        print(f"RAG ассистент инициализирован с моделью {model}")
        print(f"База данных: {db_path}, коллекция: {collection_name}")
        print(f"Всего документов в базе: {self.crawler.get_data_count()}")

    async def generate_answer(
        self,
        question: str,
        n_results: int = 5,
        verbose: bool = False,
    ) -> str:
        """Генерирует ответ на вопрос с использованием RAG.

        Args:
            question: Вопрос пользователя
            n_results: Количество результатов для извлечения из базы
            verbose: Выводить ли подробную информацию о поиске

        Returns:
            Ответ на вопрос

        """
        # Проверяем, что база не пуста
        if self.crawler.get_data_count() == 0:
            return "База данных пуста. Необходимо сначала выполнить сканирование сайта."

        # Получаем релевантные документы из базы
        search_results = self.crawler.search_by_query(question, n_results=n_results)

        if verbose:
            print("\nРезультаты поиска:")
            print(format_search_results(search_results))

        # Готовим контекст из найденных документов
        context = self._prepare_context(search_results)

        if not context:
            return "Не удалось найти релевантную информацию для ответа на этот вопрос."

        # Отправляем запрос к OpenAI
        system_prompt = f"""
        Ты - полезный ассистент который работает как телеграмм бот, ты отвечаешь на вопросы на основе предоставленной информации.
        Используй ТОЛЬКО информацию из контекста для ответа.
        Если в контексте нет достаточной информации, честно признай, что не можешь ответить на вопрос.
        Отвечай на русском языке кратко и по существу, но можешь дополнительно пояснять и рекомендовать.
        Вместе с ответом выдавай ссылки на документы, из которых была взята информация.
        Сейчас {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}
        """

        user_prompt = f"""
        Вопрос: {question}
        
        Контекст:
        {context}
        """

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        return response.choices[0].message.content.strip()

    def _prepare_context(self, search_results: dict[str, Any]) -> str:
        """Готовит контекст для запроса к модели из результатов поиска.

        Args:
            search_results: Результаты поиска из ChromaDB

        Returns:
            Строка с контекстом для запроса

        """
        context_parts = []

        if not search_results.get("documents") or not search_results.get("documents")[0]:
            return ""

        for i, (doc, metadata) in enumerate(
            zip(search_results["documents"][0], search_results["metadatas"][0]),
        ):
            title = metadata.get("title", "Без названия")
            url = metadata.get("url", "")

            # Формируем часть контекста с метаданными и содержимым
            context_part = f"--- Документ {i + 1} ---\n"
            context_part += f"Название: {title}\n"
            context_part += f"URL: {url}\n"
            context_part += f"Содержимое:\n{doc}\n\n"

            context_parts.append(context_part)

        return "\n".join(context_parts)


async def answer_question(
    question: str,
    db_path: str = "./chroma_db",
    collection_name: str = "website_data",
    model: str = "gpt-4o-mini",
    n_results: int = 5,
    verbose: bool = False,
) -> None:
    """Отвечает на вопрос пользователя.

    Args:
        question: Вопрос пользователя
        db_path: Путь к директории ChromaDB
        collection_name: Имя коллекции
        model: Модель OpenAI для генерации ответов
        n_results: Количество результатов для извлечения из базы
        verbose: Выводить ли подробную информацию о поиске

    """
    print(f"Вопрос: {question}")

    # Получаем API ключ из переменных окружения
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Создаем экземпляр RAG ассистента
    assistant = RAGAssistant(
        db_path=db_path,
        collection_name=collection_name,
        openai_api_key=openai_api_key,
        model=model,
    )

    # Генерируем ответ
    answer = await assistant.generate_answer(
        question=question,
        n_results=n_results,
        verbose=verbose,
    )

    print("\nОтвет:")
    print(answer)


def main() -> None:
    """Точка входа в программу."""
    # Настраиваем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="RAG система для ответов на вопросы")

    parser.add_argument("question", help="Ваш вопрос")
    parser.add_argument("--db-path", default="./chroma_db", help="Путь к базе данных ChromaDB")
    parser.add_argument("--collection", default="website_data", help="Имя коллекции")
    parser.add_argument("--model", default="gpt-4o-mini", help="Модель OpenAI для ответов")
    parser.add_argument(
        "--results",
        type=int,
        default=5,
        help="Количество документов для поиска (по умолчанию: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Выводить подробную информацию о поиске",
    )

    args = parser.parse_args()

    # Запускаем асинхронную функцию
    asyncio.run(
        answer_question(
            question=args.question,
            db_path=args.db_path,
            collection_name=args.collection,
            model=args.model,
            n_results=args.results,
            verbose=args.verbose,
        ),
    )


if __name__ == "__main__":
    main()
