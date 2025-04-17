import argparse
import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from crawler import WebsiteCrawler

# Загружаем переменные из .env файла
load_dotenv()


def format_search_results(results: dict[str, Any]) -> str:
    """Форматирует результаты поиска в читаемый вид.

    Args:
        results: Словарь с результатами поиска

    Returns:
        Отформатированная строка с результатами

    """
    formatted_output = []

    if not results.get("documents") or not results.get("documents")[0]:
        return "Результаты не найдены"

    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        formatted_output.append(f"{i + 1}. {metadata['title']}")
        formatted_output.append(f"   URL: {metadata['url']}")
        formatted_output.append(
            f"   Оценка: {metadata['score']:.2f}, Глубина: {metadata['depth']}",
        )

        # Получаем первые 350 символов текста для превью
        preview = doc[:150] + "..." if len(doc) > 350 else doc
        formatted_output.append(f"   Фрагмент: {preview}\n")

    return "\n".join(formatted_output)


async def search_database(
    query: str,
    db_path: str = "./chroma_db",
    collection_name: str = "website_data",
    n_results: int = 5,
) -> None:
    """Выполняет поиск по базе данных.

    Args:
        query: Строка запроса
        db_path: Путь к директории ChromaDB
        collection_name: Имя коллекции
        n_results: Количество результатов для отображения

    """
    print(f"Поиск по запросу: '{query}'")
    print(f"База данных: {db_path}, коллекция: {collection_name}")

    # Получаем API ключ из переменных окружения
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Создаем экземпляр краулера с указанным путем к базе данных и API ключом
    crawler = WebsiteCrawler(
        db_path=db_path,
        collection_name=collection_name,
        openai_api_key=openai_api_key,
    )

    # Проверяем количество документов в базе
    doc_count = crawler.get_data_count()
    print(f"Всего документов в базе: {doc_count}")

    if doc_count > 0:
        # Выполняем поиск
        results = crawler.search_by_query(query, n_results=n_results)

        # Выводим отформатированные результаты
        print("\nРезультаты поиска:")
        print(format_search_results(results))
    else:
        print("База данных пуста. Необходимо сначала выполнить сканирование.")


def main() -> None:
    """Точка входа в программу."""
    # Настраиваем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Поиск по базе данных веб-страниц")

    parser.add_argument("query", help="Поисковый запрос")
    parser.add_argument("--db-path", default="./chroma_db", help="Путь к базе данных ChromaDB")
    parser.add_argument("--collection", default="website_data", help="Имя коллекции")
    parser.add_argument(
        "--results",
        type=int,
        default=5,
        help="Количество результатов (по умолчанию: 5)",
    )

    args = parser.parse_args()

    # Запускаем асинхронную функцию поиска
    asyncio.run(
        search_database(
            query=args.query,
            db_path=args.db_path,
            collection_name=args.collection,
            n_results=args.results,
        ),
    )


if __name__ == "__main__":
    main()
