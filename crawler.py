import asyncio
import os
import uuid
from typing import Any

import chromadb
from bs4 import BeautifulSoup
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    ContentTypeFilter,
    DomainFilter,
    FilterChain,
    URLPatternFilter,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()


def extract_title_from_html(html: str) -> str:
    """Извлекает заголовок страницы из HTML-контента с использованием BeautifulSoup.

    Args:
        html: HTML-контент страницы

    Returns:
        Заголовок страницы или "Без названия" если не найден

    """
    if not html:
        return "Без названия"

    soup = BeautifulSoup(html, "lxml")

    # Попытка найти тег title
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    # Попытка найти заголовок h1
    if soup.h1 and soup.h1.string:
        return soup.h1.string.strip()

    # Если ничего не найдено
    return "Без названия"


class WebsiteCrawler:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "website_data",
        openai_api_key: str | None = None,
    ):
        """Инициализация краулера с настройками хранилища ChromaDB.

        Args:
            db_path: Путь к директории для хранения базы данных ChromaDB
            collection_name: Имя коллекции для хранения данных
            openai_api_key: API ключ OpenAI для embedding функции (если не указан,
                            будет использована встроенная функция)

        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # Используем ключ из .env файла, если не передан явно
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        # Настраиваем embedding функцию
        if openai_api_key:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-ada-002",
            )
        else:
            # Используем встроенную функцию, если ключ не предоставлен
            ef = embedding_functions.DefaultEmbeddingFunction()

        # Создаем или получаем коллекцию
        try:
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=ef,
            )
        except NotFoundError:
            # Если коллекция не найдена, создаем новую
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=ef,
            )

        # Выводим информацию об инициализации коллекции
        print(f"Коллекция '{collection_name}' инициализирована, сохранение в '{db_path}'")
        print(f"Текущее количество документов: {self.collection.count()}")

    async def crawl_website(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 100,
        include_external: bool = False,
        keywords: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        allowed_patterns: list[str] | None = None,
        verbose: bool = True,
        check_robots_txt: bool = True,
    ) -> int:
        """Сканирует веб-сайт и сохраняет данные в ChromaDB.

        Args:
            url: URL сайта для сканирования
            max_depth: Максимальная глубина сканирования
            max_pages: Максимальное количество страниц для сканирования
            include_external: Сканировать ли внешние домены
            keywords: Ключевые слова для приоритизации страниц
            allowed_domains: Список разрешенных доменов
            allowed_patterns: Шаблоны URL для включения
            verbose: Выводить ли подробную информацию во время сканирования
            check_robots_txt: Учитывать ли правила robots.txt

        Returns:
            Количество сохраненных страниц

        """
        # Определение базового домена, если не указаны разрешенные домены
        if not allowed_domains:
            from urllib.parse import urlparse

            base_domain = urlparse(url).netloc
            allowed_domains = [base_domain]

        # Настройка фильтров
        filters = []

        # Добавляем фильтр доменов только если НЕ включены внешние сайты
        if not include_external and allowed_domains:
            filters.append(DomainFilter(allowed_domains=allowed_domains))

        if allowed_patterns:
            filters.append(URLPatternFilter(patterns=allowed_patterns))

        # Добавляем фильтр по типу контента (только HTML)
        filters.append(ContentTypeFilter(allowed_types=["text/html"]))

        filter_chain = FilterChain(filters) if filters else None

        # Настройка скорера для приоритизации страниц
        if not keywords:
            keywords = ["информация", "данные", "документ", "руководство"]

        keyword_scorer = KeywordRelevanceScorer(keywords=keywords, weight=0.7)

        # Настройка стратегии сканирования
        crawl_strategy = BestFirstCrawlingStrategy(
            max_depth=max_depth,
            include_external=include_external,
            filter_chain=filter_chain,
            url_scorer=keyword_scorer,
            max_pages=max_pages,
        )

        # Настройка генератора markdown для получения чистого текста
        markdown_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.6, min_word_threshold=30),
            options={"ignore_links": True, "body_width": 0},
        )

        # Настройка конфигурации сканирования
        config = CrawlerRunConfig(
            deep_crawl_strategy=crawl_strategy,
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=True,  # Используем потоковый режим для обработки результатов в реальном времени
            verbose=verbose,
            markdown_generator=markdown_generator,
            check_robots_txt=check_robots_txt,
            remove_overlay_elements=True,  # Удаляем всплывающие элементы и оверлеи
        )

        # Выполнение сканирования
        pages_processed = 0
        async with AsyncWebCrawler() as crawler:
            async for result in await crawler.arun(url, config=config):
                # Проверяем успешность запроса
                if not result.success or (
                    hasattr(result, "status_code") and result.status_code != 200
                ):
                    if verbose:
                        error_msg = getattr(result, "error_message", "Неизвестная ошибка")
                        print(f"Ошибка при обработке {result.url}: {error_msg}")
                    continue

                # Создаем уникальный ID на основе URL
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, result.url))
                page_url = result.url

                # Используем markdown текст вместо HTML для лучшего извлечения информации
                if (
                    hasattr(result, "markdown")
                    and result.markdown
                    and hasattr(result.markdown, "raw_markdown")
                ):
                    page_content = result.markdown.raw_markdown
                else:
                    # Запасной вариант - очищенный HTML
                    page_content = (
                        result.cleaned_html
                        if hasattr(result, "cleaned_html") and result.cleaned_html
                        else result.html
                    )

                # Извлекаем заголовок из очищенного HTML
                page_title = extract_title_from_html(
                    result.cleaned_html if hasattr(result, "cleaned_html") else result.html,
                )

                score = result.metadata.get("score", 0)
                depth = result.metadata.get("depth", 0)

                if verbose:
                    print(f"Глубина: {depth} | Оценка: {score:.2f} | {page_url}")

                # Создаем метаданные для страницы с индексом чанка 0
                page_metadata = {
                    "url": page_url,
                    "title": page_title,
                    "depth": depth,
                    "score": score,
                    "chunk_index": 0,  # Каждая страница сохраняется как один чанк
                    "crawled_at": result.metadata.get("timestamp", ""),
                }

                # Сохраняем страницу как один чанк
                self.collection.add(
                    documents=[page_content],
                    metadatas=[page_metadata],
                    ids=[doc_id],
                )

                pages_processed += 1

        print(f"Сканирование завершено. Обработано {pages_processed} страниц.")
        return pages_processed

    def get_data_count(self) -> int:
        """Возвращает количество документов в коллекции"""
        return self.collection.count()

    def search_by_query(self, query: str, n_results: int = 5) -> dict[str, Any]:
        """Поиск по содержимому в базе данных.

        Args:
            query: Текст запроса
            n_results: Количество результатов для возврата

        Returns:
            Словарь с результатами поиска

        """
        # Явно преобразуем результат из QueryResult в словарь
        query_result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        # Преобразуем в словарь для устранения ошибки типизации
        return dict(query_result)


async def main():
    """Пример использования краулера."""
    # Создаем экземпляр краулера
    crawler = WebsiteCrawler()

    # URL для сканирования
    target_url = "https://romantikarkhyz.ru/"  # Замените на нужный URL

    # Запускаем сканирование
    await crawler.crawl_website(
        url=target_url,
        max_depth=5,
        max_pages=500,
        keywords=["информация", "отель", "номера", "справка", "правила"],
        verbose=True,
        check_robots_txt=True,
    )

    # Проверяем количество сохраненных документов
    doc_count = crawler.get_data_count()
    print(f"В базе данных сохранено {doc_count} документов.")


if __name__ == "__main__":
    asyncio.run(main())
