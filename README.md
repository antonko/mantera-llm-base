# Mantera LLM Base

Экспериментальный проект для сканирования веб-сайтов и работы с данными через RAG с использованием LLM моделей.

## Запуск проекта

### Сканирование веб-сайта

```bash
# Запуск сканирования
uv run crawler.py
```

### Поиск по базе данных

```bash
# Базовый поиск
uv run search.py "детское меню"

# Поиск с дополнительными параметрами
uv run search.py "конференц-зал" --db-path "./chroma_db" --collection "website_data" --results 10
```

## Назначение

Проект разработан для решения задач в области корпоративной системы знаний, позволяя:

- Извлекать информацию с веб-сайтов с помощью crawl4ai
- Сохранять и индексировать данные в векторной базе данных (ChromaDB)
- Обрабатывать запросы к собранной информации с использованием LLM моделей через OpenAI API

## Стек технологий

- Python 3.10+
- ChromaDB
- Crawl4ai
- OpenAI

## Компоненты проекта
