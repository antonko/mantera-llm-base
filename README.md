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

### Ответы на вопросы с помощью RAG

```bash
# Задать вопрос системе
uv run rag.py "Есть ли в отеле детское меню?"

# С дополнительными параметрами
uv run rag.py "Какие развлечения предлагает отель?" --model "gpt-4" --results 8 --verbose
```

### Запуск Telegram-бота

```bash
# Запуск бота
uv run bot.py
```

Перед запуском бота необходимо:

1. Скопировать `.env.example` в `.env`
2. Указать в файле `.env` значения для `OPENAI_API_KEY` и `TELEGRAM_BOT_TOKEN`

## Назначение

Проект разработан для решения задач в области корпоративной системы знаний, позволяя:

- Извлекать информацию с веб-сайтов с помощью crawl4ai
- Сохранять и индексировать данные в векторной базе данных (ChromaDB)
- Обрабатывать запросы к собранной информации с использованием LLM моделей через OpenAI API
- Отвечать на вопросы с использованием технологии RAG (Retrieval-Augmented Generation)

## Стек технологий

- Python 3.10+
- ChromaDB
- Crawl4ai
- OpenAI
- Aiogram 3.x

## Компоненты проекта

- `crawler.py` - Модуль для сканирования и индексации веб-сайтов
- `search.py` - Инструмент для поиска по базе данных
- `rag.py` - Система ответов на вопросы с использованием RAG
- `bot.py` - Telegram-бот с поддержкой истории диалогов и RAG
