ПР-1: Настройка окружения и первый notebook
Байжигит Акаев

# Credit Risk ML

Учебный проект по дисциплине «Машинное обучение»: предсказание просрочки по кредиту (кредитный скоринг).

## Датасет
Give Me Some Credit (Kaggle, соревнование «Give Me Some Credit»).
Нужен файл `cs-training.csv`: скачать с Kaggle и положить в папку `data/`.
Сам файл в репозитории не хранится (он в `.gitignore`).

## Структура проекта
- `data/` — данные (не в git)
- `notebooks/` — notebook'ы с практическими работами
- `src/` — собственный код (например, трансформеры)
- `reports/` — data dictionary и отчёты
- `requirements.txt` — версии библиотек
- `tests/` — тесты собственных трансформеров

## Как запустить
1. Установить Python 3.11+.
2. Создать и активировать окружение (Windows PowerShell):
```
   python -m venv venv
   .\venv\Scripts\Activate.ps1
```
3. Установить библиотеки:
```
   pip install -r requirements.txt
```
4. Положить `cs-training.csv` в папку `data/`.
5. Открыть notebook из папки `notebooks/` в VS Code (ядро `venv`) или Jupyter и выполнить `Run All`.
6. Запустить тесты: `python -m pytest -v`

## Практические работы
| ПР | Тема | Файл |
|---|---|---|
| 1 | Окружение и первичный анализ | `notebooks/01_setup_eda.ipynb` |
| 2 | Разведочный анализ (EDA) | `notebooks/02_eda.ipynb` |
| 3 | Пайплайн предобработки | `notebooks/03_pipeline.ipynb` |
| 4 | Линейная регрессия, градиентный спуск | `notebooks/04_regression.ipynb` |
| 5 | Классификатор, дисбаланс классов | `notebooks/05_classification.ipynb` |
| 6 | Классификация текстов | `notebooks/06_text_classification.ipynb` |
| 7 | Случайный лес, важность признаков, SHAP | `notebooks/07_random_forest.ipynb` |