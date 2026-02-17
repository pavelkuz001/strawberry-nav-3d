[![CI](https://github.com/pavelkuz001/strawberry-nav-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/pavelkuz001/strawberry-nav-3d/actions/workflows/ci.yml)

# strawberry-nav-3d

Prototype pipeline: **strawberry detection + depth → 3D goal → (optionally) 3D planning → motion control**.

Ключевая идея: репозиторий остаётся “чистым” после `git clone`. Тяжёлые ассеты (веса моделей и т.п.) **не коммитятся**, а лежат в **пользовательском кэше** и скачиваются при первом запуске.

---

## What is implemented (per Tech Spec)

### ✅ A) Detector + depth (from `strawberry_detector`)
- Берём ближайшую ягоду по `closest_strawberry_id`
- Используем данные: `bbox.center_x`, `bbox.center_y`, `depth.center_meters`
- Генерируются артефакты: `*.json`, `*_depth.npy`, маски, визуализация.

### ✅ B) Pixel → metric (camera intrinsics)
Перевод из пиксельных координат (u,v) в метрические (X,Y,Z) относительно камеры:
- `fx = fy = 886.81`, `cx = cy = 512.0` (при 1024×1024)
- `X = (u - cx)/fx * Z`
- `Y = (v - cy)/fy * Z`
- `Z = depth.center_meters`

### ✅ C) Sim2D top-down autonomous demo (по ТЗ)
В `src/sim2d` реализован автономный сценарий “вид сверху”:
- Робот = точка + направление (pose: x,y,theta)
- Цели = ягоды из detector JSON
- Есть цикл, который:
  - обновляет “наблюдение” от детектора с частотой и задержкой
  - пересчитывает цель из (u,v,depth) → world goal
  - обновляет позу робота по одометрии (интеграция команд)
- FSM: **SEARCH → APPROACH → STOP**
  - SEARCH: вращение на месте до появления цели
  - APPROACH: движение к цели
  - STOP: остановка на расстоянии **0.20 m**
- Плавность: торможение начинается за **0.50 m** до стоп-границы

### ✅ D) “Motor backend” abstraction (задел под ROS)
Навигация вызывает слой “моторов” через backend:
- `math` — лимитер скорости/ускорения (pure python)
- `raw` — прямой passthrough (debug)
- `ros` — заглушка (пока как `math`)
- `runbot` — если доступен файл `src_motors_gamepad/.../vel_acc_constraint.py`, использует именно его; иначе падает обратно на pure python

> Сейчас ROS-связи нет: в симуляции используем математику (как и договаривались), но архитектура оставляет место для ROS-реализации без переписывания логики.

### ✅ E) CI smoke test (Ubuntu)
GitHub Actions: install deps → compile → import smoke.

---

## Repo structure (важное)

- `src/`
  - `main.py` — pipeline demo (детектор → цель → дальнейшая логика)
  - `loop.py` — loop-раннер (опционально)
  - `sim2d/` — **TЗ demo: top-down симуляция + FSM + геометрия + motor backend**
- `third_party/strawberry_detector/`
  - детектор (устанавливается editable)
- `results/`
  - локальные выходные файлы (игнорируются git)

---

## Запуск 2D симуляции

Для запуска 2D симуляции (Sim2D), визуализирующей логику робота (Поиск -> Приближение -> Остановка):

### Базовый запуск
```bash
python -m src.sim2d.run --json results/strawberry_in_green_house.json
```

### С рандомизацией положения клубники
Используйте флаг `--random-angle`, чтобы рандомизировать угол положения клубники, сохраняя дистанцию:
```bash
python -m src.sim2d.run --json results/strawberry_in_green_house.json --random-angle
```

### Другие аргументы
- `--headless`: Запуск без визуализации (только консоль).
- `--steps N`: Установить максимальное количество шагов (по умолчанию 3000).
- `--save-fig path/to/image.png`: Сохранить финальное состояние симуляции как изображение.

---

## Quick start (CPU)

### 0) Prerequisites
- Python 3.9+  
- `git`
- Интернет для первого скачивания весов

> Warning `NotOpenSSLWarning (LibreSSL ...)` в macOS может появляться — это **не ошибка** (если всё запускается).

### 1) Clone + venv
```bash
git clone https://github.com/pavelkuz001/strawberry-nav-3d.git
cd strawberry-nav-3d

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell