# auditenv. RL-среда для ИИ-аудитора

Офлайн-среда для дообучения и проверки ИИ-аудитора по кейсу FinTech ([docs/CASE.md](docs/CASE.md)).

- **Генератор.** По `seed` и `config` строит вымышленную компанию за один месяц 2026 года: счета клиентам и поставщиков, банковские платежи, курсы и книгу проводок.
- **Ошибки.** В книгу вносятся скрытые ошибки четырёх типов: дубль счёта поставщика, ошибка разнесения платежа, неверный курс, ошибка периода. Также два связанных сценария, C1 и C2.
- **Двойники.** Рядом лежат корректные записи, похожие на ошибки.
- **Агент** расследует книгу через пять инструментов `reset/step` и сдаёт выводы с доказательствами.
- **Оценщик** работает вне процесса агента и сверяет выводы с закрытым эталоном строгим правилом кейса §05.

Только стандартная библиотека Python 3.11, без сети, GPU и внешних API. **Все компании, люди, счета, платежи и курсы вымышлены.**

**Лицензия и зависимости.** Код под лицензией MIT, текст в [LICENSE](LICENSE). Зависимостей для запуска нет (`dependencies = []` в `pyproject.toml`), нужен Python 3.11 или новее; необязательный `pytest` для разработки закреплён на 8.3.4. Образ Docker закреплён на `python:3.11.11-slim-bookworm` и его digest `sha256:081075da…` (`Dockerfile`).

## Для жюри

Свою политику можно подключить тремя способами. Все команды запускаются из корня репозитория. Примеры этого раздела выполняет тест `tests/test_readme_jury.py`.

**1. Python в одном процессе.** Политика это ваш код, который вызывает `reset(seed, config)` и затем `step(action)` до submit. Пакет не устанавливается, поэтому код запускается файлом в корне репозитория, через `python3 - < файл` или с `PYTHONPATH=.`:

```python
from auditenv.env import AuditEnv

env = AuditEnv("train")                       # "eval": политика видит награду 0 и пустой info
observation, info = env.reset(7, {})          # seed и config генератора (раздел 5)
episode_id = observation["manifest"]["episode_id"]
action = {"action": "list_records", "table": "ledger", "limit": 50}
while True:
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
    cursor = observation["result"]["next_cursor"]
    if cursor is not None:                    # следующая страница книги
        action = {"action": "list_records", "table": "ledger", "cursor": cursor, "limit": 50}
    else:                                     # книга прочитана: здесь ваша политика решает, что сдать
        action = {"action": "submit", "findings": {"episode_id": episode_id, "findings": []}}
print(episode_id, observation["steps_left"], reward, info["grading"])
```

Вывод:

```text
EGGYPY 91 -1.0 {'tp': 0, 'fp': 0, 'fn': 4, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
```

- **Эпизод.** Это тот же `EGGYPY`, что в разделах 2 и 3: seed 7, config по умолчанию, в книге 4 ошибки. Пример читает книгу за 8 шагов и сдаёт пустой ответ, поэтому награда −1, а шагов осталось 91 из 100.
- **Контракт.** `step` возвращает `(observation, reward, terminated, truncated, info)`. Действия и лимиты описаны в разделе 4, config в разделе 5. В режиме `eval` политика видит награду 0 и пустой `info`, а оценка остаётся в `env.grader_log`.
- **Только для отладки.** Политика в том же процессе может прочитать закрытый эталон, поэтому оценивать её нужно третьим способом.

**2. Отдельный процесс, `--policy-cmd`.** Политика читает из stdin и пишет в stdout по строке JSON. Протокол описан в разделе 4, пример лежит в `examples/my_policy.py`:

```bash
python3 -m auditenv.run --policy-cmd "python3 examples/my_policy.py" --config '{}' --seed 7 --mode train
```

Раннер печатает одну строку отчёта, как в разделе 3. В ней `"policy": "external"`, `"steps": 2`, `"terminated": true` и `"reward": -1.0`: пример сдаёт пустой ответ на том же `EGGYPY`. Вне Docker политика работает от вашего пользователя и тоже может прочитать эталон.

**3. Закрытый прогон набора в Docker.** Так политику оценивают. Она работает без сети от пользователя `auditpolicy` и не видит ни эталон, ни файл набора. `hidden_set.json` это набор в формате `configs/dev_v2.json`, `{version, split, episodes: [{seed, config}]}`:

```bash
docker build -t auditenv:dev .
mkdir -m 755 policy && cp examples/my_policy.py policy/ && chmod -R a+rX policy
mkdir -m 700 jury_in jury_out && cp hidden_set.json jury_in/
docker run --rm --network none -v "$PWD/policy:/policy:ro" -v "$PWD/jury_in:/in:ro" -v "$PWD/jury_out:/out" auditenv:dev python3 -m auditenv.run --policy-cmd "python3 /policy/my_policy.py" --config /in/hidden_set.json --mode eval --sandbox required --trajectories /out/trajectories.jsonl --grader-log /out/grader_log.jsonl
```

- **Вывод.** Раннер печатает строку на эпизод и строку итога с P, R, F1 и средней наградой. Траектории и журнал оценщика лежат в `jury_out`, их можно читать без `sudo`.
- **Своя политика.** Её файлы кладутся в `policy/` вместо `examples/my_policy.py`. Имя файла политики пишется дважды, в `cp` и в `--policy-cmd`.
- **Подробности.** Права папок и ключи строки итога описаны в разделе 4, «Закрытый прогон жюри». CI запускает эти команды дословно на наборе из двух эпизодов dev.

## 1. Одна команда проверки

```bash
make verify
```

Скрипт `scripts/verify.sh` выполняет три шага:
1. юнит-тесты, 479 тестов на stdlib `unittest`;
2. стресс по одному seed на каждый из 11 config `configs/stress_v3.json`: генератор, строгий верификатор V, oracle 1.0, побайтная воспроизводимость в свежих процессах при `PYTHONHASHSEED` 0 и 1. Итоговая строка стресса это говорит прямо: `"determinism": "11 of 11 episodes byte-identical …"`;
3. последняя строка вывода `verify: OK`.

**Какие тесты пропускаются и где идут.** Каждый тест выполняется хотя бы в одном шаге CI.
- **Без root** пропускаются 10 тестов песочницы: им нужен пользователь `auditpolicy` из Docker-образа. Они идут в Docker-шаге CI. Это класс `WithRoot` в `tests/test_sandbox.py` с проверкой закрытых папок образа, `tests/test_rehearsal_sandbox.py` и по одному тесту `test_in_docker_…` в `tests/test_policy_cmd.py` и `tests/test_episode_sets.py`.
- **В Docker** пропускаются 5 тестов. Они идут в первом шаге CI, `make verify` на Python 3.11.
  - 4 теста режима без песочницы: там песочница есть всегда. Это класс `WithoutRoot` в `tests/test_sandbox.py` (2 теста), `test_eval_without_the_sandbox_warns_in_one_line_and_still_runs` и `test_sandbox_required_without_root_refuses_with_one_message`.
  - `test_the_closed_run_is_the_command_of_the_ci_step` в `tests/test_readme_jury.py`: папка `.github` в образ не копируется.

Детектор QA на старом эталоне `tests/fixtures/golden_v0.1` в единую команду не входит. Он запускается отдельно, `make qa`, и печатает замечания про прежнюю схему эталона: это ожидаемо и на проверку не влияет.

Тот же скрипт в Docker, без сети. Так он идёт в CI вместе со всеми тестами песочницы:

```bash
make docker-verify
```

## 2. Эпизод по seed и config

```bash
python3 -c "from auditenv.generate import generate_episode; generate_episode(7, {}, 'out/ep7')"
```

В `out/ep7` появляются две папки:
- `public/` с файлами `manifest.json`, `policy.json`, `ledger.csv`, `invoices.jsonl`, `payments.jsonl`, `fx.csv` в схеме кейса §03;
- `private/` с файлами `truth.json` (эталон, его читает только оценщик) и `provenance.json` (история генерации и внесённых изменений для разбора, оценщик её не читает).

Тот же `seed` и `config` дают те же файлы побайтно. Config `{}` это config по умолчанию: от 150 до 200 документов и по одной ошибке каждого типа.

**Дата проводки по `policy.json`.** Счёт проводится по местной дате компании для своего `issued_at`, платёж по местной дате своего `timestamp` (`posting_date_rule`). Курс берётся за ту же дату, в выходной или праздник из `fx.holidays` за предыдущий рабочий день (`fx.date_rule`, `fx.non_business_fallback`). Дата на документе в поясе контрагента может отличаться на день. Например, счёт поставщика от 16 июня 23:05 по его времени (+02:00) в компании с +03:00 проводится 17 июня по курсу 17 июня, и это не ошибка. Время строк `ledger.csv` всегда 12:00 по времени компании и условное, период определяется по дате.

Независимая проверка эпизода:

```bash
python3 -m auditenv.verify out/ep7
```

Верификатор V написан отдельно от генератора. Он печатает по строке JSON на каждую найденную проблему, а на корректном эпизоде ничего не печатает и выходит с кодом 0.

## 3. Четыре политики жюри

Встроенные политики кейса §06: `random` (случайный исследователь с фиксированным seed), `reconcile` (простая сверка), `flag_all` (атака «пометить всё»), `oracle` (только для оценщика). Готовый эпизод:

```bash
python3 -m auditenv.run --policy oracle --mode eval --episode-dir out/ep7
```

Вывод это одна строка JSON:

```json
{"code_version": "1.0.0", "config_sha256": "282e69dc9bf14e7fe9e29a79ebd6c61ff6fbd9cde81b706c7e8b61ee31e04c3a", "episode_id": "EGGYPY", "f1": 1.0, "fn": 0, "fp": 0, "inspection_steps": 0, "mode": "eval", "policy": "oracle", "policy_seed": 0, "precision": 1.0, "recall": 1.0, "reward": 1.0, "seed": 0, "seen_reward": 0.0, "steps": 1, "terminated": true, "tp": 4, "truncated": false, "wall_time_s": 0.001}
```

Эпизод прямо из генератора, через `env.reset(seed, config)`:

```bash
python3 -m auditenv.run --policy reconcile --mode eval --config '{"scenarios": {"C1": 1}}' --seed 7
```

Поля вывода:

| поле | что значит |
|---|---|
| `reward` | настоящая награда оценщика |
| `seen_reward` | что видела политика: в режиме `eval` всегда 0 (кейс §04), в режиме `train` терминальная награда |
| `tp`, `fp`, `fn`, `precision`, `recall`, `f1` | оценка выводов. При неверной структуре submit они `null`, награда −1, а поле `error` даёт код, например `episode_id_mismatch` |
| `steps` | сколько строк политики раннер отдал в `env.step`, submit тоже |
| `inspection_steps` | шаги до submit, штраф 0.002 за каждый |
| `terminated`, `truncated` | эпизод закончился submit или обрезан (лимит времени, 100 шагов, политика закрыла вывод) |
| `seed` | seed эпизода для `env.reset`. При `--episode-dir` он не используется и равен значению `--seed`, по умолчанию 0 |
| `policy`, `policy_cmd` | имя встроенной политики или `external` и команда для `--policy-cmd` |
| `policy_seed` | seed процесса встроенной политики (`random`), у внешней не используется |
| `mode` | `train` или `eval` |
| `sandbox` | `uid`: политика работала от `auditpolicy`; `none`: без песочницы. У oracle поля нет, он работает в процессе оценщика |
| `wall_time_s` | время эпизода после reset, в секундах |
| `code_version` | версия кода `auditenv` |
| `config_sha256` | sha256 нормализованного config эпизода, как в `provenance.json` и `index.json`; у готового эпизода без `provenance.json` `null` |
| `episode_set` | версия набора, если `--config` дал файл набора |

`--trajectories PATH` дописывает траекторию в файл, `--grader-log PATH` дописывает журнал оценщика. Seed и config не попадают в командную строку процесса политики.

Четыре политики жюри и диагностический spray (раздел 7) на открытых train и dev разом. Команда перезаписывает `docs/METRICS.md`:

```bash
make metrics
```

Короткое демо на тестовом эпизоде `tests/fixtures/mini_v03`, траектории пишутся в `runs/`:

```bash
make demo
```

## 4. Протокол для внешней политики

Политика это отдельный процесс. Среда пишет ему в stdin по одной строке JSON на сообщение и читает из stdout по одной строке JSON на действие. Минимальная политика, `examples/my_policy.py`:

```python
import json
import sys

def send(action):
    print(json.dumps(action), flush=True)     # flush обязателен: без него раннер ждёт до лимита времени

first = json.loads(sys.stdin.readline())       # после reset: {observation, info}
episode_id = first["observation"]["manifest"]["episode_id"]
send({"action": "list_records", "table": "ledger", "limit": 2})
reply = json.loads(sys.stdin.readline())       # после шага: {observation, reward, terminated, truncated, info}
send({"action": "submit", "findings": {"episode_id": episode_id, "findings": []}})
final = json.loads(sys.stdin.readline())       # ответ на submit, эпизод окончен
```

Эта политика лежит в репозитории как `examples/my_policy.py`. Свои политики репозитория делают то же через `policies/stdio.py` (`serve`).

**Своя политика одной командой.** Из корня репозитория:

```bash
python3 -m auditenv.run --policy-cmd "python3 examples/my_policy.py" --episode-dir tests/fixtures/mini_v03 --mode train
```

- **Лимиты и журналы те же, что у встроенных политик:** 100 шагов, 60 секунд, журнал оценщика `--grader-log` и траектории `--trajectories`.
- **Флаги.** `--policy` и `--policy-cmd` взаимоисключающие, нужен ровно один. Вместо `--episode-dir` можно передать `--config` с `--seed N`. `--config` принимает строку JSON (`'{}'`), путь к файлу с одним config или файл набора эпизодов вида `configs/dev_v2.json`. С набором и `--seed` берётся этот эпизод набора, без `--seed` идут все эпизоды набора по очереди, по строке на эпизод и строка итога.
- **Конец эпизода.** После submit раннер отправляет политике последний ответ, закрывает её stdin, ждёт до 2 секунд и завершает процесс. Лимит времени на эпизод 60 секунд, его меняет `--time-limit S`.
- **Остановка раннера.** На Ctrl-C, SIGTERM, SIGHUP или SIGQUIT (`Ctrl-\`) раннер в любой момент прогона, включая ожидание выхода политики после эпизода, делает одно и то же:
  - убивает группу процессов политики. В Docker это все процессы пользователя `auditpolicy`, вне Docker потомок, который вышел из группы через `setsid`, остаётся жить;
  - удаляет временные папки, в том числе папку проверки набора с уже собранными эпизодами;
  - печатает в stderr одну строку `auditenv.run: stopped by signal N`, без трассировки Python;
  - выходит с кодом 128 + номер сигнала: 130 на Ctrl-C, 143 на SIGTERM, 129 на SIGHUP, 131 на SIGQUIT.

  Сигнал, который был проигнорирован при запуске, например под `nohup`, остаётся проигнорированным. Чего раннер не может:
  - **`kill -9`** перехватить нельзя: политика и временные папки остаются. В Docker их убирает остановка контейнера;
  - **прерванный эпизод.** Если сигнал пришёл до записи журналов или посреди неё, эпизода нет ни в обычных файлах журналов, ни в отчёте, даже если он уже оценён: начатая запись обрезается до состояния перед эпизодом. FIFO обрезать нельзя, в нём может остаться часть эпизода. Если сигнал пришёл после записи журналов, но до строки отчёта, эпизод есть в обоих журналах, а строки отчёта для него нет.
- **Вывод в канал.** Если читатель stdout ушёл, например `| head -1`, прогон останавливается тихо, как от SIGPIPE, с той же уборкой и кодом 141. Если stdout закрыт с самого запуска или запись в него не удалась (полный диск), раннер выходит с одной строкой и кодом 2.
- **Журналы.** `--trajectories` и `--grader-log` дописываются в конец файла. Обычный файл получает права 600, даже если он был до запуска, когда раннер его владелец или root. Чужой файл, в который можно только дописывать, и устройства вроде `/dev/null` сохраняют свои права. Если папки журнала нет, раннер сразу выходит с одной строкой ошибки и кодом 2. Если запись в журнал не удалась, например диск полон, раннер обрезает оба журнала до состояния перед этим эпизодом и выходит с одной строкой и кодом 2: журналы кончаются целыми строками и согласованы, а строка ошибки называет эпизод, перед которым они кончаются. Если места не хватило генератору или папке политики, тоже одна строка и код 2. Ошибки раннер, в том числе ошибки флагов командной строки, пишет прямо в stderr: даже если stderr никто не читает (`2>&1 | true`), код выхода остаётся 2.
- **Пути.** Относительный путь к существующему файлу в команде раннер делает абсолютным, потому что политика стартует в пустой временной папке.

В Docker политика работает от пользователя `auditpolicy` и не видит `private/`. Путь к своей политике подставьте вместо `examples/my_policy.py`:

```bash
docker build -t auditenv:dev .
docker run --rm --network none -v "$PWD/examples/my_policy.py:/policy/my_policy.py:ro" auditenv:dev python3 -m auditenv.run --policy-cmd "python3 /policy/my_policy.py" --config '{}' --seed 7 --mode eval --sandbox required
```

**Закрытый прогон жюри.** Делайте его только в Docker-образе. Одна команда прогоняет весь скрытый набор и пишет траектории и журнал оценщика в смонтированную папку:

```bash
mkdir -m 755 policy && cp examples/my_policy.py policy/ && chmod -R a+rX policy
mkdir -m 700 jury_in jury_out && cp hidden_set.json jury_in/
docker run --rm --network none -v "$PWD/policy:/policy:ro" -v "$PWD/jury_in:/in:ro" -v "$PWD/jury_out:/out" auditenv:dev python3 -m auditenv.run --policy-cmd "python3 /policy/my_policy.py" --config /in/hidden_set.json --mode eval --sandbox required --trajectories /out/trajectories.jsonl --grader-log /out/grader_log.jsonl
```

- **Три папки.**
  - `policy`: политика, только для чтения. Политика из нескольких файлов кладётся туда же целиком. `chmod -R a+rX` нужен, если ваш umask 077: иначе `cp` сделает файлы недоступными для `auditpolicy`. Имя файла политики пишется дважды, в `cp` и в `--policy-cmd`.
  - `jury_in`: файл набора `hidden_set.json` в формате `configs/dev_v2.json`, `{version, split, episodes: [{seed, config}]}`. `version` обязателен, seed неотрицательные и не повторяются. До первого эпизода раннер проверяет config каждой записи и один раз собирает генератором каждый эпизод, который пойдёт в прогон: так ловятся и config, которые отвергает только генератор, например книга слишком мала для заказанных ошибок. Если запись неверна, прогон не начинается и ничего не пишет, а ошибка называет номер записи и её seed, у повтора оба номера.
  - `jury_out`: журналы.
- **Почему набор не в папке политики.** По seed и config политика могла бы сама собрать эпизод вместе с эталоном. Папка `jury_in` создана с правами 700 и принадлежит вам, поэтому раннер от root её читает, а `auditpolicy` нет.
- **Журналы.** Раннер пишет `jury_out` с правами 600 и в конце отдаёт файлы владельцу папки `jury_out`, то есть вам: `sudo` не нужен, а `auditpolicy` их не читает.
- **Вывод:** строка JSON на каждый эпизод и строка итога с ключами `episode_set`, `split`, `episodes`, `code_version`, `invalid`, `truncated`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_reward`, `mean_steps`. `truncated` это число эпизодов без submit: лимит времени, 100 шагов или политика закрыла вывод. По правилам кейса такой эпизод оценивается как пустой ответ, поэтому на чистой компании он даёт +0.8, и по F1 не видно, что ответа не было. P, R и F1 микро по эпизодам, `mean_reward` точное среднее наград. `invalid` это число эпизодов, где `findings.json` отклонён из-за неверной структуры: у них награда −1, а в строке эпизода `tp`, `fp`, `fn` равны `null`. В итог такой эпизод входит так же, как в `docs/METRICS.md`: TP 0, его выводы как FP, но не меньше одного, и все ошибки эталона как FN. Поэтому неверный пустой ответ на чистой компании не выглядит идеальным, F1 у такого набора 0.
- **Почему только в Docker.** Вне Docker политика работает от того же пользователя, что и раннер, и может прочитать `private/`, то есть эталон. Поэтому `--mode eval` без песочницы печатает в stderr предупреждение в одну строку. Сам прогон при этом идёт: так работают наши скрипты метрик и демо.
- **Как устроено в образе.** Раннер работает от root, политика от `auditpolicy`. `--sandbox required` без песочницы сразу отказывается с одним сообщением.
- **Проверка в CI.** Эти же команды с примером политики и набором из двух эпизодов dev запускаются в CI после сборки образа. CI проверяет три вещи:
  - в `jury_out` есть траектории обоих эпизодов и журнал оценщика, и их можно прочитать без `sudo`;
  - пользователь 10002, от которого работает политика, получает `PermissionError` и на файле набора, и на журналах в смонтированных папках;
  - то же проверяет тест в Docker.

Тот же раннер доступен из Python. Пакет не устанавливается, поэтому код ниже запускается из корня репозитория: файлом в корне, через `python3 - < файл` или с `PYTHONPATH=.`:

```python
import sys
from pathlib import Path
from auditenv.env import AuditEnv
from auditenv.runner import run_subprocess_episode

env = AuditEnv("train")                       # "eval" прячет награду от политики
Path("runs").mkdir(exist_ok=True)
with open("runs/my_policy.jsonl", "w", encoding="utf-8") as trajectory:
    result = run_subprocess_episode([sys.executable, str(Path("examples/my_policy.py").resolve())], env, 0,
                                    {"episode_dir": "tests/fixtures/mini_v03"}, trajectory=trajectory)
print(result, env.grader_log[-1]["reward"])
```

- **Путь к политике нужен абсолютный.** Раннер запускает процесс политики в пустой временной папке, а не в папке репозитория.
- **Окружение политики минимальное:** `PATH`, `LANG`, `PYTHONPATH` с корнем репозитория и `PYTHONHASHSEED`, если он задан. Вывод политики в stderr отбрасывается.
- **Конец эпизода без submit.** Если политика упала, закрыла stdout или молчит дольше 60 секунд, эпизод обрезается (`truncated`). Ответ считается пустым, со штрафом за все 100 шагов: −1 на книге с ошибками и +0.8 на чистой. Так же кончается сотый шаг без submit.
- **Config.** Вместо `{"episode_dir": ...}` можно передать config генератора, например `{}`. Тогда эпизод строится по `seed`, это третий аргумент.
- **Обвязка в стиле Gymnasium.** `env.reset(seed=7, options={})` равносилен `env.reset(7, {})`, config и `options` вместе дают `ValueError`, а `env.close()` ничего не делает и его можно вызывать повторно.

Ниже настоящий обмен политики того же устройства на `tests/fixtures/mini_v03` в режиме `train`; вместо пустого ответа она сдаёт один верный вывод. Длинные части заменены на `…`.

После `reset` политика получает задачу, manifest, policy, число записей по таблицам, сигнатуры действий и остаток шагов:

```json
{"info": {}, "observation": {"actions": {"calculator": "calculator(expression)", "get_document": "get_document(doc_id)", "get_record": "get_record(table, id)", "list_records": "list_records(table, cursor=null, limit=50)", "submit": "submit(findings)"}, "manifest": {"base_currency": "KZT", "counts": {"fx": 88, "invoices": 14, "ledger": 48, "payments": 7}, "documents": 21, "episode_id": "KARPSR", "period_end": "2026-09-30", "period_start": "2026-09-01", …}, "policy": {…}, "steps_left": 100, "tables": {…}, "task": "You audit one company for one reporting period. …"}}
```

Политика отвечает действием:

```json
{"action": "list_records", "table": "ledger", "limit": 2}
```

Ответ `step`:

```json
{"info": {}, "observation": {"result": {"next_cursor": "c1.KARPSR.ledger.2", "records": [{"account": "1030", "credit_minor": 0, "currency": "KZT", "debit_minor": 31550000, …, "row_id": "0N1PQJ", "timestamp": "2026-09-18T12:00:00+05:00"}, {…}], "table": "ledger"}, "steps_left": 99}, "reward": 0.0, "terminated": false, "truncated": false}
```

Политика сдаёт `findings.json` (кейс §05):

```json
{"action": "submit", "findings": {"episode_id": "KARPSR", "findings": [{"type": "duplicate_invoice", "affected_ids": ["invoice/1TNM24", "invoice/F7SF66"], "claimed_amount_minor": 325000, "currency": "EUR", "evidence_ids": ["invoice/1TNM24", "invoice/F7SF66", "ledger/CKBBZ4", "ledger/KBYZZ8"], "explanation": "the same EUR bill keyed twice with the vendor reference retyped"}]}}
```

Ответ с наградой:

```json
{"info": {"grading": {"f1": 0.2222222222222222, "fn": 7, "fp": 0, "precision": 1.0, "recall": 0.125, "tp": 1}}, "observation": {"result": {"status": "submitted"}, "steps_left": 98}, "reward": -0.5575555555555556, "terminated": true, "truncated": false}
```

Один верный вывод из восьми ошибок даёт F1 = 2/9. Награда равна 2·F1 − 1 − 0.002·1, где 1 это шаг до submit, то есть −0.5576. В режиме `eval` политика на тот же submit увидит `"reward": 0.0` и пустой `info`.

**Действия** (кейс §04). Машиночитаемые схемы JSON Schema 2020-12 лежат в `schemas/`: `action.schema.json` для строки действия и `findings.schema.json` для `findings.json`. Два правила среда проверяет сверх схемы: `episode_id` должен совпадать с эпизодом, а целые числа (`claimed_amount_minor`, `limit`) пишутся без дробной части и показателя. По JSON Schema 2020-12 `325000.0` это целое, а среда даёт на него ошибку.

| действие | аргументы |
|---|---|
| `list_records` | `table` из `ledger`, `invoices`, `payments`, `fx`; `cursor` (`null` для начала); `limit` от 1 до 50 |
| `get_record` | `table`, `id`; ID можно писать с префиксом таблицы или без |
| `get_document` | `doc_id`: значение поля `doc_id` записи счёта или платежа (например `WRJD8T`). По `invoice_id` или `payment_id` запись читает `get_record` |
| `calculator` | `expression`: десятичные числа, `+ - * /` и скобки, до 200 символов, вложенность скобок до 20, показатель степени в числе до ±40. Результат округляется до 18 знаков после запятой (`1/3` даёт `0.333333333333333333`, `1e-20` даёт `0`) и по модулю должен быть меньше `1e34`: `1e35` или `1e40` в числе допустимы, но как результат дают `result_too_large`. Ошибки: `expression_too_long`, `bad_expression`, `division_by_zero`, `result_too_large` |
| `submit` | `findings`: объект `findings.json` |

**Лимиты:**
- 100 шагов на эпизод, submit тоже шаг;
- 60 секунд на эпизод;
- не больше 30 выводов, не больше 10 `evidence_ids` на вывод, `explanation` не длиннее 1000 символов.

**Неверное сообщение.** Сообщение политики разбирает `auditenv.server.decode_message`. Неверным считается сообщение:
- больше 1 МБ, то есть 1 000 000 байт;
- не JSON, с NaN, Infinity или `1e999`;
- с целым длиннее 30 цифр или вложенностью глубже 32;
- с битым UTF-8 или одиночным суррогатом.

Такое сообщение даёт короткую ошибку `bad_action` в наблюдении. Шаг расходуется, эпизод продолжается. Другие ошибки (`unknown_action`, `unknown_table`, `bad_arguments`, `bad_limit`, `bad_cursor`, `not_found`, ошибки калькулятора) тоже расходуют шаг. Ответ `not_found` одинаков для любого отсутствующего ID.

**Неверная структура submit.** Чужой `episode_id`, неверные поля или превышение лимитов завершают эпизод с наградой −1 (кейс §05). Код ошибки уходит в журнал оценщика, а в режиме `train` ещё и в `info.grading.error`.

**Вывод.** Все шесть полей обязательны: `type`, `affected_ids`, `claimed_amount_minor`, `currency`, `evidence_ids`, `explanation`. Без любого из них награда −1. Вывод совпадает с ошибкой эталона, когда совпадают тип и набор `affected_ids`, сумма лежит в допуске (1 минимальная единица), валюта совпадает, а `evidence_ids` содержат весь набор доказательств (SPEC §5.2). По типам это выглядит так:

| тип | `affected_ids` | сумма | валюта | доказательства |
|---|---|---|---|---|
| `duplicate_invoice` | оба счёта | `total_minor` счёта | валюта счёта | оба счёта и строка 3310 каждого |
| `payment_allocation` | платёж и счёт | сумма платежа минус его погашения этого счёта, она же сумма строки 3510 | валюта платежа | платёж, счёт, строки погашения и строка 3510 |
| `fx_rate` | несущая строка книги | модуль разницы базы строки и пересчёта `doc_amount_minor` по курсу даты по правилу | базовая | документ, несущая строка, `fx/<fx_id>` курса по правилу |
| `period_cutoff` | несущая строка книги | база несущей строки | базовая | документ и несущая строка |

Несущая строка: 1210 у счёта клиенту, 3310 у счёта поставщика, 1030 у платежа.

## 5. Config

Значения по умолчанию в `auditenv/config.py`. Неизвестный ключ, `bool` вместо числа или значение вне границ дают `ValueError` с коротким текстом. Порядок ключей и элементов списков не влияет на эпизод.

| ключ | по умолчанию | допустимо |
|---|---|---|
| `period` | `null` | `null` (месяц выбирает seed) или `"2026-MM"`, только месяцы 2026 года |
| `base_currency` | `"KZT"` | только `"KZT"` |
| `documents` | `[150, 200]` | `[min, max]`, 20 ≤ min ≤ max ≤ 200 |
| `foreign_pct` | `[0, 60]` | `[min, max]` от 0 до 100; `fx_rate` и C1 требуют max больше 0 |
| `company_offsets` | `["+03:00", "+04:00", "+05:00", "+06:00"]` | непустой список разных сдвигов `±HH:MM`, часы от 00 до 14, минуты 00, 30 или 45 |
| `errors` | по 1 каждого из `duplicate_invoice`, `payment_allocation`, `fx_rate`, `period_cutoff` | число ≥ 0 на тип; типы, не названные в объекте, равны 0 |
| `scenarios` | `{"C1": 0, "C2": 0}` | число ≥ 0 на сценарий |
| `twins` | `[3, 8]` | `[min, max]` от 0 до 20: желаемое число двойников, описанных в `provenance.json`. На маленьких книгах генератор ставит столько, сколько помещается, и может дать меньше min без ошибки; фактическое число видно в `provenance.json`. При max 0 генератор не сажает двойников позднего внесения, полуночи по UTC и возврата |

- **Всего не больше 12 ошибок**, C1 считается за 3, C2 за 1.
- **Чистая компания:** все `errors` и `scenarios` равны 0.
- **C1 возможен только в 2026-03, 2026-06, 2026-09 и 2026-11.** Ему нужны два рабочих дня с разными курсами на границе месяцев. При `period: null` генератор выбирает подходящий месяц сам, другой месяц даёт `ValueError` со списком подходящих.
- **C1 не работает со сдвигом только `+00:00`.**
- **Если книга не вмещает запрошенные ошибки**, `ValueError` называет нужное число документов. Это зависит только от config, а не от seed.

## 6. Открытые эпизоды train и dev

- **Train:** `episodes/train/<seed>/`, seed 1000…1009, набор config в `configs/train_v2.json`.
- **Dev:** `episodes/dev/<seed>/`, seed 2000…2009, набор config в `configs/dev_v2.json`.
- **Наборы.** Файлы в `configs/*_v2.json` это наборы `{version, split, comment, episodes: [{seed, config}]}`. Их принимает `--config`, например `python3 -m auditenv.run --policy oracle --config configs/dev_v2.json --seed 2000` повторяет эпизод dev 2000 (`DQ1576`).
- **Индекс:** `index.json` в каждой папке набора, с seed, `episode_id`, sha256 config, признаком чистоты и числом ошибок.
- **Состав:** из 20 эпизодов 5 чистых и 5 со сценариями C1 или C2.
- **Траектории** четырёх политик на train в режиме `train` лежат в `episodes/train/trajectories.jsonl`. Ключи по кейсу §09: `episode_id, step, action, observation, reward, terminated, truncated`. Закрытого эталона в наблюдениях нет.
- **Порядок в файле траекторий.** Эпизоды идут по seed, 1000…1009. Внутри эпизода подряд идут `random`, `reconcile`, `flag_all` и `oracle`, каждая политика с шага 0. Имени политики в строке нет: кейс §09 задаёт ровно семь ключей, поэтому политику даёт порядок, а новый прогон начинается со строки `step` 0.

Пересборка даёт те же файлы побайтно:

```bash
make episodes
```

Траектории train пересобираются так. Скрипт дописывает в конец файла, поэтому команда сначала удаляет старый `runs/trajectories.jsonl`, например после `make demo`. Результат совпадает с закоммиченным `episodes/train/trajectories.jsonl` побайтно:

```bash
rm -f runs/trajectories.jsonl && python3 scripts/metrics.py episodes/train --mode train --out runs/train.md --trajectories runs/trajectories.jsonl
```

## 7. Результаты политик

Из [docs/METRICS.md](docs/METRICS.md), 20 эпизодов train и dev, режим eval:

| политика | F1 | награда, смесь | награда, с ошибками | награда, чистые |
|---|---:|---:|---:|---:|
| oracle | 1.000 | 1.000 | 1.000 | 1.000 |
| reconcile | 0.297 | −0.520 | −0.360 | −1.000 |
| spray (диагностика) | 0.064 | −0.913 | −0.885 | −1.000 |
| flag_all | 0.000 | −1.000 | −1.000 | −1.000 |
| random | 0.000 | −1.000 | −1.000 | −1.000 |

- **«Пометить всё»** не выше молчания ни на одном эпизоде и даёт −1 на каждой чистой компании.
- **Сверка** пропускает 6 ошибок, все это ошибки периода, у которых дата документа по UTC ещё в периоде: 3 в сценарии C1 и 3 счёта, выставленные в первые 15 минут после местной полуночи (`docs/METRICS.md`, раздел «Пропуски сверки по сценариям»). Из-за лимита в 30 выводов она не пропускает ничего.
- **Spray** это не политика жюри, а диагностика угадывания у границы периода (`policies/spray.py`).

Повторить отчёт побайтно, кроме столбца времени:

```bash
make metrics
```

## 8. Парные проверки

```bash
make paired
```

Скрипт `scripts/paired.py` проводит 20 проверок на `episodes/dev` (кейс §06 и §10, «Воспроизводимость и обобщение»). Каждая проверка делает обе части ниже и проходит, только если прошли обе:
- **Повтор.** Тот же seed и те же действия дают побайтно ту же траекторию и награду.
- **Перестановка и смена ID** через `auditenv/transform.py` `permute_and_rename(episode_dir, out_dir, salt)`. Строки переставлены, все ID новые, truth переходит в тот же truth с новыми ID и теми же суммами. Oracle получает 1.0 на оригинале и на копии, сверка получает ту же награду.

## 9. Известные ограничения

- **Песочница** (отдельный пользователь `auditpolicy`, закрытые `private/`) действует только в Docker-образе, где раннер работает от root. На машине разработчика политика запускается от того же пользователя, поэтому закрытый прогон только в Docker (раздел 4).
- **Валюты.** База только KZT. Документы в USD, EUR, RUB и CNY, курсы синтетические: случайное блуждание по рабочим дням.
- **Учёт.** Нет переоценки валютных остатков на конец периода, НДС, банковских комиссий и кросс-валютных оплат. Все учётные допущения собраны в [DESIGN.md](DESIGN.md), раздел 5.
- **Возвраты** клиенту только в базовой валюте.
- **Период** только месяцы 2026 года.
- **C1** возможен только в марте, июне, сентябре и ноябре 2026 года (раздел 5).
- **Доля документов** в выходные и праздники 2–6% почти всегда выдерживается для книг обычного размера: от 150 до 200 документов дают 2.0–6.0%, чуть выше потолка 1 книга из 60, 6.02%. В книге на 20–30 документов каждый документ весит 3–5%, и два-три обязательных документа в выходной дают до 13.6%: вне диапазона 32 книги из 60 (`python3 scripts/data_checks.py`).
- **Формула награды кейса** (R-08). Спрей по кандидатам с верной формой проигрывает молчанию на смеси с 25% чистых компаний, только когда на одну ошибку приходится больше 4 похожих корректных записей. Цифры в [DESIGN.md](DESIGN.md).

## 10. Документы

- [LICENSE](LICENSE): лицензия MIT.
- [schemas/](schemas): JSON Schema действия и `findings.json`.
- [SPEC.md](SPEC.md): схема файлов, контракт среды, оценка, инварианты.
- [DESIGN.md](DESIGN.md): как устроены генератор, двойники и оценщик, разбор провалов политик.
- [docs/CASE.md](docs/CASE.md): текст кейса организаторов.
- [docs/METRICS.md](docs/METRICS.md): метрики политик на train и dev.
- [docs/PM_DECISIONS.md](docs/PM_DECISIONS.md): решения по открытым вопросам.
