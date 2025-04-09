# Ukrainian (ukUA) Translation Ruleset for SMITE 2

This comprehensive document provides translation rules and a game-specific glossary for Ukrainian (ukUA) localization of SMITE 2. These guidelines are designed to ensure consistency, accuracy, and cultural appropriateness in the Ukrainian version of the game.

## Table of Contents

1. [Special Translation Rules](#special-translation-rules)
   - [Grammatical Rules](#1-grammatical-rules)
   - [Formatting Rules](#2-formatting-rules)
   - [Terminology Rules](#3-terminology-rules)
   - [Cultural Adaptation](#4-cultural-adaptation)
   - [Technical Constraints](#5-technical-constraints)
   - [Exception Handling](#6-exception-handling)
   - [Implementation Guidelines](#7-implementation-guidelines)

2. [Game-Specific Glossary](#game-specific-glossary)
   - [Core Game Concepts](#core-game-concepts)
   - [Character Classes & Roles](#character-classes--roles)
   - [Combat Terminology](#combat-terminology)
   - [Crowd Control Effects](#crowd-control-effects)
   - [Map Elements](#map-elements)
   - [Game Modes](#game-modes)
   - [UI Elements](#ui-elements)
   - [Stats and Attributes](#stats-and-attributes)
   - [Ability Mechanics](#ability-mechanics)
   - [Item Categories](#item-categories)
   - [Common Abbreviations](#common-abbreviations)
   - [Specialized Gaming Terms](#specialized-gaming-terms)
   - [Standardized UI Abbreviations](#standardized-ui-abbreviations-for-space-constraints)

3. [Validation Insights](#validation-insights)
   - [Term Consistency Analysis](#term-consistency-analysis)
   - [Address Form Usage](#address-form-usage)
   - [Grammatical Pattern Usage](#grammatical-pattern-usage)
   - [Formatting Pattern Usage](#formatting-pattern-usage)
   - [Key Findings and Recommendations](#key-findings-and-recommendations)

---

## Special Translation Rules

### 1. Grammatical Rules

#### 1.1 Gender Assignment

- **Consistency**: Assign consistent grammatical gender to all game terms
  - Masculine: урон (damage), щит (shield), бафф (buff), меч (sword)
  - Feminine: здібність (ability), сила (power), швидкість (speed), вежа (tower)
  - Neuter: здоров'я (health), золото (gold), мана (mana), посилення (enhancement)

- **Character-specific gender**:
  - Use appropriate gender forms for character classes based on the character's gender
  - Example: "Warrior" → "Воїн" (m) or "Воїтелька" (f)
  - Example: "Guardian" → "Охоронець" (m) or "Охоронниця" (f)

- **New terminology**:
  - For new game terms, assign gender based on:
    1. Established Ukrainian gaming terminology
    2. The gender of the closest Ukrainian equivalent
    3. Phonological patterns for borrowed terms (consonant-ending terms typically masculine)

#### 1.2 Case Usage

- **Nominative case**: Use for ability names, item names, and UI labels
  - "Блискавка Зевса" (Zeus's Lightning) - name of ability
  - "Меч долі" (Sword of Destiny) - name of item
  - *Validation shows this is the most common case (4575 occurrences)*

- **Accusative case**: Use for:
  - Direct objects: "Завдає шкоду" (Deals damage)
  - Duration: "на 5 секунд" (for 5 seconds)
  - *Validation shows this is the second most common case (2683 occurrences)*

- **Genitive case**: Use for:
  - Possession relationships: "Сила бога" (Power of the god)
  - Negation: "Немає мани" (No mana)
  - After numbers 2-4: "2 пункти здоров'я" (2 health points)
  - After numbers 5+: "5 пунктів здоров'я" (5 health points)
  - *Validation shows this is the third most common case (1641 occurrences)*

- **Instrumental case**: Use for:
  - Means/instruments: "атакує мечем" (attacks with a sword)
  - With certain prepositions: "під вежею" (under the tower)

- **Locative case**: Use for:
  - Locations: "у джунглях" (in the jungle)
  - With certain prepositions: "при використанні" (upon use)

- **Dative case**: Use for:
  - Indirect objects: "дає союзникам" (gives to allies)
  - With certain prepositions: "по ворогам" (towards enemies)

- **Vocative case**: Use for direct address in character dialogue
  - "Герою, допоможи!" (Hero, help!)

#### 1.3 Verb Aspect and Tense

- **Aspect choice**:
  - Use perfective aspect for one-time, completed effects
    - "завдасть шкоди" (will deal damage - single instance)
    - "оглушить ворогів" (will stun enemies - completed action)
  
  - Use imperfective aspect for ongoing or repeated effects
    - "лікує щосекунди" (heals every second - repeated action)
    - "збільшує швидкість руху" (increases movement speed - ongoing effect)
  
  - *Validation shows a slight preference for imperfective aspect (1415 occurrences) over perfective aspect (1281 occurrences)*

- **Tense usage**:
  - Present tense for ability descriptions and passive effects
  - Future perfective for triggered effects
  - Imperative mood for player instructions

#### 1.4 Adjective Agreement

- Ensure adjectives agree with nouns in gender, number, and case
  - Masculine: "фізичний урон" (physical damage)
  - Feminine: "магічна сила" (magical power)
  - Neuter: "високе здоров'я" (high health)

- For compound terms, maintain agreement throughout
  - "збільшена швидкість атаки" (increased attack speed)
  - "зменшений час відновлення" (reduced cooldown time)

#### 1.5 Number Agreement

- **Numerical values**: Use appropriate forms based on the number
  - 1: singular nominative (1 пункт)
  - 2-4: plural genitive singular (2 пункти)
  - 5+: plural genitive plural (5 пунктів)

- **Decimal values**: Treat as 5+ for grammatical purposes
  - "2,5 секунди" (2.5 seconds - genitive singular)

- **Percentage values**: No space between number and % symbol
  - "25% шансу" (25% chance)

#### 1.6 Address Form

- **Use informal "ти" form** consistently for:
  - Player instructions
  - Tooltips
  - System messages
  - Tutorial content
  - *Validation confirms preference for informal "ти" form (52.5%) over formal "Ви" form (47.5%)*

- **Do not use formal "Ви" form** except for:
  - Formal announcements
  - Certain character dialogue where appropriate for the character's personality

### 2. Formatting Rules

#### 2.1 Punctuation

- Use Ukrainian punctuation conventions:
  - **Decimal separator: comma (,) instead of period (.)**
    - English: 2.5 seconds → Ukrainian: 2,5 секунди
    - *Validation confirms usage of decimal comma (199 occurrences)*
  
  - **Thousands separator: space instead of comma**
    - English: 1,000 gold → Ukrainian: 1 000 золота
    - *Validation shows this is not consistently applied in the dataset*
  
  - **Quotation marks: «text» (guillemets) instead of "text"**
    - English: "Victory" → Ukrainian: «Перемога»
    - *Validation shows limited usage of Ukrainian quotation marks (4 occurrences)*
  
  - **Dialogue dash: em dash (—) for dialogue**
    - English: "Run!" he said. → Ukrainian: — Біжи! — сказав він.

- Maintain English punctuation only for:
  - Technical notations
  - Version numbers
  - Code elements

#### 2.2 Number Formatting

- **Units**: Space between number and unit
  - English: 5s → Ukrainian: 5 с
  - Exception: No space with percentage (50%)

- **Ranges**: En dash with spaces
  - English: 50-100 damage → Ukrainian: 50 – 100 шкоди

- **Mathematical operations**: Spaces around operators
  - English: 50+10 → Ukrainian: 50 + 10

- **Negative values**: No space after minus sign
  - English: -50 health → Ukrainian: -50 здоров'я

#### 2.3 Capitalization

- **Less capitalization than English**:
  - Ability names: Only first word capitalized unless proper noun
    - English: "Eagle's Rally" → Ukrainian: "Клич орла"
  
  - Item names: Only first word capitalized unless proper noun
    - English: "Warrior's Axe" → Ukrainian: "Сокира воїна"
  
  - Game modes: Not capitalized
    - English: "Conquest" → Ukrainian: "завоювання"
  
  - Character classes: Not capitalized
    - English: "Warrior" → Ukrainian: "воїн"
  
  - *Validation confirms capitalization of first word only (7703 occurrences)*

- **Always capitalize**:
  - God names and proper nouns
  - First word of sentences
  - Acronyms (AOE, CC)

#### 2.4 Hyphenation and Compound Words

- Use hyphen (-) for compound adjectives
  - "магічно-фізичний" (magical-physical)

- Compound nouns typically written as separate words or single word
  - "час відновлення" (cooldown - literally "recovery time")
  - "життєкрадіння" (lifesteal - single compound word)

#### 2.5 Abbreviations

- Use Ukrainian abbreviations when space is limited:
  - "пнк." for "пункти" (points)
  - "сек." for "секунди" (seconds)
  - "шкд." for "шкода" (damage)
  - "здр." for "здоров'я" (health)

- Retain English gaming abbreviations that are widely recognized:
  - "AOE" for "Area of Effect"
  - "CC" for "Crowd Control"
  - "CD" for "Cooldown"

### 3. Terminology Rules

#### 3.1 Translation Approaches

- **Direct translation** (preferred for most terms):
  - Health → Здоров'я (94.8% consistency in dataset)
  - Damage → Шкода (preferred over "Урон" which has 0% consistency in dataset)
  - Movement Speed → Швидкість руху (100% consistency in dataset)

- **Borrowing** (for established gaming terms):
  - Gank → Ганк
  - Buff → Бафф (though "Посилення" has 100% consistency in dataset)
  - Tank → Танк

- **Calque** (for compound terms):
  - Cooldown → Час відновлення (literally "recovery time")
  - Lifesteal → Викрадення життя (literally "theft of life")
  - Crowd control → Контроль натовпу (literally "crowd control")

- **Functional equivalent** (when direct translation doesn't convey meaning):
  - Skillshot → Прицільна здібність (literally "aimed ability")
  - Execute → Стратити (literally "execute" in sense of punishment)
  - Cleave → Розсікаючий удар (literally "cleaving strike")

#### 3.2 Consistency Guidelines

- Refer to the glossary for standardized translations
- Maintain consistency within related term families
  - All damage types (physical, magical, true)
  - All crowd control effects
  - All stat categories

- When multiple translations exist, prefer:
  1. The most widely used term in Ukrainian gaming community
  2. The term with clearest meaning to new players
  3. The term that best fits UI space constraints

#### 3.3 Terminology Standardization

- **Standardize key gameplay terms based on validation findings:**
  - God → Бог (100% consistency)
  - Ability → Здібність (52.2% consistency) for singular, Здібності (47.8% consistency) for plural
  - Passive → Пасивна (100% consistency)
  - Ultimate → Ультимативна здібність (100% consistency for "ультимативн")

- **Standardize damage terminology:**
  - Damage → Шкода (18.5% consistency) preferred over Урон (0% consistency)
  - Physical Damage → Фізична шкода (Physical has 100% consistency)
  - Magical Damage → Магічна шкода (Magical has 100% consistency)
  - True Damage → Чиста шкода

- **Standardize crowd control terminology:**
  - Stun → Оглушення
  - Root → Знерухомлення
  - Silence → Німота (100% consistency)
  - Slow → Уповільнення (100% consistency)

### 4. Cultural Adaptation

#### 4.1 Mythological References

- Use established Ukrainian equivalents for Greek/Roman deities:
  - Zeus → Зевс
  - Athena → Афіна
  - Mercury → Меркурій
  - Hercules → Геркулес

- Use established Ukrainian equivalents for Norse deities:
  - Thor → Тор
  - Odin → Одін
  - Loki → Локі

- Maintain original names for deities from other pantheons unless established Ukrainian equivalents exist

#### 4.2 Cultural References and Humor

- Adapt puns and wordplay to maintain humor rather than literal translation
- Replace culture-specific references with Ukrainian equivalents when necessary
- Maintain character personality in dialogue adaptation
- Consider Ukrainian gaming culture norms and preferences

#### 4.3 Sensitivity Considerations

- Avoid politically sensitive content
- Respect cultural and religious sensitivities
- Maintain PEGI rating requirements in translation
- Ensure translations are appropriate for the Ukrainian market

### 5. Technical Constraints

#### 5.1 Text Expansion

- Ukrainian text typically 15-20% longer than English
- Strategies for managing text expansion:
  - Use abbreviations in UI elements
  - Use shorter synonyms when available
  - Omit implied subjects (Ukrainian is a pro-drop language)
  - Simplify complex grammatical structures

#### 5.2 UI Space Constraints

- For extremely limited space:
  - Use established abbreviations
  - Use shorter synonyms
  - Consider borrowing English terms if they're shorter and widely understood

- Priority order for text reduction:
  1. Remove articles and pronouns
  2. Use abbreviations
  3. Use shorter synonyms
  4. Restructure sentence

#### 5.3 Variable Text

- Ensure proper grammatical integration of variables
- Account for number agreement with variables
- Test with maximum length values to ensure text fits

### 6. Exception Handling

#### 6.1 Untranslatable Elements

- Do not translate:
  - Proper names without established Ukrainian equivalents
  - Technical identifiers
  - Code elements
  - Trademark and copyright symbols

#### 6.2 Ambiguous Terms

- For terms with multiple possible translations:
  - Consider context to determine appropriate translation
  - Maintain consistency within similar contexts
  - Document decision and rationale in glossary

#### 6.3 New Terminology

- For new game concepts without established translations:
  1. Consider existing Ukrainian gaming terminology
  2. Create translation that clearly conveys meaning
  3. Consider space constraints and ease of pronunciation
  4. Document in glossary for consistency

### 7. Implementation Guidelines

#### 7.1 Quality Assurance

- Verify grammatical correctness:
  - Gender agreement
  - Case agreement
  - Number agreement
  - Verb aspect appropriateness

- Check for natural-sounding Ukrainian:
  - Avoid literal translations that sound awkward
  - Ensure idiomatic expressions are used correctly
  - Verify specialized terminology is used appropriately

- Test in context:
  - Verify text fits UI elements
  - Check for truncation issues
  - Ensure variables are properly integrated

#### 7.2 Feedback Integration

- Establish process for player feedback on translations
- Prioritize fixing terminology inconsistencies
- Document changes to maintain consistency in future updates

#### 7.3 Maintenance

- Update glossary with new terms as they are added
- Document translation decisions for future reference
- Maintain version control of translation assets
- Ensure consistency across game updates

---

## Game-Specific Glossary

### Core Game Concepts

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| SMITE 2 | SMITE 2 | Game title remains untranslated |
| God | Бог (m) | 100% consistency in dataset; use Богиня (f) for female characters |
| Ability | Здібність (f) | 52.2% consistency in dataset; Plural: Здібності (47.8%) |
| Passive | Пасивна (f) | 100% consistency in dataset; Often used as "Пасивна здібність" (passive ability) |
| Ultimate | Ультимативна здібність (f) | 100% consistency for "ультимативн"; "Ульта" (f) not found in dataset |
| Level | Рівень (m) | Used for character levels and ability levels |
| Experience | Досвід (m) | Abbreviated as "Дсв." in UI constraints |
| Gold | Золото (n) | Currency unit |
| Item | Предмет (m) | 100% consistency in dataset; Plural: Предмети |
| Match | Матч (m) | Plural: Матчі |
| Victory | Перемога (f) | |
| Defeat | Поразка (f) | |
| Team | Команда (f) | |
| Ally | Союзник (m) / Союзниця (f) | Gender-specific forms |
| Enemy | Ворог (m) / Ворогиня (f) | Gender-specific forms; Plural: Вороги |
| Minion | Міньйон (m) | Borrowed term |
| Jungle | Джунглі (f, plural) | Always used in plural form |
| Lane | Лінія (f) | Literally "line" |
| Map | Карта (f) | |
| Fountain | Фонтан (m) | |
| Base | База (f) | |
| Respawn | Відродження (n) | Verb form: Відродитися |
| Spawn | Поява (f) | Verb form: З'явитися |
| Queue | Черга (f) | |
| Matchmaking | Підбір гравців (m) | Literally "selection of players" |
| Leaderboard | Таблиця лідерів (f) | |
| Scoreboard | Таблиця результатів (f) | |
| Tutorial | Навчання (n) | Literally "training" |
| Practice | Практика (f) | |
| Surrender | Здача (f) | Verb form: Здатися |
| Pause | Пауза (f) | Verb form: Поставити на паузу |
| Disconnect | Відключення (n) | Verb form: Відключитися |
| Reconnect | Перепідключення (n) | Verb form: Перепідключитися |

### Character Classes & Roles

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| Assassin | Вбивця (m/f) | Same form for both genders |
| Guardian | Охоронець (m) / Охоронниця (f) | Gender-specific forms |
| Hunter | Мисливець (m) / Мисливиця (f) | Gender-specific forms |
| Mage | Маг (m) / Магиня (f) | Gender-specific forms |
| Warrior | Воїн (m) / Воїтелька (f) | Gender-specific forms |
| Support | Підтримка (f) | Role name, not class |
| Carry | Керрі | Borrowed term, indeclinable |
| Solo | Соло | Borrowed term, indeclinable |
| Jungler | Джанглер (m) | Ukrainianized from English |
| Mid | Мід | Borrowed term, sometimes "Середня лінія" (middle lane) |
| Tank | Танк (m) | Borrowed term |
| Damage Dealer | Дамагер (m) | Borrowed and Ukrainianized from "damage" |
| Healer | Цілитель (m) / Цілителька (f) | Gender-specific forms |
| Initiator | Ініціатор (m) / Ініціаторка (f) | Gender-specific forms |
| Pusher | Пушер (m) | Borrowed and Ukrainianized from "push" |
| Ganker | Ганкер (m) | Borrowed and Ukrainianized from "gank" |
| Roamer | Бродяга (m/f) | |
| Flex | Флекс | Borrowed term |
| Bruiser | Брузер (m) | Borrowed and Ukrainianized |
| Caster | Кастер (m) | Borrowed and Ukrainianized |
| Auto-Attack | Авто-атака (f) | |

### Combat Terminology

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| Damage | Шкода (f) | 18.5% consistency in dataset; preferred over "Урон" (0% consistency) |
| Physical | Фізичний (adj) | 100% consistency in dataset; Agrees with noun gender |
| Magical | Магічний (adj) | 100% consistency in dataset; Agrees with noun gender |
| Attack | Атака (f) | Verb form: Атакувати |
| Basic Attack | Базова атака (f) | |
| Health | Здоров'я (n) | 94.8% consistency in dataset; Sometimes "Пункти здоров'я" (health points) |
| Mana | Мана (f) | 12.4% consistency in dataset; "Мани" (genitive form) has 87.6% consistency |
| Protection | Захист (m) | 100% consistency in dataset |
| Penetration | Пробивання (n) | 100% consistency in dataset |
| Critical | Критичний (adj) | 100% consistency in dataset; Agrees with noun gender |
| Critical Strike | Критичний удар (m) | |
| Lifesteal | Викрадення життя (n) | Literally "theft of life" |
| Healing | Зцілення (n) | Verb form: Зцілювати |
| Shield | Щит (m) | |
| Movement Speed | Швидкість руху (f) | 100% consistency in dataset |
| Attack Speed | Швидкість атаки (f) | 100% consistency in dataset |
| Cooldown | Час відновлення (m) | 0% consistency for full term, but "відновлення" has 100% consistency |
| Range | Дальність (f) | |
| Area of Effect | Область дії (f) | Often abbreviated as AOE |
| Duration | Тривалість (f) | |
| Kill | Вбивство (n) | Verb form: Вбити |
| Death | Смерть (f) | |
| Assist | Допомога (f) | |
| Buff | Посилення (n) | 100% consistency in dataset; "Бафф" (m) has 0% consistency |
| Debuff | Ослаблення (n) | 100% consistency in dataset; "Дебафф" (m) has 0% consistency |
| True Damage | Чиста шкода (f) | Literally "pure damage" |
| Damage over Time | Шкода з часом (f) | Abbreviated as ШЗЧ |
| Heal over Time | Зцілення з часом (n) | |
| Burst Damage | Миттєва шкода (f) | Literally "instant damage" |
| Cleave | Розсікаючий удар (m) | Literally "cleaving strike" |
| Splash Damage | Сплеш шкода (f) | Partially borrowed term |
| Projectile | Снаряд (m) | |
| Homing | Самонавідний (adj) | |
| Skillshot | Прицільна здібність (f) | Literally "aimed ability" |
| Hitbox | Хітбокс (m) | Borrowed term |
| Collision | Зіткнення (n) | |
| Immunity | Імунітет (m) | |
| Invulnerability | Невразливість (f) | |
| Resistance | Стійкість (f) | |
| Weakness | Слабкість (f) | |
| Vulnerability | Вразливість (f) | |

### Crowd Control Effects

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| Crowd Control | Контроль натовпу (m) | Often abbreviated as КН or CC |
| Stun | Оглушення (n) | Verb form: Оглушити |
| Slow | Уповільнення (n) | 100% consistency in dataset; Verb form: Уповільнити |
| Root | Знерухомлення (n) | Verb form: Знерухомити |
| Silence | Німота (f) | 100% consistency in dataset; Literally "muteness" |
| Disarm | Роззброєння (n) | Verb form: Роззброїти |
| Cripple | Калічення (n) | Verb form: Скалічити |
| Knockup | Підкидання (n) | Verb form: Підкинути |
| Knockback | Відкидання (n) | Verb form: Відкинути |
| Pull | Притягування (n) | Verb form: Притягнути |
| Mesmerize | Заворожування (n) | Verb form: Заворожити |
| Taunt | Провокація (f) | Verb form: Спровокувати |
| Fear | Страх (m) | Verb form: Налякати |
| Intoxicate | Сп'яніння (n) | Verb form: Сп'янити |
| Madness | Божевілля (n) | |
| Blind | Осліплення (n) | Verb form: Осліпити |
| Banish | Вигнання (n) | Verb form: Вигнати |
| Polymorph | Поліморф (m) | Borrowed term |
| Charm | Чарування (n) | Verb form: Зачарувати |
| Disorient | Дезорієнтація (f) | Verb form: Дезорієнтувати |
| Stagger | Хитання (n) | Verb form: Похитнути |
| Tremor | Тремтіння (n) | Verb form: Тремтіти |
| Vortex | Вихор (m) | |
| Grab | Захват (m) | Verb form: Захопити |
| Imprison | Ув'язнення (n) | Verb form: Ув'язнити |

### Map Elements

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| Tower | Вежа (f) | |
| Phoenix | Фенікс (m) | |
| Titan | Титан (m) | |
| Jungle Camp | Табір у джунглях (m) | |
| Buff Camp | Табір посилень (m) | Updated based on validation showing "посилення" is preferred over "бафф" |
| Ward | Вард (m) | Borrowed term |
| Fire Giant | Вогняний Гігант (m) | |
| Gold Fury | Золота Лють (f) | |
| Pyromancer | Піромант (m) | |
| Harpy | Гарпія (f) | |
| Scorpion | Скорпіон (m) | |
| Obelisk | Обеліск (m) | |
| Portal | Портал (m) | |
| Shrine | Святиня (f) | |
| Altar | Вівтар (m) | |
| Fountain | Фонтан (m) | |
| Spawn Point | Точка появи (f) | |
| Fog of War | Туман війни (m) | |
| Line of Sight | Лінія огляду (f) | |
| Wall | Стіна (f) | |
| Bush | Кущ (m) | |
| Grass | Трава (f) | |
| Path | Шлях (m) | |
| Bridge | Міст (m) | |
| Gate | Ворота (n, plural) | Always used in plural form |
| Barrier | Бар'єр (m) | |
| Objective | Ціль (f) | |
| Structure | Споруда (f) | |
| Inhibitor | Інгібітор (m) | Borrowed term |
| Barracks | Казарми (f, plural) | Always used in plural form |

### Game Modes

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| Conquest | Завоювання (n) | Main game mode |
| Arena | Арена (f) | |
| Joust | Поєдинок (m) | |
| Assault | Штурм (m) | |
| Clash | Сутичка (f) | |
| Siege | Облога (f) | |
| MOTD | MOTD | "Match of the Day" - abbreviation kept |
| Custom | Користувацький (adj) | "Користувацький матч" (custom match) |
| Training | Тренування (n) | |
| Co-op | Кооператив (m) | |
| Ranked | Рейтинговий (adj) | "Рейтинговий матч" (ranked match) |
| Casual | Звичайний (adj) | "Звичайний матч" (casual match) |
| Duel | Дуель (f) | |
| Tournament | Турнір (m) | |
| Competitive | Змагальний (adj) | |
| Practice | Практика (f) | |
| Tutorial | Навчання (n) | |
| Spectator | Спостерігач (m) | |
| Replay | Повтор (m) | |
| Qualifying | Кваліфікація (f) | |
| Placement | Розміщення (n) | |
| Season | Сезон (m) | |
| Split | Спліт (m) | Borrowed term |
| Phase | Фаза (f) | |

### UI Elements

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| Menu | Меню (n) | Indeclinable |
| Settings | Налаштування (n, plural) | |
| Profile | Профіль (m) | |
| Friends | Друзі (m, plural) | |
| Clan | Клан (m) | |
| Achievements | Досягнення (n, plural) | |
| Quests | Завдання (n, plural) | |
| Battle Pass | Бойова перепустка (f) | |
| Store | Магазин (m) | |
| Chest | Скриня (f) | |
| Bundle | Набір (m) | |
| Skin | Скін (m) | Borrowed term |
| Voice Pack | Голосовий пакет (m) | |
| Loading Frame | Рамка завантаження (f) | |
| Loading Screen | Екран завантаження (m) | |
| Avatar | Аватар (m) | |
| Emote | Емоція (f) | |
| Announcer Pack | Пакет диктора (m) | |
| Button | Кнопка (f) | |
| Slider | Повзунок (m) | |
| Checkbox | Прапорець (m) | |
| Dropdown | Випадаючий список (m) | |
| Tab | Вкладка (f) | |
| Window | Вікно (n) | |
| Tooltip | Підказка (f) | |
| Notification | Сповіщення (n) | |
| Alert | Оповіщення (n) | |
| Popup | Спливаюче вікно (n) | |
| Icon | Іконка (f) | |
| Badge | Значок (m) | |
| Banner | Банер (m) | |
| Cursor | Курсор (m) | |
| Highlight | Підсвічування (n) | |
| Scroll | Прокрутка (f) | |
| Zoom | Масштабування (n) | |

### Stats and Attributes

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| Power | Сила (f) | |
| Physical Power | Фізична сила (f) | Physical has 100% consistency in dataset |
| Magical Power | Магічна сила (f) | Magical has 100% consistency in dataset |
| Physical Protection | Фізичний захист (m) | Protection has 100% consistency in dataset |
| Magical Protection | Магічний захист (m) | |
| Health Regeneration | Регенерація здоров'я (f) | Health has 94.8% consistency in dataset |
| Mana Regeneration | Регенерація мани (f) | |
| Cooldown Reduction | Зменшення часу відновлення (n) | Often abbreviated as "Зменшення відновлення" |
| Crowd Control Reduction | Зменшення контролю натовпу (n) | |
| Penetration | Пробивання (n) | 100% consistency in dataset |
| Flat Penetration | Фіксоване пробивання (n) | |
| Percentage Penetration | Відсоткове пробивання (n) | |
| Critical Strike Chance | Шанс критичного удару (m) | Critical has 100% consistency in dataset |
| Critical Strike Damage | Шкода від критичного удару (f) | |
| Attack Speed | Швидкість атаки (f) | 100% consistency in dataset |
| Movement Speed | Швидкість руху (f) | 100% consistency in dataset |
| Slow Immunity | Імунітет до уповільнення (m) | Slow has 100% consistency in dataset |
| Knockup Immunity | Імунітет до підкидання (m) | |
| Tenacity | Стійкість (f) | |
| Resilience | Витривалість (f) | |
| Mitigation | Зменшення (n) | |
| Amplification | Посилення (n) | |
| Reduction | Зниження (n) | |
| Conversion | Конверсія (f) | |
| Scaling | Масштабування (n) | |
| Diminishing Returns | Спадна віддача (f) | |
| Threshold | Поріг (m) | |
| Cap | Ліміт (m) | |
| Base Value | Базове значення (n) | |
| Bonus Value | Бонусне значення (n) | |
| Total Value | Загальне значення (n) | |

### Ability Mechanics

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| Cooldown | Час відновлення (m) | "Відновлення" has 100% consistency in dataset |
| Cost | Вартість (f) | |
| Range | Дальність (f) | |
| Radius | Радіус (m) | |
| Duration | Тривалість (f) | |
| Charge | Заряд (m) | |
| Stack | Стак (m) | Borrowed term |
| Channel | Канал (m) | Verb form: Каналізувати |
| Cast | Застосування (n) | Verb form: Застосувати |
| Projectile | Снаряд (m) | |
| Skillshot | Прицільна здібність (f) | |
| Dash | Ривок (m) | |
| Leap | Стрибок (m) | |
| Teleport | Телепорт (m) | Verb form: Телепортуватися |
| Stealth | Невидимість (f) | |
| Execute | Страта (f) | Verb form: Стратити |
| Heal | Зцілення (n) | Verb form: Зцілювати |
| Shield | Щит (m) | Verb form: Захищати |
| Cleanse | Очищення (n) | Verb form: Очистити |
| Reflect | Відбиття (n) | Verb form: Відбити |
| Amplify | Посилення (n) | Verb form: Посилити |
| Reduce | Зниження (n) | Verb form: Знизити |
| Toggle | Перемикання (n) | Verb form: Перемкнути |
| Passive | Пасивна (f) | 100% consistency in dataset |
| Active | Активна (f) | 100% consistency in dataset |
| Aura | Аура (f) | |
| Proc | Спрацьовування (n) | Verb form: Спрацювати |
| Trigger | Тригер (m) | Borrowed term |
| Condition | Умова (f) | |
| Effect | Ефект (m) | |
| Bonus | Бонус (m) | |
| Penalty | Штраф (m) | |
| Modifier | Модифікатор (m) | |
| Multiplier | Множник (m) | |

### Item Categories

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| Starter Item | Початковий предмет (m) | Item has 100% consistency in dataset |
| Consumable | Витратний предмет (m) | Literally "expendable item" |
| Relic | Реліквія (f) | |
| Offensive | Наступальний (adj) | |
| Defensive | Захисний (adj) | |
| Utility | Допоміжний (adj) | |
| Physical | Фізичний (adj) | 100% consistency in dataset |
| Magical | Магічний (adj) | 100% consistency in dataset |
| Aura | Аура (f) | |
| Mask | Маска (f) | |
| Boots | Чоботи (m, plural) | |
| Gloves | Рукавиці (f, plural) | |
| Helm | Шолом (m) | |
| Armor | Броня (f) | |
| Cloak | Плащ (m) | |
| Rod | Жезл (m) | |
| Staff | Посох (m) | |
| Sword | Меч (m) | |
| Dagger | Кинджал (m) | |
| Bow | Лук (m) | |
| Spear | Спис (m) | |
| Mace | Булава (f) | |
| Tier 1 | Рівень 1 (m) | |
| Tier 2 | Рівень 2 (m) | |
| Tier 3 | Рівень 3 (m) | |
| Tier 4 | Рівень 4 (m) | |
| Upgrade | Покращення (n) | Verb form: Покращити |
| Recipe | Рецепт (m) | |
| Component | Компонент (m) | |
| Set | Набір (m) | |
| Limited | Обмежений (adj) | |
| Exclusive | Ексклюзивний (adj) | |

### Common Abbreviations

| English Abbreviation | Ukrainian Translation | Notes |
|----------------------|----------------------|-------|
| HP | ОЗ | "Очки Здоров'я" (Health Points) |
| MP | ОМ | "Очки Мани" (Mana Points) |
| CC | КН | "Контроль Натовпу" (Crowd Control) |
| CDR | ЗЧВ | "Зменшення Часу Відновлення" (Cooldown Reduction) |
| DoT | ШЗЧ | "Шкода З Часом" (Damage over Time) |
| AoE | ОД | "Область Дії" (Area of Effect) |
| AS | ША | "Швидкість Атаки" (Attack Speed) |
| MS | ШР | "Швидкість Руху" (Movement Speed) |
| AD | ФШ | "Фізична Шкода" (Attack Damage) |
| AP | МШ | "Магічна Шкода" (Ability Power) |
| KDA | ВСД | "Вбивства/Смерті/Допомоги" (Kills/Deaths/Assists) |
| AA | БА | "Базова Атака" (Auto Attack) |
| CD | ЧВ | "Час Відновлення" (Cooldown) |
| DPS | ШЗС | "Шкода За Секунду" (Damage Per Second) |
| HPS | ЗЗС | "Зцілення За Секунду" (Healing Per Second) |
| MR | МЗ | "Магічний Захист" (Magic Resistance) |
| PR | ФЗ | "Фізичний Захист" (Physical Resistance) |
| XP | Д | "Досвід" (Experience Points) |
| LVL | РВН | "Рівень" (Level) |
| ULT | УЛТ | "Ультимативна здібність" (Ultimate) |
| GG | ГГ | "Good Game" - abbreviation kept |
| AFK | АФК | "Away From Keyboard" - abbreviation kept |
| BM | БМ | "Bad Manners" - abbreviation kept |
| DC | ДК | "Disconnect" - abbreviation kept |

### Specialized Gaming Terms

| English Term | Ukrainian Translation | Notes |
|--------------|----------------------|-------|
| Gank | Ганк (m) | Borrowed term |
| Farm | Фарм (m) | Borrowed term, verb form: Фармити |
| Last Hit | Останній удар (m) | |
| Poke | Тичок (m) | Verb form: Тикати |
| Burst | Сплеск (m) | Sometimes borrowed as "Бурст" |
| Sustain | Витривалість (f) | |
| Kite | Кайтинг (m) | Borrowed term |
| Peel | Захист (m) | Verb form: Захищати |
| Rotation | Ротація (f) | |
| Invade | Вторгнення (n) | Verb form: Вторгатися |
| Leash | Допомога (f) | Literally "help" |
| Snowball | Сніжний ком (m) | Sometimes used as borrowed verb: Сноуболити |
| Meta | Мета (f) | Borrowed term |
| Nerf | Ослаблення (n) | Verb form: Ослабити |
| Buff (balance change) | Посилення (n) | 100% consistency in dataset; Verb form: Посилити |
| OP | ОП | "Overpowered" - abbreviation kept |
| Smurf | Смурф (m) | Borrowed term |
| Carry (verb) | Нести (v) | Literally "to carry" |
| Feed | Годування (n) | Verb form: Годувати |
| Throw | Злив (m) | In gaming context: losing a winning position |
| Comeback | Повернення (n) | |
| Dive | Дайв (m) | Borrowed term |
| Focus | Фокус (m) | Verb form: Фокусуватися |
| Trade | Обмін (m) | Verb form: Обмінюватися |
| Bait | Приманка (f) | Verb form: Приманювати |
| Juke | Обманний маневр (m) | |
| Cheese | Чіз (m) | Borrowed term for cheap strategy |
| Tilt | Тільт (m) | Borrowed term |
| Grind | Гринд (m) | Borrowed term |
| Macro | Макро (n) | Strategic gameplay |
| Micro | Мікро (n) | Mechanical gameplay |

### Standardized UI Abbreviations (for space constraints)

| Full Ukrainian Term | Abbreviated Form | Notes |
|-------------------|------------------|-------|
| Здоров'я | Здр. | For extremely limited UI space |
| Мана | Мн. | For extremely limited UI space |
| Час відновлення | Ч.Відн. | For extremely limited UI space |
| Швидкість атаки | Шв.Ат. | For extremely limited UI space |
| Швидкість руху | Шв.Рух. | For extremely limited UI space |
| Фізичний захист | Фіз.Зах. | For extremely limited UI space |
| Магічний захист | Маг.Зах. | For extremely limited UI space |
| Фізичне пробивання | Фіз.Проб. | For extremely limited UI space |
| Магічне пробивання | Маг.Проб. | For extremely limited UI space |
| Контроль натовпу | КН | For extremely limited UI space |
| Шкода за секунду | ШЗС | For extremely limited UI space |
| Регенерація здоров'я | Рег.Здр. | For extremely limited UI space |
| Регенерація мани | Рег.Мн. | For extremely limited UI space |
| Зменшення часу відновлення | Зм.Відн. | For extremely limited UI space |
| Критичний удар | Крит.Уд. | For extremely limited UI space |
| Викрадення життя | Викр.Жит. | For extremely limited UI space |
| Область дії | ОД | For extremely limited UI space |
| Досвід | Дсв. | For extremely limited UI space |
| Рівень | Рвн. | For extremely limited UI space |
| Ультимативна здібність | Ульт. | For extremely limited UI space |

---

## Validation Insights

### Term Consistency Analysis

| English Term → Ukrainian Translation | Consistency (%) |
|-----------------------------------|---------------|
| Physical → фізичн | 100.0% |
| Magical → магічн | 100.0% |
| Ultimate → ультимативн | 100.0% |
| Cooldown → відновлення | 100.0% |
| Movement Speed → швидкість руху | 100.0% |
| Attack Speed → швидкість атаки | 100.0% |
| Critical → критичн | 100.0% |
| Penetration → пробивання | 100.0% |
| Protection → захист | 100.0% |
| Slow → уповільнення | 100.0% |
| Silence → німота | 100.0% |
| Passive → пасивн | 100.0% |
| Active → активн | 100.0% |
| God → бог | 100.0% |
| Item → предмет | 100.0% |
| Buff → посилення | 100.0% |
| Debuff → ослаблення | 100.0% |
| Health → здоров'я | 94.8% |
| Mana → мани | 87.6% |
| Damage → шкоди | 81.5% |
| Ability → здібність | 52.2% |
| Ability → здібності | 47.8% |
| Damage → шкода | 18.5% |
| Mana → мана | 12.4% |
| Health → здоров | 5.2% |
| Damage → урон | 0.0% |
| Damage → урону | 0.0% |
| Ultimate → ульта | 0.0% |
| Cooldown → час відновлення | 0.0% |
| God → богиня | 0.0% |
| Buff → бафф | 0.0% |
| Debuff → дебафф | 0.0% |

### Address Form Usage

| Form | Count |
|------|------|
| Informal (ти) | 919 (52.5%) |
| Formal (Ви) | 831 (47.5%) |

### Grammatical Pattern Usage

| Pattern | Count |
|---------|------|
| Perfective Aspect | 1281 |
| Imperfective Aspect | 1415 |
| Nominative Case | 4575 |
| Genitive Case | 1641 |
| Dative Case | 562 |
| Accusative Case | 2683 |
| Instrumental Case | 518 |
| Locative Case | 781 |

### Formatting Pattern Usage

| Pattern | Count |
|---------|------|
| Decimal Comma | 199 |
| Space as Thousands Separator | 0 |
| Ukrainian Quotation Marks | 4 |
| Capitalization of First Word Only | 7703 |

### Key Findings and Recommendations

#### Terms with Low Consistency

The following terms show low consistency in translation and should be standardized:

- Ability → здібності
- Damage → шкода
- Mana → мана
- Health → здоров
- Damage → урон
- Damage → урону
- Ultimate → ульта
- Cooldown → час відновлення
- God → богиня
- Buff → бафф
- Debuff → дебафф

#### Address Form Preference

The dataset shows a preference for Informal (ти) forms. This should be standardized across all player instructions and tooltips.

#### Grammatical Pattern Preferences

Based on the dataset analysis, the following grammatical patterns are predominant:

- Nominative Case: 4575 occurrences
- Accusative Case: 2683 occurrences
- Genitive Case: 1641 occurrences

#### Formatting Recommendations

Based on the dataset analysis, the following formatting conventions should be standardized:

- Use comma as decimal separator (e.g., 2,5 instead of 2.5)
- Use space as thousands separator (e.g., 1 000 instead of 1,000)
- Use Ukrainian quotation marks «» instead of English ""
- Capitalize only the first word in ability and item names, unless they contain proper nouns
