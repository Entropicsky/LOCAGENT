# Russian (ruRU) Translation Ruleset for SMITE 2

This comprehensive ruleset provides guidelines and standardized terminology for translating SMITE 2 content into Russian. It is designed to ensure consistency, clarity, and cultural appropriateness across all game content.

## Table of Contents
1. [Special Translation Rules](#special-translation-rules)
   - [Grammatical Rules](#grammatical-rules)
   - [Formatting Rules](#formatting-rules)
   - [Terminology Rules](#terminology-rules)
   - [Cultural Adaptation](#cultural-adaptation)
   - [Technical Constraints](#technical-constraints)
   - [Exception Handling](#exception-handling)
   - [Implementation Guidelines](#implementation-guidelines)
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
   - [Grammatical Pattern Analysis](#grammatical-pattern-analysis)
   - [Formatting Pattern Analysis](#formatting-pattern-analysis)
   - [Key Recommendations](#key-recommendations)

## Special Translation Rules

### Grammatical Rules

#### Gender and Number Agreement
- **Gender Assignment**: Assign gender based on the natural gender of characters (e.g., "бог" for male gods, "богиня" for female gods).
- **Number Agreement**: Ensure adjectives and verbs agree with the number of the noun they modify.
- **Collective Nouns**: Treat team-related terms as singular entities (e.g., "команда получает").

#### Case System
- **Nominative Case**: Use for subject of sentences and ability names (28.9% in dataset).
- **Genitive Case**: Use for possession and after certain prepositions (26.1% in dataset).
- **Accusative Case**: Use for direct objects and targets of abilities (29.2% in dataset).
- **Dative Case**: Use for indirect objects and beneficiaries of actions (7.3% in dataset).
- **Instrumental Case**: Use for means or instruments (1.6% in dataset).
- **Prepositional Case**: Use after specific prepositions (6.9% in dataset).

#### Verb Forms
- **Aspect**: Use imperfective aspect for ongoing effects and repeatable actions (95.4% in dataset).
- **Perfective Aspect**: Reserve for one-time effects and completed actions (4.6% in dataset).
- **Tense**: Use present tense for describing ability effects and passive bonuses.
- **Imperative**: Use only for direct player instructions.

#### Address Form
- **Formal Address**: Use the formal "Вы" form for all player instructions (98.7% in dataset).
- **Consistency**: Maintain the same address form throughout related text.

#### Adjective Placement
- **Standard Placement**: Place adjectives before the nouns they modify.
- **Special Cases**: Place adjectives after nouns only for technical or specialized terms.

### Formatting Rules

#### Capitalization
- **First Word**: Capitalize the first word of sentences, ability names, and item names (57.4% in dataset).
- **Proper Nouns**: Capitalize names of specific gods, unique locations, and special events.
- **Ability Components**: Do not capitalize common terms within ability descriptions unless at the beginning of a sentence.

#### Punctuation
- **Decimal Separator**: Use comma as decimal separator (97.3% in dataset).
- **Thousands Separator**: Use space as thousands separator (e.g., "1 000").
- **List Separator**: Use semicolons to separate items in a list.
- **Quotation Marks**: Use guillemets («») for direct quotations.
- **Parentheses**: Use for supplementary information and clarifications.

#### Number Formatting
- **Percentages**: Place percentage sign after the number with a space (e.g., "25 %").
- **Units**: Place unit abbreviations after the number with a space (e.g., "50 ед.").
- **Ranges**: Use en dash without spaces (e.g., "5–10").

### Terminology Rules

#### Standardized Terms
- **Damage**: Standardize as "урон" (61.3% in dataset)
- **Health**: Standardize as "здоровье" (20.8% in dataset)
- **Mana**: Standardize as "мана" (8.6% in dataset)
- **Ability**: Standardize as "способность" (38.6% in dataset)
- **Ultimate**: Standardize as "ультимативная способность" (0% in dataset)
- **Cooldown**: Standardize as "перезарядка" (49.4% in dataset)
- **Protection**: Standardize as "защита" (41.0% in dataset)
- **Penetration**: Standardize as "пробивание" (69.0% in dataset)
- **Critical**: Standardize as "критический" (11.1% in dataset)
- **Movement Speed**: Standardize as "скорость передвижения" (58.4% in dataset)
- **Attack Speed**: Standardize as "скорость атаки" (62.2% in dataset)
- **God**: Standardize as "бог" for male, "богиня" for female (24.5% in dataset)
- **Passive**: Standardize as "пассивный" (94.6% in dataset)
- **Stun**: Standardize as "оглушение" (14.3% in dataset)
- **Root**: Standardize as "обездвиживание" (6.7% in dataset)
- **Silence**: Standardize as "немота" (3.8% in dataset)
- **Slow**: Standardize as "замедление" (15.9% in dataset)
- **Buff**: Standardize as "усиление" (18.8% in dataset)
- **Debuff**: Standardize as "ослабление" (3.6% in dataset)
- **Crowd Control**: Standardize as "контроль толпы" (0% in dataset)

#### Abbreviations
- **Standard Abbreviations**: Use only widely recognized abbreviations.
- **Unit Abbreviations**: Use "ед." for units, "с" for seconds, "%" for percent.
- **Consistency**: Do not mix abbreviated and full forms in the same text.

#### Borrowings
- **Transliteration**: Transliterate English gaming terms only when no Russian equivalent exists.
- **Established Terms**: Use established Russian gaming terminology where available.
- **Hybrid Terms**: Avoid mixing Russian and transliterated terms except for established gaming jargon.

### Cultural Adaptation

#### Cultural References
- **Mythology**: Preserve original mythological references but provide Russian equivalents where helpful.
- **Wordplay**: Adapt wordplay to maintain humor while being culturally appropriate.
- **Idioms**: Replace English idioms with Russian equivalents that convey the same meaning.

#### Localization Depth
- **Deep Localization**: Fully adapt cultural references for narrative elements.
- **Light Localization**: Maintain more direct translations for technical gameplay elements.

### Technical Constraints

#### Character Limits
- **UI Space Constraints**: Use abbreviated forms only when necessary due to space limitations.
- **Tooltip Optimization**: Structure tooltips to fit information efficiently in limited space.
- **Abbreviation Consistency**: When abbreviations are necessary, use the same abbreviations consistently.

#### Font Considerations
- **Cyrillic Support**: Ensure all text displays correctly with Cyrillic characters.
- **Text Expansion**: Account for potential text expansion in Russian (typically 20-30% longer than English).

### Exception Handling

#### Untranslatable Terms
- **Proper Names**: Leave god names, unique ability names, and proper nouns untranslated but transliterated.
- **Brand Terms**: Maintain SMITE-specific branded terms with appropriate Russian grammatical endings.

#### Ambiguous Terms
- **Context-Dependent Translation**: Translate terms differently based on context when necessary.
- **Clarification**: Add clarifying words when the Russian translation might be ambiguous.

### Implementation Guidelines

#### Quality Assurance
- **Consistency Check**: Verify terminology consistency across similar abilities and items.
- **Context Review**: Review translations in-game to ensure they make sense in context.
- **Native Review**: Have translations reviewed by native Russian speakers familiar with gaming terminology.

#### Maintenance
- **Terminology Database**: Maintain a database of standardized terms for future updates.
- **Change Tracking**: Document any terminology changes for future reference.
- **Feedback Integration**: Incorporate player feedback on translations when appropriate.

## Game-Specific Glossary

### Core Game Concepts

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| God | бог (m), богиня (f) | Use gender-appropriate form based on character |
| Ability | способность | Standardized from validation (38.6% consistency) |
| Ultimate | ультимативная способность | Standardized from validation (0% consistency) |
| Passive | пассивный | High consistency in dataset (94.6%) |
| Active | активный | Use consistently with "пассивный" |
| Item | предмет | Standard translation |
| Relic | реликвия | Standard translation |
| Consumable | расходуемый предмет | Full form for clarity |
| Level | уровень | Standard translation |
| Experience | опыт | Standard translation |
| Gold | золото | Standard translation |
| Lane | линия | Standard translation |
| Jungle | лес | Standard translation |
| Minion | миньон | Gaming term, maintain transliteration |
| Tower | башня | Standard translation |
| Phoenix | феникс | Mythological term |
| Titan | титан | Mythological term |

### Character Classes & Roles

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| Warrior | воин | Standard translation |
| Guardian | страж | Standard translation |
| Mage | маг | Standard translation |
| Hunter | охотник | Standard translation |
| Assassin | убийца | Standard translation |
| Support | поддержка | Standard translation |
| Carry | керри | Gaming term, maintain transliteration |
| Solo | соло | Gaming term, maintain transliteration |
| Mid | мид | Gaming term, maintain transliteration |
| Jungle (role) | лесник | Standard translation for the role |

### Combat Terminology

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| Damage | урон | Standardized from validation (61.3% consistency) |
| Physical Damage | физический урон | Compound with standardized "урон" |
| Magical Damage | магический урон | Compound with standardized "урон" |
| True Damage | чистый урон | Compound with standardized "урон" |
| Health | здоровье | Standardized from validation (20.8% consistency) |
| Mana | мана | Standardized from validation (8.6% consistency) |
| Protection | защита | Standardized from validation (41.0% consistency) |
| Physical Protection | физическая защита | Compound with standardized "защита" |
| Magical Protection | магическая защита | Compound with standardized "защита" |
| Penetration | пробивание | Standardized from validation (69.0% consistency) |
| Flat Penetration | плоское пробивание | Compound with standardized "пробивание" |
| Percentage Penetration | процентное пробивание | Compound with standardized "пробивание" |
| Critical | критический | Standardized from validation (11.1% consistency) |
| Critical Strike | критический удар | Compound with standardized "критический" |
| Critical Chance | шанс критического удара | Compound with standardized "критический" |
| Lifesteal | вампиризм | Standard translation |
| Healing | исцеление | Standard translation |
| Cooldown | перезарядка | Standardized from validation (49.4% consistency) |
| Cooldown Reduction | сокращение перезарядки | Compound with standardized "перезарядка" |
| Attack Speed | скорость атаки | Standardized from validation (62.2% consistency) |
| Movement Speed | скорость передвижения | Standardized from validation (58.4% consistency) |
| Range | дальность | Standard translation |
| Area of Effect | область действия | Standard translation |
| Projectile | снаряд | Standard translation |

### Crowd Control Effects

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| Crowd Control | контроль толпы | Standardized from validation (0% consistency) |
| Stun | оглушение | Standardized from validation (14.3% consistency) |
| Root | обездвиживание | Standardized from validation (6.7% consistency) |
| Silence | немота | Standardized from validation (3.8% consistency) |
| Slow | замедление | Standardized from validation (15.9% consistency) |
| Cripple | калечение | Standard translation |
| Knockup | подбрасывание | Standard translation |
| Knockback | отбрасывание | Standard translation |
| Pull | притягивание | Standard translation |
| Mesmerize | гипноз | Standard translation |
| Fear | страх | Standard translation |
| Taunt | насмешка | Standard translation |
| Madness | безумие | Standard translation |
| Disarm | обезоруживание | Standard translation |
| Blind | ослепление | Standard translation |
| Intoxicate | опьянение | Standard translation |
| Banish | изгнание | Standard translation |

### Map Elements

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| Map | карта | Standard translation |
| Base | база | Standard translation |
| Fountain | фонтан | Standard translation |
| Buff Camp | лагерь усиления | Compound with standardized "усиление" |
| Red Buff | красное усиление | Compound with standardized "усиление" |
| Blue Buff | синее усиление | Compound with standardized "усиление" |
| Yellow Buff | желтое усиление | Compound with standardized "усиление" |
| Purple Buff | фиолетовое усиление | Compound with standardized "усиление" |
| Fire Giant | огненный гигант | Standard translation |
| Gold Fury | золотая фурия | Standard translation |
| Pyromancer | пиромант | Standard translation |
| Harpy | гарпия | Standard translation |
| Jungle Boss | босс леса | Standard translation |
| Ward | вард | Gaming term, maintain transliteration |
| Sentry Ward | сторожевой вард | Compound with transliterated "вард" |

### Game Modes

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| Conquest | завоевание | Standard translation |
| Arena | арена | Standard translation |
| Joust | поединок | Standard translation |
| Assault | штурм | Standard translation |
| Clash | столкновение | Standard translation |
| Siege | осада | Standard translation |
| MOTD | MOTD | Keep as acronym (Match of the Day) |
| Ranked | рейтинговый | Standard translation |
| Casual | обычный | Standard translation |
| Custom | пользовательский | Standard translation |

### UI Elements

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| Menu | меню | Standard translation |
| Settings | настройки | Standard translation |
| Store | магазин | Standard translation |
| Profile | профиль | Standard translation |
| Friends | друзья | Standard translation |
| Clan | клан | Standard translation |
| Match History | история матчей | Standard translation |
| Leaderboard | таблица лидеров | Standard translation |
| Queue | очередь | Standard translation |
| Lobby | лобби | Gaming term, maintain transliteration |
| Loading Screen | экран загрузки | Standard translation |
| Victory | победа | Standard translation |
| Defeat | поражение | Standard translation |
| Surrender | сдаться | Standard translation |
| Pause | пауза | Standard translation |
| Chat | чат | Standard translation |
| Scoreboard | табло | Standard translation |

### Stats and Attributes

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| Power | сила | Standard translation |
| Physical Power | физическая сила | Compound with "сила" |
| Magical Power | магическая сила | Compound with "сила" |
| Defense | защита | Same as "Protection" for consistency |
| Armor | броня | Alternative term for specific items |
| Shield | щит | Standard translation |
| Regeneration | восстановление | Standard translation |
| Health Regeneration | восстановление здоровья | Compound with standardized "здоровье" |
| Mana Regeneration | восстановление маны | Compound with standardized "мана" |
| Aura | аура | Standard translation |
| Buff | усиление | Standardized from validation (18.8% consistency) |
| Debuff | ослабление | Standardized from validation (3.6% consistency) |
| Immunity | иммунитет | Standard translation |
| Resistance | сопротивление | Standard translation |
| Mitigation | смягчение | Standard translation |

### Ability Mechanics

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| Cast | применение | Standard translation |
| Channel | канализация | Standard translation |
| Charge | заряд | Standard translation |
| Dash | рывок | Standard translation |
| Leap | прыжок | Standard translation |
| Teleport | телепорт | Standard translation |
| Execute | казнь | Standard translation |
| Heal | исцеление | Standard translation |
| Shield | щит | Standard translation |
| Stealth | невидимость | Standard translation |
| Reveal | обнаружение | Standard translation |
| Cleanse | очищение | Standard translation |
| Reset | сброс | Standard translation |
| Stack | заряд | Standard translation |
| Proc | срабатывание | Standard translation |
| Toggle | переключение | Standard translation |
| Stance | стойка | Standard translation |

### Item Categories

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| Starter Item | начальный предмет | Standard translation |
| Boots | сапоги | Standard translation |
| Offensive | наступательный | Standard translation |
| Defensive | оборонительный | Standard translation |
| Utility | универсальный | Standard translation |
| Aura Item | аура-предмет | Standard translation |
| Consumable | расходуемый предмет | Standard translation |
| Potion | зелье | Standard translation |
| Ward | вард | Gaming term, maintain transliteration |
| Relic | реликвия | Standard translation |
| Tier 1/2/3 | уровень 1/2/3 | Standard translation |

### Common Abbreviations

| English Abbreviation | Russian Translation | Notes |
|----------------------|---------------------|-------|
| HP | ОЗ (очки здоровья) | Standard abbreviation |
| MP | ОМ (очки маны) | Standard abbreviation |
| CC | КТ (контроль толпы) | Abbreviation of standardized term |
| CDR | СП (сокращение перезарядки) | Abbreviation of standardized term |
| DoT | УВ (урон во времени) | Standard abbreviation |
| AoE | ОД (область действия) | Standard abbreviation |
| AS | СА (скорость атаки) | Abbreviation of standardized term |
| MS | СП (скорость передвижения) | Abbreviation of standardized term |
| DPS | УВС (урон в секунду) | Standard abbreviation |
| AA | ОА (обычная атака) | Standard abbreviation |

### Specialized Gaming Terms

| English Term | Russian Translation | Notes |
|--------------|---------------------|-------|
| Meta | мета | Gaming term, maintain transliteration |
| Gank | ганк | Gaming term, maintain transliteration |
| Farm | фарм | Gaming term, maintain transliteration |
| Poke | пок | Gaming term, maintain transliteration |
| Burst | бурст | Gaming term, maintain transliteration |
| Sustain | сустейн | Gaming term, maintain transliteration |
| Rotation | ротация | Gaming term, maintain transliteration |
| Invade | вторжение | Standard translation |
| Secure | захват | Standard translation |
| Snowball | сноуболл | Gaming term, maintain transliteration |
| Teamfight | командный бой | Standard translation |
| Splitpush | сплитпуш | Gaming term, maintain transliteration |
| Backdoor | бэкдор | Gaming term, maintain transliteration |
| Dive | дайв | Gaming term, maintain transliteration |
| Peel | пил | Gaming term, maintain transliteration |
| Kite | кайт | Gaming term, maintain transliteration |
| Juke | джук | Gaming term, maintain transliteration |
| Bait | приманка | Standard translation |

### Standardized UI Abbreviations for Space Constraints

| Full Russian Term | Abbreviated Form | Notes |
|-------------------|------------------|-------|
| здоровье | здор. | For UI space constraints |
| мана | мана | Short enough, no abbreviation needed |
| урон | урон | Short enough, no abbreviation needed |
| защита | защ. | For UI space constraints |
| скорость атаки | ск. атаки | For UI space constraints |
| скорость передвижения | ск. передв. | For UI space constraints |
| перезарядка | перезар. | For UI space constraints |
| пробивание | проб. | For UI space constraints |
| контроль толпы | КТ | For UI space constraints |
| усиление | усил. | For UI space constraints |
| ослабление | ослаб. | For UI space constraints |

## Validation Insights

### Term Consistency Analysis

The validation process revealed several key terms with low consistency in the dataset:

| English Term | Primary Russian Translation | Consistency | Recommendation |
|--------------|----------------------------|-------------|----------------|
| Ultimate | ультимативная способность | 0.0% | Standardize this term |
| Crowd Control | контроль толпы | 0.0% | Standardize this term |
| Debuff | ослабление | 3.6% | Standardize this term |
| Silence | немота | 3.8% | Standardize this term |
| Root | обездвиживание | 6.7% | Standardize this term |
| Mana | мана | 8.6% | Standardize this term |
| Critical | критический | 11.1% | Standardize this term |
| Stun | оглушение | 14.3% | Standardize this term |
| Slow | замедление | 15.9% | Standardize this term |
| Buff | усиление | 18.8% | Standardize this term |
| Health | здоровье | 20.8% | Standardize this term |
| God | бог | 24.5% | Standardize with gender variants |
| Ability | способность | 38.6% | Standardize this term |
| Protection | защита | 41.0% | Standardize this term |
| Cooldown | перезарядка | 49.4% | Standardize this term |
| Movement Speed | скорость передвижения | 58.4% | Standardize this term |
| Damage | урон | 61.3% | Standardize this term |
| Attack Speed | скорость атаки | 62.2% | Standardize this term |
| Penetration | пробивание | 69.0% | Standardize this term |
| Passive | пассивный | 94.6% | Continue using this term |

### Grammatical Pattern Analysis

#### Address Form Usage
- Formal 'Вы' form: 98.7%
- Informal 'ты' form: 1.3%
- **Recommendation**: Standardize on formal 'Вы' form for player instructions.

#### Verb Aspect Usage
- Imperfective aspect: 95.4%
- Perfective aspect: 4.6%
- **Recommendation**: Continue using imperfective aspect for ongoing effects and perfective for one-time effects.

#### Case Usage Patterns
- Nominative: 28.9%
- Genitive: 26.1%
- Dative: 7.3%
- Accusative: 29.2%
- Instrumental: 1.6%
- Prepositional: 6.9%

### Formatting Pattern Analysis

#### Capitalization Patterns
- First word capitalized only: 57.4%
- All lowercase: 9.2%

#### Number Formatting
- Decimal comma (e.g., 2,5): 97.3%
- Decimal period (e.g., 2.5): 2.7%
- **Recommendation**: Standardize on decimal comma.

### Key Recommendations

1. **Address Form**: Use formal "Вы" form consistently for all player instructions (98.7% in dataset).
2. **Verb Aspect**: Use imperfective aspect for ongoing effects and repeatable actions (95.4% in dataset).
3. **Number Formatting**: Use comma as decimal separator (97.3% in dataset).
4. **Terminology Standardization**: Standardize key terms with low consistency, particularly:
   - "Ultimate" (ультимативная способность)
   - "Crowd Control" (контроль толпы)
   - "Health" (здоровье)
   - "Ability" (способность)
5. **Gender Forms**: Use appropriate gender forms for gods (бог/богиня) based on character gender.
6. **Case Usage**: Follow natural Russian case usage patterns, with emphasis on nominative (28.9%) and accusative (29.2%) cases for ability descriptions.
7. **Gaming Terms**: Maintain established transliterations for gaming-specific terms that have no good Russian equivalents.
8. **Abbreviations**: Use standardized abbreviations only when necessary for UI space constraints.
