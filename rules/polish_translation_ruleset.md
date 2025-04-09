# Polish (plPL) Translation Ruleset for SMITE 2

This comprehensive document provides the special translation rules and game-specific glossary for translating SMITE 2 content from English to Polish. These guidelines are designed to ensure consistency, accuracy, and cultural appropriateness in the Polish localization.

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

## Special Translation Rules

### 1. Grammatical Rules

#### 1.1 Gender Assignment

- **Consistency**: Assign consistent grammatical gender to all game terms
  - Masculine: przedmiot (item), atak (attack), poziom (level), miecz (sword)
  - Feminine: zdolność (ability), tarcza (shield), moc (power), wieża (tower)
  - Neuter: zdrowie (health), złoto (gold), wzmocnienie (buff), osłabienie (debuff)

- **Character-specific gender**:
  - Use appropriate gender forms for character classes based on the character's gender
  - Example: "Warrior" → "Wojownik" (m) or "Wojowniczka" (f)
  - Example: "Guardian" → "Strażnik" (m) or "Strażniczka" (f)

- **New terminology**:
  - For new game terms, assign gender based on:
    1. Established Polish gaming terminology
    2. The gender of the closest Polish equivalent
    3. Phonological patterns for borrowed terms (consonant-ending terms typically masculine)

#### 1.2 Case Declension

- **Nominative case**: Use for ability names, item names, and UI labels
- **Genitive case**: Use for:
  - Possession relationships (e.g., "Moc Posejdona" - Power of Poseidon)
  - Negation (e.g., "Brak many" - No mana)
  - After numbers 2-4 in singular form (e.g., "2 punkty zdrowia")
  - After numbers 5+ in plural form (e.g., "5 punktów zdrowia")

- **Accusative case**: Use for:
  - Direct objects (e.g., "Zadaje obrażenia" - Deals damage)
  - Duration expressions (e.g., "na 5 sekund" - for 5 seconds)

- **Instrumental case**: Use for:
  - Means/instruments (e.g., "z mieczem" - with a sword)
  - After certain prepositions (e.g., "pod wieżą" - under the tower)

- **Locative case**: Use for:
  - Locations (e.g., "w dżungli" - in the jungle)
  - After certain prepositions (e.g., "o zdolności" - about the ability)

- **Dative case**: Use for:
  - Indirect objects (e.g., "daje sojusznikom" - gives to allies)
  - After certain prepositions (e.g., "ku wrogom" - towards enemies)

- **Vocative case**: Rarely used, primarily for direct address in character dialogue

#### 1.3 Verb Aspect and Tense

- **Aspect choice**:
  - **Prefer perfective aspect (69.1% in dataset)** for one-time, completed effects
    - Example: "zadaje obrażenia" (deals damage - single instance)
    - Example: "ogłuszy przeciwników" (will stun enemies - completed action)
  
  - Use imperfective aspect for ongoing or repeated effects
    - Example: "leczy co sekundę" (heals every second - repeated action)
    - Example: "zwiększa prędkość ruchu" (increases movement speed - ongoing effect)

- **Tense usage**:
  - Present tense for ability descriptions and passive effects
  - Future perfective for triggered effects
  - Imperative mood for player instructions

#### 1.4 Adjective Agreement

- Ensure adjectives agree with nouns in gender, number, and case
  - Masculine: "fizyczny atak" (physical attack)
  - Feminine: "magiczna moc" (magical power)
  - Neuter: "wysokie obrażenia" (high damage)

- For compound terms, maintain agreement throughout
  - "zwiększona prędkość ataku" (increased attack speed)
  - "zmniejszony czas odnowienia" (reduced cooldown time)

#### 1.5 Number Agreement

- **Numerical values**: Use appropriate forms based on the number
  - 1: singular nominative (1 punkt)
  - 2-4: plural genitive singular (2 punkty)
  - 5+: plural genitive plural (5 punktów)

- **Decimal values**: Treat as 5+ for grammatical purposes
  - "2,5 sekundy" (2.5 seconds - genitive singular)

- **Percentage values**: No space between number and % symbol
  - "25% szansy" (25% chance)

#### 1.6 Address Form

- **Use informal "ty" form (98.5% in dataset)** consistently for:
  - Player instructions
  - Tooltips
  - System messages
  - Tutorial content

- Avoid formal "Pan/Pani" forms (only 1.5% in dataset) unless specifically required for character dialogue

### 2. Formatting Rules

#### 2.1 Punctuation

- Use Polish punctuation conventions:
  - **Decimal separator: comma (,) instead of period (.)** (180 instances vs. 4 instances in dataset)
    - English: 2.5 seconds → Polish: 2,5 sekundy
  
  - **Thousands separator: space instead of comma** (2 instances vs. 4 instances in dataset)
    - English: 1,000 gold → Polish: 1 000 złota
  
  - **Quotation marks: „text" (lower-upper) instead of "text"** (Currently underused - only 1 instance vs. 672 instances of English quotes)
    - English: "Victory" → Polish: „Zwycięstwo"
  
  - **Dialogue dash: em dash (—) for dialogue** (Currently underused - only 5 instances)
    - English: "Run!" he said. → Polish: — Uciekaj! — powiedział.

- Maintain English punctuation only for:
  - Technical notations
  - Version numbers
  - Code elements

#### 2.2 Number Formatting

- **Units**: Space between number and unit
  - English: 5s → Polish: 5 s
  - Exception: No space with percentage (50%)

- **Ranges**: Em dash without spaces
  - English: 50-100 damage → Polish: 50–100 obrażeń

- **Mathematical operations**: Spaces around operators
  - English: 50+10 → Polish: 50 + 10

- **Negative values**: No space after minus sign
  - English: -50 health → Polish: -50 zdrowia

#### 2.3 Capitalization

- **Less capitalization than English**:
  - Ability names: Only first word capitalized unless proper noun
    - English: "Eagle's Rally" → Polish: "Zew orła"
  
  - Item names: Only first word capitalized unless proper noun
    - English: "Warrior's Axe" → Polish: "Topór wojownika"
  
  - Game modes: Not capitalized
    - English: "Conquest" → Polish: "podbój"
  
  - Character classes: Not capitalized
    - English: "Warrior" → Polish: "wojownik"

- **Always capitalize**:
  - God names and proper nouns
  - First word of sentences
  - Acronyms (AOE, CC)

#### 2.4 Hyphenation and Compound Words

- Use hyphen (-) for compound adjectives
  - "magiczno-fizyczny" (magical-physical)

- Avoid excessive compound words; use prepositional phrases when clearer
  - Instead of "obrażeniaodczarów" use "obrażenia od czarów" (damage from spells)

#### 2.5 Abbreviations

- Use Polish abbreviations when space is limited:
  - "pkt." for "punkty" (points)
  - "sek." for "sekundy" (seconds)
  - "obr." for "obrażenia" (damage)
  - "zdrow." for "zdrowie" (health)

- Retain English gaming abbreviations that are widely recognized:
  - "AOE" for "Area of Effect"
  - "CC" for "Crowd Control" (12.5% usage in dataset)
  - "CD" for "Cooldown"

### 3. Terminology Rules

#### 3.1 Translation Approaches

- **Direct translation** (preferred for most terms):
  - Health → Zdrowie (15.2% consistency, needs improvement)
  - Damage → Obrażenia (59.7% consistency, needs improvement)
  - Movement Speed → Prędkość ruchu (1.7% consistency, needs significant improvement)

- **Borrowing** (for established gaming terms):
  - Gank → Gank (100% borrowed in dataset)
  - Buff → Buff/Wzmocnienie (73.9% borrowed vs. 13.2% translated in dataset)
  - Tank → Tank (57.1% borrowed vs. 42.9% other in dataset)

- **Calque** (for compound terms):
  - Cooldown → Czas odnowienia (70.0% consistency in dataset)
  - Lifesteal → Kradzież życia (61.2% consistency in dataset)
  - Crowd control → Kontrola tłumu (8.3% consistency, needs improvement)

- **Functional equivalent** (when direct translation doesn't convey meaning):
  - Skillshot → Strzał precyzyjny
  - Execute → Egzekucja
  - Cleave → Cios obszarowy

#### 3.2 Consistency Guidelines

- Refer to the glossary for standardized translations
- Maintain consistency within related term families
  - All damage types (physical, magical, true)
  - All crowd control effects
  - All stat categories

- When multiple translations exist, prefer:
  1. The most widely used term in Polish gaming community
  2. The term with clearest meaning to new players
  3. The term that best fits UI space constraints

#### 3.3 Terminology Standardization

- **Standardize key gameplay terms with low consistency:**
  - God → Bóg (19.8% consistency)
  - Ability → Zdolność (35.2% consistency)
  - Passive → Pasywna (2.3% consistency)
  - Ultimate → Umiejętność specjalna (0.0% consistency)
  - Movement Speed → Prędkość ruchu (1.7% consistency)
  - Attack Speed → Prędkość ataku (0.7% consistency)
  - Penetration → Przebicie (0.0% consistency)
  - Root → Unieruchomienie (0.0% consistency)
  - Silence → Uciszenie (0.0% consistency)
  - Lane → Aleja (0.0% consistency)
  - Ward → Totem (0.0% consistency)

- **Standardize crowd control terminology:**
  - Stun → Ogłuszenie (6.1% consistency, needs improvement)
  - Root → Unieruchomienie (0.0% consistency, needs significant improvement)
  - Silence → Uciszenie (0.0% consistency, needs significant improvement)
  - Slow → Spowolnienie (17.7% consistency, needs improvement)

### 4. Cultural Adaptation

#### 4.1 Mythological References

- Use established Polish equivalents for Greek/Roman deities:
  - Zeus → Zeus (same)
  - Athena → Atena (Polish spelling)
  - Mercury → Merkury (Polish spelling)
  - Hercules → Herkules (Polish spelling)

- Use established Polish equivalents for Norse deities:
  - Thor → Thor (same)
  - Odin → Odyn (Polish spelling)
  - Loki → Loki (same)

- Maintain original names for deities from other pantheons unless established Polish equivalents exist

#### 4.2 Cultural References and Humor

- Adapt puns and wordplay to maintain humor rather than literal translation
- Replace culture-specific references with Polish equivalents when necessary
- Maintain character personality in dialogue adaptation
- Consider Polish gaming culture norms and preferences

#### 4.3 Sensitivity Considerations

- Avoid politically sensitive content
- Respect cultural and religious sensitivities
- Maintain PEGI rating requirements in translation
- Ensure translations are appropriate for the Polish market

### 5. Technical Constraints

#### 5.1 Text Expansion

- Polish text typically 15-20% longer than English
- Strategies for managing text expansion:
  - Use abbreviations in UI elements
  - Use shorter synonyms when available
  - Omit implied subjects (Polish is a pro-drop language)
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
  - Proper names without established Polish equivalents
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
  1. Consider existing Polish gaming terminology
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

- Check for natural-sounding Polish:
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

## Game-Specific Glossary

### Core Game Concepts

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| SMITE 2 | SMITE 2 | N/A | Game title remains untranslated |
| God | Bóg (m) / Bogini (f) | 19.8% | Gender-specific forms used based on character; needs standardization |
| Ability | Zdolność (f) | 35.2% | Plural: Zdolności; needs standardization |
| Passive | Pasywna (f) | 2.3% | Often used as "Umiejętność pasywna"; needs standardization |
| Ultimate | Umiejętność specjalna (f) | 0.0% | Literally "special ability"; needs standardization |
| Level | Poziom (m) | 67.3% | Used for character levels and ability levels |
| Experience | Doświadczenie (n) | 40.0% | Abbreviated as "Dośw." in UI constraints |
| Gold | Złoto (n) | 27.7% | Currency unit; needs standardization |
| Item | Przedmiot (m) | 96.9% | Plural: Przedmioty; high consistency |
| Match | Mecz (m) | N/A | Plural: Mecze |
| Victory | Zwycięstwo (n) | N/A | |
| Defeat | Porażka (f) | N/A | |
| Team | Drużyna (f) | N/A | |
| Ally | Sojusznik (m) | N/A | Feminine: Sojuszniczka (f) |
| Enemy | Wróg (m) | N/A | Feminine: Wroga (f), Plural: Wrogowie |
| Minion | Sługus (m) | 100.0% | High consistency |
| Jungle | Dżungla (f) | 3.1% | Needs standardization |
| Lane | Aleja (f) | 0.0% | Sometimes "Linia" (f); needs standardization |
| Map | Mapa (f) | N/A | |
| Fountain | Fontanna (f) | N/A | |
| Base | Baza (f) | N/A | |
| Respawn | Odrodzenie (n) | N/A | Verb form: Odrodzić się |
| Spawn | Pojawienie (n) | N/A | Verb form: Pojawić się |
| Queue | Kolejka (f) | N/A | |
| Matchmaking | Dobieranie graczy (n) | N/A | |
| Leaderboard | Tablica wyników (f) | N/A | |
| Scoreboard | Tablica wyników (f) | N/A | |
| Tutorial | Samouczek (m) | N/A | |
| Practice | Praktyka (f) | N/A | |
| Surrender | Poddanie (n) | N/A | Verb form: Poddać się |
| Pause | Pauza (f) | N/A | Verb form: Spauzować |
| Disconnect | Rozłączenie (n) | N/A | Verb form: Rozłączyć się |
| Reconnect | Ponowne połączenie (n) | N/A | Verb form: Połączyć się ponownie |

### Character Classes & Roles

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| Assassin | Zabójca (m) | N/A | Feminine: Zabójczyni (f) |
| Guardian | Strażnik (m) | N/A | Feminine: Strażniczka (f) |
| Hunter | Łowca (m) | N/A | Feminine: Łowczyni (f) |
| Mage | Mag (m) | N/A | Feminine: Magini (f) |
| Warrior | Wojownik (m) | N/A | Feminine: Wojowniczka (f) |
| Support | Wsparcie (n) | 33.3% | Role name, not class |
| Carry | Carry | N/A | Often untranslated, sometimes "Nośnik" |
| Solo | Solo | N/A | Untranslated role name |
| Jungler | Dżungler (m) | N/A | Polonized from English |
| Mid | Środkowy (m) | N/A | Literally "middle one" |
| Tank | Tank (m) | 57.1% | Borrowed term, sometimes "Czołg" |
| Damage Dealer | Zadający obrażenia (m) | N/A | |
| Healer | Uzdrowiciel (m) | N/A | Feminine: Uzdrowicielka (f) |
| Initiator | Inicjator (m) | N/A | Feminine: Inicjatorka (f) |
| Pusher | Puszujący (m) | N/A | Gaming term, from English "push" |
| Ganker | Ganker (m) | N/A | Borrowed gaming term |
| Roamer | Wędrowiec (m) | N/A | Feminine: Wędrowczyni (f) |
| Flex | Flex | N/A | Untranslated role name |
| Bruiser | Osiłek (m) | N/A | |
| Caster | Rzucający czary (m) | N/A | |
| Auto-Attack | Auto-atakujący (m) | N/A | |

### Combat Terminology

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| Damage | Obrażenia (n, plural) | 59.7% | No singular form commonly used; needs standardization |
| Physical | Fizyczny (adj) | 38.2% | Agrees with noun gender; needs standardization |
| Magical | Magiczny (adj) | 39.4% | Agrees with noun gender; needs standardization |
| Attack | Atak (m) | 98.5% | Verb form: Atakować; high consistency |
| Basic Attack | Atak podstawowy (m) | 22.2% | Needs standardization |
| Health | Zdrowie (n) | 15.2% | Sometimes "Punkty zdrowia" (health points); needs standardization |
| Mana | Mana (f) | 8.6% | Borrowed term; needs standardization |
| Protection | Ochrona (f) | 34.3% | Needs standardization |
| Penetration | Przebicie (n) | 0.0% | Needs standardization |
| Critical | Krytyczny (adj) | 31.6% | Agrees with noun gender; needs standardization |
| Critical Strike | Trafienie krytyczne (n) | N/A | |
| Lifesteal | Kradzież życia (f) | 61.2% | Literally "life theft" |
| Healing | Leczenie (n) | 52.2% | Verb form: Leczyć |
| Shield | Tarcza (f) | 26.9% | Needs standardization |
| Movement Speed | Prędkość ruchu (f) | 1.7% | Needs significant standardization |
| Attack Speed | Prędkość ataku (f) | 0.7% | Needs significant standardization |
| Cooldown | Czas odnowienia (m) | 70.0% | Literally "renewal time"; good consistency |
| Range | Zasięg (m) | 81.8% | High consistency |
| Area of Effect | Obszar działania (m) | N/A | Often abbreviated as AOE |
| Duration | Czas trwania (m) | 39.3% | Needs standardization |
| Kill | Zabójstwo (n) | 48.2% | Verb form: Zabić; needs standardization |
| Death | Śmierć (f) | 35.6% | Needs standardization |
| Assist | Asysta (f) | 6.7% | Needs standardization |
| Buff | Wzmocnienie (n) | 22.2% | Sometimes borrowed as "Buff" (m) (73.9% in dataset); consider standardizing to "Buff" |
| Debuff | Osłabienie (n) | 10.9% | Sometimes borrowed as "Debuff" (m); needs standardization |
| True Damage | Obrażenia nieuchronne (n) | N/A | Literally "inevitable damage" |
| Damage over Time | Obrażenia w czasie (n) | N/A | Abbreviated as OWT |
| Heal over Time | Leczenie w czasie (n) | N/A | |
| Burst Damage | Nagłe obrażenia (n) | N/A | Sometimes borrowed as "Burst" |
| Cleave | Cios obszarowy (m) | N/A | |
| Splash Damage | Obrażenia rozbryzgowe (n) | N/A | |
| Projectile | Pocisk (m) | N/A | |
| Homing | Naprowadzany (adj) | N/A | |
| Skillshot | Strzał precyzyjny (m) | N/A | |
| Hitbox | Obszar trafienia (m) | N/A | Sometimes borrowed as "Hitbox" |
| Collision | Kolizja (f) | N/A | |
| Immunity | Odporność (f) | N/A | |
| Invulnerability | Niewrażliwość (f) | N/A | |
| Resistance | Odporność (f) | N/A | |
| Weakness | Słabość (f) | N/A | |
| Vulnerability | Wrażliwość (f) | N/A | |

### Crowd Control Effects

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| Crowd Control | Kontrola tłumu (f) | 8.3% | Often abbreviated as CC (12.5% in dataset); needs standardization |
| Stun | Ogłuszenie (n) | 6.1% | Verb form: Ogłuszyć; needs standardization |
| Slow | Spowolnienie (n) | 17.7% | Verb form: Spowolnić; needs standardization |
| Root | Unieruchomienie (n) | 0.0% | Verb form: Unieruchomić; needs standardization |
| Silence | Uciszenie (n) | 0.0% | Verb form: Uciszyć; needs standardization |
| Disarm | Rozbrojenie (n) | N/A | Verb form: Rozbroić |
| Cripple | Okaleczenie (n) | N/A | Verb form: Okaleczyć |
| Knockup | Podrzucenie (n) | N/A | Verb form: Podrzucić |
| Knockback | Odrzucenie (n) | N/A | Verb form: Odrzucić |
| Pull | Przyciągnięcie (n) | N/A | Verb form: Przyciągnąć |
| Mesmerize | Zahipnotyzowanie (n) | N/A | Verb form: Zahipnotyzować |
| Taunt | Sprowokowanie (n) | N/A | Verb form: Sprowokować |
| Fear | Przestraszenie (n) | N/A | Verb form: Przestraszyć |
| Intoxicate | Upojenie (n) | N/A | Verb form: Upoić |
| Madness | Szaleństwo (n) | N/A | |
| Blind | Oślepienie (n) | N/A | Verb form: Oślepić |
| Banish | Wygnanie (n) | N/A | Verb form: Wygnać |
| Polymorph | Polimorfizm (m) | N/A | Verb form: Przemienić |
| Charm | Urok (m) | N/A | Verb form: Oczarować |
| Disorient | Dezorientacja (f) | N/A | Verb form: Zdezorientować |
| Stagger | Zachwianie (n) | N/A | Verb form: Zachwiać |
| Tremor | Drżenie (n) | N/A | Verb form: Zadrżeć |
| Vortex | Wir (m) | N/A | |
| Grab | Chwyt (m) | N/A | Verb form: Chwycić |
| Imprison | Uwięzienie (n) | N/A | Verb form: Uwięzić |

### Map Elements

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| Tower | Wieża (f) | 45.0% | Needs standardization |
| Phoenix | Feniks (m) | 100.0% | High consistency |
| Titan | Tytan (m) | 89.2% | High consistency |
| Jungle Camp | Obóz w dżungli (m) | N/A | |
| Buff Camp | Obóz wzmocnień (m) | N/A | |
| Ward | Totem (m) | 0.0% | Sometimes "Strażnik" (m); needs standardization |
| Fire Giant | Ognisty Gigant (m) | N/A | |
| Gold Fury | Złota Furia (f) | N/A | |
| Pyromancer | Piromanta (m) | N/A | |
| Harpy | Harpia (f) | N/A | |
| Scorpion | Skorpion (m) | N/A | |
| Obelisk | Obelisk (m) | N/A | |
| Portal | Portal (m) | N/A | |
| Shrine | Kapliczka (f) | N/A | |
| Altar | Ołtarz (m) | N/A | |
| Fountain | Fontanna (f) | N/A | |
| Spawn Point | Punkt odrodzenia (m) | N/A | |
| Fog of War | Mgła wojny (f) | N/A | |
| Line of Sight | Linia wzroku (f) | N/A | |
| Wall | Ściana (f) | N/A | |
| Bush | Krzak (m) | N/A | |
| Grass | Trawa (f) | N/A | |
| Path | Ścieżka (f) | N/A | |
| Bridge | Most (m) | N/A | |
| Gate | Brama (f) | N/A | |
| Barrier | Bariera (f) | N/A | |
| Objective | Cel (m) | N/A | |
| Structure | Struktura (f) | N/A | |
| Inhibitor | Inhibitor (m) | N/A | |
| Barracks | Koszary (f, plural) | N/A | |

### Game Modes

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| Conquest | Podbój (m) | N/A | Main game mode |
| Arena | Arena (f) | N/A | |
| Joust | Pojedynek (m) | N/A | |
| Assault | Szturm (m) | N/A | |
| Clash | Starcie (n) | N/A | |
| Siege | Oblężenie (n) | N/A | |
| MOTD | MOTD | N/A | "Match of the Day" - abbreviation kept |
| Custom | Niestandardowy (adj) | N/A | "Niestandardowy mecz" (custom match) |
| Training | Trening (m) | N/A | |
| Co-op | Współpraca (f) | N/A | |
| Ranked | Rankingowy (adj) | N/A | "Mecz rankingowy" (ranked match) |
| Casual | Zwykły (adj) | N/A | "Mecz zwykły" (casual match) |
| Duel | Pojedynek (m) | N/A | |
| Tournament | Turniej (m) | N/A | |
| Competitive | Rywalizacja (f) | N/A | |
| Practice | Praktyka (f) | N/A | |
| Tutorial | Samouczek (m) | N/A | |
| Spectator | Obserwator (m) | N/A | |
| Replay | Powtórka (f) | N/A | |
| Qualifying | Kwalifikacje (f, plural) | N/A | |
| Placement | Umieszczenie (n) | N/A | |
| Season | Sezon (m) | N/A | |
| Split | Podział (m) | N/A | |
| Phase | Faza (f) | N/A | |

### UI Elements

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| Menu | Menu (n) | N/A | Indeclinable |
| Settings | Ustawienia (n, plural) | N/A | |
| Profile | Profil (m) | N/A | |
| Friends | Znajomi (m, plural) | N/A | |
| Clan | Klan (m) | N/A | |
| Achievements | Osiągnięcia (n, plural) | N/A | |
| Quests | Zadania (n, plural) | N/A | |
| Battle Pass | Przepustka bojowa (f) | N/A | |
| Store | Sklep (m) | N/A | |
| Chest | Skrzynia (f) | N/A | |
| Bundle | Pakiet (m) | N/A | |
| Skin | Skórka (f) | N/A | |
| Voice Pack | Pakiet głosowy (m) | N/A | |
| Loading Frame | Ramka ładowania (f) | N/A | |
| Loading Screen | Ekran ładowania (m) | N/A | |
| Avatar | Awatar (m) | N/A | |
| Emote | Emotka (f) | N/A | |
| Announcer Pack | Pakiet komentatora (m) | N/A | |
| Button | Przycisk (m) | N/A | |
| Slider | Suwak (m) | N/A | |
| Checkbox | Pole wyboru (n) | N/A | |
| Dropdown | Lista rozwijana (f) | N/A | |
| Tab | Zakładka (f) | N/A | |
| Window | Okno (n) | N/A | |
| Tooltip | Podpowiedź (f) | N/A | |
| Notification | Powiadomienie (n) | N/A | |
| Alert | Alert (m) | N/A | |
| Popup | Wyskakujące okno (n) | N/A | |
| Icon | Ikona (f) | N/A | |
| Badge | Odznaka (f) | N/A | |
| Banner | Baner (m) | N/A | |
| Cursor | Kursor (m) | N/A | |
| Highlight | Podświetlenie (n) | N/A | |
| Scroll | Przewijanie (n) | N/A | |
| Zoom | Przybliżenie (n) | N/A | |

### Stats and Attributes

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| Power | Moc (f) | N/A | |
| Physical Power | Moc fizyczna (f) | N/A | |
| Magical Power | Moc magiczna (f) | N/A | |
| Physical Protection | Ochrona fizyczna (f) | N/A | |
| Magical Protection | Ochrona magiczna (f) | N/A | |
| Health Regeneration | Regeneracja zdrowia (f) | N/A | |
| Mana Regeneration | Regeneracja many (f) | N/A | |
| Cooldown Reduction | Redukcja czasu odnowienia (f) | N/A | Often abbreviated as "Redukcja odnowienia" |
| Crowd Control Reduction | Redukcja kontroli tłumu (f) | N/A | |
| Penetration | Przebicie (n) | 0.0% | Needs standardization |
| Flat Penetration | Przebicie stałe (n) | N/A | |
| Percentage Penetration | Przebicie procentowe (n) | N/A | |
| Critical Strike Chance | Szansa na trafienie krytyczne (f) | N/A | |
| Critical Strike Damage | Obrażenia od trafienia krytycznego (n) | N/A | |
| Attack Speed | Prędkość ataku (f) | 0.7% | Needs significant standardization |
| Movement Speed | Prędkość ruchu (f) | 1.7% | Needs significant standardization |
| Slow Immunity | Odporność na spowolnienie (f) | N/A | |
| Knockup Immunity | Odporność na podrzucenie (f) | N/A | |
| Tenacity | Wytrwałość (f) | N/A | |
| Resilience | Odporność (f) | N/A | |
| Mitigation | Złagodzenie (n) | N/A | |
| Amplification | Wzmocnienie (n) | N/A | |
| Reduction | Redukcja (f) | N/A | |
| Conversion | Konwersja (f) | N/A | |
| Scaling | Skalowanie (n) | N/A | |
| Diminishing Returns | Malejące zwroty (m, plural) | N/A | |
| Threshold | Próg (m) | N/A | |
| Cap | Limit (m) | N/A | |
| Base Value | Wartość podstawowa (f) | N/A | |
| Bonus Value | Wartość dodatkowa (f) | N/A | |
| Total Value | Wartość całkowita (f) | N/A | |

### Ability Mechanics

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| Cooldown | Czas odnowienia (m) | 70.0% | Good consistency |
| Cost | Koszt (m) | N/A | |
| Range | Zasięg (m) | 81.8% | High consistency |
| Radius | Promień (m) | N/A | |
| Duration | Czas trwania (m) | 39.3% | Needs standardization |
| Charge | Ładunek (m) | N/A | |
| Stack | Stos (m) | N/A | Verb form: Stackować (borrowed) |
| Channel | Kanałowanie (n) | N/A | Verb form: Kanałować |
| Cast | Rzucenie (n) | N/A | Verb form: Rzucić |
| Projectile | Pocisk (m) | N/A | |
| Skillshot | Strzał precyzyjny (m) | N/A | |
| Dash | Zryw (m) | N/A | |
| Leap | Skok (m) | N/A | |
| Teleport | Teleport (m) | N/A | |
| Stealth | Ukrycie (n) | N/A | Sometimes "Niewidzialność" (f) |
| Execute | Egzekucja (f) | N/A | Verb form: Egzekwować |
| Heal | Leczenie (n) | 52.2% | Verb form: Leczyć |
| Shield | Tarcza (f) | 26.9% | Verb form: Osłaniać; needs standardization |
| Cleanse | Oczyszczenie (n) | N/A | Verb form: Oczyścić |
| Reflect | Odbicie (n) | N/A | Verb form: Odbić |
| Amplify | Wzmocnienie (n) | N/A | Verb form: Wzmocnić |
| Reduce | Redukcja (f) | N/A | Verb form: Redukować |
| Toggle | Przełączenie (n) | N/A | Verb form: Przełączyć |
| Passive | Pasywna (f) | 2.3% | Needs standardization |
| Active | Aktywna (f) | N/A | |
| Aura | Aura (f) | N/A | |
| Proc | Wyzwolenie (n) | N/A | Gaming term, sometimes borrowed as "Proc" |
| Trigger | Wyzwalacz (m) | N/A | Verb form: Wyzwolić |
| Condition | Warunek (m) | N/A | |
| Effect | Efekt (m) | N/A | |
| Bonus | Bonus (m) | N/A | |
| Penalty | Kara (f) | N/A | |
| Modifier | Modyfikator (m) | N/A | |
| Multiplier | Mnożnik (m) | N/A | |

### Item Categories

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| Starter Item | Przedmiot początkowy (m) | N/A | |
| Consumable | Przedmiot jednorazowy (m) | N/A | Literally "one-time item" |
| Relic | Relikt (m) | N/A | |
| Offensive | Ofensywny (adj) | N/A | |
| Defensive | Defensywny (adj) | N/A | |
| Utility | Użytkowy (adj) | N/A | |
| Physical | Fizyczny (adj) | 38.2% | Needs standardization |
| Magical | Magiczny (adj) | 39.4% | Needs standardization |
| Aura | Aura (f) | N/A | |
| Mask | Maska (f) | N/A | |
| Boots | Buty (m, plural) | N/A | |
| Gloves | Rękawice (f, plural) | N/A | |
| Helm | Hełm (m) | N/A | |
| Armor | Zbroja (f) | N/A | |
| Cloak | Płaszcz (m) | N/A | |
| Rod | Różdżka (f) | N/A | |
| Staff | Kostur (m) | N/A | |
| Sword | Miecz (m) | N/A | |
| Dagger | Sztylet (m) | N/A | |
| Bow | Łuk (m) | N/A | |
| Spear | Włócznia (f) | N/A | |
| Mace | Buława (f) | N/A | |
| Tier 1 | Poziom 1 (m) | N/A | |
| Tier 2 | Poziom 2 (m) | N/A | |
| Tier 3 | Poziom 3 (m) | N/A | |
| Tier 4 | Poziom 4 (m) | N/A | |
| Upgrade | Ulepszenie (n) | N/A | Verb form: Ulepszyć |
| Recipe | Przepis (m) | N/A | |
| Component | Komponent (m) | N/A | |
| Set | Zestaw (m) | N/A | |
| Limited | Limitowany (adj) | N/A | |
| Exclusive | Ekskluzywny (adj) | N/A | |

### Common Abbreviations

| English Abbreviation | Polish Translation | Consistency | Notes |
|----------------------|-------------------|-------------|-------|
| HP | PZ | N/A | "Punkty Zdrowia" (Health Points) |
| MP | PM | N/A | "Punkty Many" (Mana Points) |
| CC | KT | N/A | "Kontrola Tłumu" (Crowd Control); consider keeping "CC" (12.5% in dataset) |
| CDR | ROD | N/A | "Redukcja Odnowienia" (Cooldown Reduction) |
| DoT | OWT | N/A | "Obrażenia w Czasie" (Damage over Time) |
| AoE | OD | N/A | "Obszar Działania" (Area of Effect) |
| AS | PA | N/A | "Prędkość Ataku" (Attack Speed) |
| MS | PR | N/A | "Prędkość Ruchu" (Movement Speed) |
| AD | OF | N/A | "Obrażenia Fizyczne" (Attack Damage) |
| AP | OM | N/A | "Obrażenia Magiczne" (Ability Power) |
| KDA | ZŚA | N/A | "Zabójstwa/Śmierci/Asysty" (Kills/Deaths/Assists) |
| AA | AP | N/A | "Atak Podstawowy" (Auto Attack) |
| CD | CO | N/A | "Czas Odnowienia" (Cooldown) |
| DPS | OPS | N/A | "Obrażenia na Sekundę" (Damage Per Second) |
| HPS | LPS | N/A | "Leczenie na Sekundę" (Healing Per Second) |
| MR | OM | N/A | "Ochrona Magiczna" (Magic Resistance) |
| PR | OF | N/A | "Ochrona Fizyczna" (Physical Resistance) |
| XP | PD | N/A | "Punkty Doświadczenia" (Experience Points) |
| LVL | POZ | N/A | "Poziom" (Level) |
| ULT | US | N/A | "Umiejętność Specjalna" (Ultimate) |
| GG | GG | N/A | "Good Game" - abbreviation kept |
| AFK | AFK | N/A | "Away From Keyboard" - abbreviation kept |
| BM | BM | N/A | "Bad Manners" - abbreviation kept |
| DC | DC | N/A | "Disconnect" - abbreviation kept |

### Specialized Gaming Terms

| English Term | Polish Translation | Consistency | Notes |
|--------------|-------------------|-------------|-------|
| Gank | Gank (m) | 100.0% | Borrowed term; high consistency |
| Farm | Farm (m) | N/A | Borrowed term, verb form: Farmić |
| Last Hit | Ostatnie trafienie (n) | N/A | |
| Poke | Zaczepka (f) | N/A | Verb form: Zaczepić |
| Burst | Wybuch (m) | N/A | Sometimes borrowed as "Burst" |
| Sustain | Wytrzymałość (f) | N/A | |
| Kite | Kiting (m) | N/A | Borrowed term |
| Peel | Ochrona (f) | N/A | Verb form: Chronić |
| Rotation | Rotacja (f) | N/A | |
| Invade | Inwazja (f) | N/A | Verb form: Najeżdżać |
| Leash | Pomoc (f) | N/A | Literally "help" |
| Snowball | Kula śnieżna (f) | N/A | Sometimes used as borrowed verb: Snowballować |
| Meta | Meta (f) | N/A | Borrowed term |
| Nerf | Osłabienie (n) | N/A | Verb form: Osłabić |
| Buff (balance change) | Wzmocnienie (n) | N/A | Verb form: Wzmocnić |
| OP | OP | N/A | "Overpowered" - abbreviation kept |
| Smurf | Smurf (m) | N/A | Borrowed term |
| Carry (verb) | Nieść (v) | N/A | Literally "to carry" |
| Feed | Karmienie (n) | N/A | Verb form: Karmić |
| Throw | Rzucenie (n) | N/A | In gaming context: losing a winning position |
| Comeback | Powrót (m) | N/A | |
| Dive | Nurkowanie (n) | N/A | Verb form: Nurkować |
| Focus | Skupienie (n) | N/A | Verb form: Skupić się |
| Trade | Wymiana (f) | N/A | Verb form: Wymieniać |
| Bait | Przynęta (f) | N/A | Verb form: Zwabić |
| Juke | Zwód (m) | N/A | Verb form: Zwodzić |
| Cheese | Tani trik (m) | N/A | Gaming slang for cheap strategy |
| Tilt | Tilt (m) | N/A | Borrowed term |
| Grind | Grind (m) | N/A | Borrowed term |
| Macro | Makro (n) | N/A | Strategic gameplay |
| Micro | Mikro (n) | N/A | Mechanical gameplay |

### Standardized UI Abbreviations (for space constraints)

| Full Polish Term | Abbreviated Form | Notes |
|-------------------|------------------|-------|
| Zdrowie | Zdr. | For extremely limited UI space |
| Mana | Mn. | For extremely limited UI space |
| Czas odnowienia | Odnow. | For extremely limited UI space |
| Prędkość ataku | Pr. at. | For extremely limited UI space |
| Prędkość ruchu | Pr. ru. | For extremely limited UI space |
| Ochrona fizyczna | Ochr. fiz. | For extremely limited UI space |
| Ochrona magiczna | Ochr. mag. | For extremely limited UI space |
| Przebicie fizyczne | Przeb. fiz. | For extremely limited UI space |
| Przebicie magiczne | Przeb. mag. | For extremely limited UI space |
| Kontrola tłumu | KT | For extremely limited UI space |
| Obrażenia na sekundę | OPS | For extremely limited UI space |
| Regeneracja zdrowia | Reg. zdr. | For extremely limited UI space |
| Regeneracja many | Reg. many | For extremely limited UI space |
| Redukcja czasu odnowienia | Red. odnow. | For extremely limited UI space |
| Trafienie krytyczne | Traf. kryt. | For extremely limited UI space |
| Kradzież życia | Kr. życia | For extremely limited UI space |
| Obszar działania | Obsz. dział. | For extremely limited UI space |
| Doświadczenie | Dośw. | For extremely limited UI space |
| Poziom | Poz. | For extremely limited UI space |
| Umiejętność specjalna | Umiej. spec. | For extremely limited UI space |
