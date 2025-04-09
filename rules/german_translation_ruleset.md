# SMITE 2 Translation Ruleset: German (deDE)

This document provides comprehensive guidelines for translating SMITE 2 content from English to German. It includes special translation rules and a game-specific glossary to ensure high-quality, consistent translations.

## Table of Contents

1. [Introduction](#introduction)
2. [Special Translation Rules](#special-translation-rules)
   - [Grammatical Rules](#1-grammatical-rules)
   - [Formatting Rules](#2-formatting-rules)
   - [Terminology Rules](#3-terminology-rules)
   - [Cultural Adaptation Rules](#4-cultural-adaptation-rules)
   - [Exception Handling Rules](#5-exception-handling-rules)
   - [Implementation Guidelines](#6-implementation-guidelines)
3. [Game-Specific Glossary](#game-specific-glossary)
   - [Core Game Concepts](#core-game-concepts)
   - [Character Classes & Roles](#character-classes--roles)
   - [Combat Terminology](#combat-terminology)
   - [Crowd Control Effects](#crowd-control-effects)
   - [Map Elements](#map-elements)
   - [Game Modes](#game-modes)
   - [UI Elements](#ui-elements)
   - [Common Abbreviations](#common-abbreviations)
   - [Standardized UI Abbreviations](#standardized-ui-abbreviations)
   - [Compound Terms](#compound-terms)
   - [Numerical Formatting](#numerical-formatting)
4. [Usage Guidelines](#usage-guidelines)
   - [Gender and Articles](#gender-and-articles)
   - [Capitalization](#capitalization)
   - [Compound Words](#compound-words)
   - [Formality Level](#formality-level)
5. [Priority Improvements](#priority-improvements-based-on-validation)

## Introduction

This ruleset has been developed through comprehensive analysis of the SMITE 2 translation database, focusing on German (deDE) translations. The rules and glossary have been validated against the dataset and refined to address inconsistencies and improve translation quality.

German translations for SMITE 2 must account for several key linguistic features:
- Three grammatical genders (masculine, feminine, neuter)
- Four-case system (nominative, accusative, dative, genitive)
- Compound word formation
- Capitalization of all nouns
- Formal vs. informal address

The following rules and glossary provide a framework for creating consistent, high-quality German translations that maintain the spirit and meaning of the original English content while adhering to German linguistic conventions.

## Special Translation Rules

### 1. Grammatical Rules

#### 1.1 Gender Assignment

German requires gender assignment (masculine, feminine, or neuter) for all nouns, which affects articles, adjectives, and other related words.

- **Rule G1**: Assign gender to game terms consistently according to the glossary
  - Example: "Damage" → "der Schaden" (masculine)
  - Example: "Ability" → "die Fähigkeit" (feminine)
  - Example: "Item" → "der Gegenstand" (masculine)
  - Example: "Mana" → "das Mana" (neuter)
  - **IMPORTANT**: Validation showed inconsistency in gender assignment for several terms; strict adherence to the glossary is essential

- **Rule G2**: For compound nouns, use the gender of the last component
  - Example: "Movement Speed" → "die Bewegungsgeschwindigkeit" (feminine, from "die Geschwindigkeit")
  - Example: "Attack Damage" → "der Angriffsschaden" (masculine, from "der Schaden")

- **Rule G3**: For English terms retained in German, assign gender based on similar German concepts
  - Example: "Buff" → "der Buff" (masculine, like "der Vorteil")
  - Example: "Proc" → "der Proc" (masculine, like "der Effekt")

#### 1.2 Case System

German uses a four-case system (nominative, accusative, dative, genitive) that affects articles, adjectives, and sometimes nouns.

- **Rule C1**: Use nominative case for subjects and menu items
  - Example: "The ability deals damage" → "Die Fähigkeit verursacht Schaden"

- **Rule C2**: Use accusative case for direct objects
  - Example: "Increases your power" → "Erhöht deine Kraft"

- **Rule C3**: Use dative case for indirect objects and after certain prepositions
  - Example: "Deals damage to enemies" → "Fügt Feinden Schaden zu"
  - Example: "With this item" → "Mit diesem Gegenstand"

- **Rule C4**: Use genitive case for possession
  - Example: "The god's ability" → "Die Fähigkeit des Gottes"

#### 1.3 Compound Word Formation

German frequently forms compound words by combining multiple words into one.

- **Rule CW1**: Create compound nouns for multi-word English terms
  - Example: "Attack Speed" → "Angriffsgeschwindigkeit"
  - Example: "Cooldown Reduction" → "Abklingzeitverringerung"
  - **IMPORTANT**: Validation showed some inconsistency in compound word formation; standardize approach

- **Rule CW2**: Use hyphens for clarity in longer compounds (more than three components)
  - Example: "Physical Protection Reduction" → "Physische-Schutz-Verringerung"

- **Rule CW3**: When compound words become too long for UI, use standardized abbreviations
  - Example: "Bewegungsgeschwindigkeit" → "Beweg.-Geschw."
  - **IMPORTANT**: Develop and maintain a list of standard abbreviations for UI space constraints

#### 1.4 Verb Forms

- **Rule V1**: Use informal "du" form for player instructions
  - Example: "Press to activate" → "Drücke zum Aktivieren"
  - Example: "You gain health" → "Du erhältst Gesundheit"
  - **IMPORTANT**: Validation confirmed consistent use of informal "du" form; maintain this standard

- **Rule V2**: Use infinitive forms for menu items and button labels
  - Example: "Activate" → "Aktivieren"
  - Example: "Purchase" → "Kaufen"

- **Rule V3**: Use present tense for ability descriptions and passive effects
  - Example: "This ability stuns enemies" → "Diese Fähigkeit betäubt Feinde"

- **Rule V4**: Use proper separable verb forms when applicable
  - Example: "Pick up" → "Aufheben" (separates to "hebt auf" in conjugated form)

### 2. Formatting Rules

#### 2.1 Capitalization

- **Rule CAP1**: Capitalize all nouns, regardless of position in the sentence
  - Example: "physical damage" → "physischer Schaden"
  - Example: "movement speed" → "Bewegungsgeschwindigkeit"
  - **IMPORTANT**: Validation confirmed consistent capitalization of nouns; maintain this standard

- **Rule CAP2**: Capitalize sentence beginnings and proper nouns
  - Example: "this item increases damage" → "Dieser Gegenstand erhöht Schaden"

- **Rule CAP3**: Do not capitalize "du" (informal you) unless at the beginning of a sentence
  - Example: "You gain health" → "Du erhältst Gesundheit"
  - Example: "When you attack" → "Wenn du angreifst"

#### 2.2 Punctuation

- **Rule P1**: Use German quotation marks
  - Example: "Victory" → „Sieg"
  - **IMPORTANT**: Validation confirmed consistent use of German quotation marks; maintain this standard

- **Rule P2**: Use comma before subordinate clauses
  - Example: "Attack, when the enemy is near" → "Greife an, wenn der Feind in der Nähe ist"

- **Rule P3**: Use commas to separate items in a list (no Oxford comma)
  - Example: "Strength, agility and intelligence" → "Stärke, Beweglichkeit und Intelligenz"

#### 2.3 Number and Unit Formatting

- **Rule N1**: Use comma as decimal separator
  - Example: "2.5 seconds" → "2,5 Sekunden"
  - **IMPORTANT**: Validation confirmed consistent use of comma as decimal separator; maintain this standard

- **Rule N2**: Use period as thousands separator
  - Example: "1,000 gold" → "1.000 Gold"

- **Rule N3**: Add a space between a number and its unit
  - Example: "5s cooldown" → "5 s Abklingzeit"
  - Example: "50% increase" → "50 % Erhöhung"
  - **IMPORTANT**: Validation confirmed consistent spacing between numbers and units; maintain this standard

### 3. Terminology Rules

#### 3.1 Game-Specific Terms

- **Rule T1**: Use established German MOBA terminology for core concepts
  - "Damage" → "Schaden" (validation showed 50% consistency - must be standardized)
  - "Cooldown" → "Abklingzeit"
  - "Ability" → "Fähigkeit" (validation showed 0% consistency - must be standardized)
  - "Ultimate" → "Ultimate"
  - "Passive" → "Passiv" (validation showed 100% consistency - maintain this standard)
  - "Crowd Control" → "Kontrolleffekt"
  - "Health" → "Gesundheit" (validation showed 0% consistency - must be standardized)
  - "Power" → "Kraft" (validation showed 0% consistency - must be standardized)
  - "God" → "Gott" (validation showed 55.6% consistency - must be standardized)
  - "Mana" → "Mana" (validation showed 100% consistency - maintain this standard)
  - "Attack" → "Angriff" (validation showed 0% consistency - must be standardized)
  - "Item" → "Gegenstand" (validation showed 50% consistency - must be standardized)

- **Rule T2**: Preserve English terms that are commonly used in the German gaming community
  - "Buff" → "Buff"
  - "Nerf" → "Nerf"
  - "Skill shot" → "Skillshot"
  - "Gank" → "Gank"

- **Rule T3**: Translate role names consistently
  - "Warrior" → "Krieger"
  - "Guardian" → "Wächter"
  - "Mage" → "Magier"
  - "Hunter" → "Jäger"
  - "Assassin" → "Assassine"

#### 3.2 UI Elements

- **Rule UI1**: Keep UI element translations concise, ideally no more than 30% longer than English
  - Example: Abbreviate where possible without losing meaning
  - Example: Use shorter synonyms when available
  - **IMPORTANT**: Develop standardized abbreviations for common UI terms

- **Rule UI2**: Use consistent translations for recurring UI elements
  - "Settings" → "Einstellungen"
  - "Options" → "Optionen"
  - "Play" → "Spielen"
  - "Exit" → "Beenden"

#### 3.3 Ability Descriptions

- **Rule AD1**: Maintain a consistent structure for ability descriptions
  - Start with a brief description of what the ability does
  - Follow with damage values and effects
  - End with cooldown information

- **Rule AD2**: Use consistent terminology for damage types
  - "Physical Damage" → "Physischer Schaden"
  - "Magical Damage" → "Magischer Schaden"

- **Rule AD3**: Use consistent terminology for crowd control effects
  - "Stun" → "Betäubung"
  - "Root" → "Verwurzelung"
  - "Silence" → "Stille"
  - "Slow" → "Verlangsamung"

### 4. Cultural Adaptation Rules

#### 4.1 Formality Level

- **Rule F1**: Use informal "du" form for all player instructions and tooltips
  - Example: "You gain health" → "Du erhältst Gesundheit"
  - Example: "Your abilities deal more damage" → "Deine Fähigkeiten verursachen mehr Schaden"
  - **IMPORTANT**: Validation confirmed consistent use of informal "du" form; maintain this standard

- **Rule F2**: Use a neutral, somewhat informal tone for game narration
  - Avoid overly formal or academic language
  - Maintain professional language while being engaging

#### 4.2 Mythology References

- **Rule M1**: Use established German names for mythological figures when they exist
  - Example: "Mercury" → "Merkur"
  - Example: "Jupiter" → "Jupiter" (unchanged)
  - **IMPORTANT**: Validation confirmed consistent handling of mythology references; maintain this standard

- **Rule M2**: Preserve original names when no established German equivalent exists
  - Example: "Ah Muzen Cab" remains "Ah Muzen Cab"

#### 4.3 Humor and Wordplay

- **Rule H1**: Adapt humor and wordplay to maintain the spirit rather than literal translation
  - Focus on creating equivalent humor in German rather than direct translation
  - Ensure cultural references are understandable to German players

### 5. Exception Handling Rules

#### 5.1 Untranslatable Terms

- **Rule E1**: Keep untranslatable game-specific terms in English, but apply German grammar rules
  - Add appropriate articles based on assigned gender
  - Apply German capitalization rules (capitalize as nouns)
  - Apply German pluralization rules when necessary

#### 5.2 Character Limitations

- **Rule L1**: When space is limited (UI elements), prioritize clarity over completeness
  - Use standardized abbreviations for common terms (e.g., "Abklingzeit" → "AZ")
  - Create compound words to save space
  - Omit articles when absolutely necessary, but preserve meaning
  - **IMPORTANT**: Validation highlighted the need for standardized abbreviations; develop and maintain a comprehensive list

#### 5.3 Inconsistency Resolution

- **Rule I1**: When encountering inconsistent translations in the existing database, prefer:
  1. The most frequently used translation
  2. The translation that best follows these rules
  3. The most recent translation
  - **IMPORTANT**: Validation identified several inconsistently translated terms; apply this rule to standardize terminology

### 6. Implementation Guidelines

#### 6.1 Translation Process

1. Identify the type of content being translated (UI, ability description, lore, etc.)
2. Apply relevant grammatical rules (gender, case, compound words)
3. Use established terminology from the glossary
4. Apply formatting rules (capitalization, punctuation, numbers)
5. Adapt cultural elements as needed
6. Review for consistency with other translations
7. Check for character limitations and adjust if necessary

#### 6.2 Quality Assurance

- Verify gender and case consistency throughout related terms
- Ensure consistent terminology usage
- Check capitalization of all nouns
- Verify that translations fit in the UI space allocated
- Confirm that the meaning is preserved accurately

#### 6.3 Priority Improvements Based on Validation

1. **Standardize key terminology**: Ensure consistent translations for "Ability" (Fähigkeit), "Attack" (Angriff), "Damage" (Schaden), "Health" (Gesundheit), "Power" (Kraft), "God" (Gott), and "Item" (Gegenstand)
2. **Maintain compound word consistency**: Standardize how compound words are formed and abbreviated
3. **Ensure gender consistency**: Standardize gender assignment for game-specific terms
4. **Develop UI abbreviations**: Create a standardized list of abbreviations for UI space constraints

## Game-Specific Glossary

### Core Game Concepts

| English | German | Gender | Notes |
|---------|--------|--------|-------|
| Ability | Fähigkeit | feminine | die Fähigkeit; **IMPORTANT**: Validation showed 0% consistency - must be standardized |
| Arena | Arena | feminine | die Arena |
| Assist | Unterstützung | feminine | die Unterstützung |
| Attack | Angriff | masculine | der Angriff; **IMPORTANT**: Validation showed 0% consistency - must be standardized |
| Basic Attack | Grundangriff | masculine | der Grundangriff |
| Buff | Buff | masculine | der Buff (English term retained) |
| Cooldown | Abklingzeit | feminine | die Abklingzeit |
| Crowd Control | Kontrolleffekt | masculine | der Kontrolleffekt |
| Damage | Schaden | masculine | der Schaden; **IMPORTANT**: Validation showed 50% consistency - must be standardized |
| Death | Tod | masculine | der Tod |
| Debuff | Debuff | masculine | der Debuff (English term retained) |
| Defense | Verteidigung | feminine | die Verteidigung |
| Effect | Effekt | masculine | der Effekt |
| Experience | Erfahrung | feminine | die Erfahrung |
| God | Gott | masculine | der Gott; **IMPORTANT**: Validation showed 55.6% consistency - must be standardized |
| Gold | Gold | neuter | das Gold |
| Health | Gesundheit | feminine | die Gesundheit; **IMPORTANT**: Validation showed 0% consistency - must be standardized |
| Item | Gegenstand | masculine | der Gegenstand; **IMPORTANT**: Validation showed 50% consistency - must be standardized |
| Kill | Tötung | feminine | die Tötung |
| Lane | Weg | masculine | der Weg |
| Level | Stufe | feminine | die Stufe |
| Mana | Mana | neuter | das Mana; **IMPORTANT**: Validation showed 100% consistency - maintain this standard |
| Map | Karte | feminine | die Karte |
| Minion | Scherge | masculine | der Scherge |
| Objective | Ziel | neuter | das Ziel |
| Passive | Passiv | neuter | das Passiv; **IMPORTANT**: Validation showed 100% consistency - maintain this standard |
| Power | Kraft | feminine | die Kraft; **IMPORTANT**: Validation showed 0% consistency - must be standardized |
| Protection | Schutz | masculine | der Schutz |
| Skill | Fertigkeit | feminine | die Fertigkeit |
| Speed | Geschwindigkeit | feminine | die Geschwindigkeit |
| Stat | Wert | masculine | der Wert |
| Ultimate | Ultimate | feminine | die Ultimate |

### Character Classes & Roles

| English | German | Gender | Notes |
|---------|--------|--------|-------|
| Assassin | Assassine | feminine | die Assassine |
| Carry | Carry | masculine | der Carry (English term retained) |
| Guardian | Wächter | masculine | der Wächter |
| Hunter | Jäger | masculine | der Jäger |
| Jungler | Jungler | masculine | der Jungler (English term retained) |
| Mage | Magier | masculine | der Magier |
| Mid | Mid | neuter | das Mid (English term retained) |
| Solo | Solo | neuter | das Solo (English term retained) |
| Support | Support | masculine | der Support (English term retained) |
| Warrior | Krieger | masculine | der Krieger |

### Combat Terminology

| English | German | Gender | Notes |
|---------|--------|--------|-------|
| Area of Effect | Flächeneffekt | masculine | der Flächeneffekt; abbreviated as "AoE" |
| Critical | Kritisch | - | adjective |
| Critical Strike | Kritischer Treffer | masculine | der Kritische Treffer |
| Damage over Time | Schaden über Zeit | masculine | der Schaden über Zeit; abbreviated as "DoT" |
| Healing | Heilung | feminine | die Heilung |
| Lifesteal | Lebensraub | masculine | der Lebensraub |
| Magical Damage | Magischer Schaden | masculine | der Magische Schaden |
| Magical Protection | Magischer Schutz | masculine | der Magische Schutz |
| Movement Speed | Bewegungsgeschwindigkeit | feminine | die Bewegungsgeschwindigkeit |
| Penetration | Durchdringung | feminine | die Durchdringung |
| Physical Damage | Physischer Schaden | masculine | der Physische Schaden |
| Physical Protection | Physischer Schutz | masculine | der Physische Schutz |
| Projectile | Projektil | neuter | das Projektil |
| Range | Reichweite | feminine | die Reichweite |
| Shield | Schild | neuter | das Schild |

### Crowd Control Effects

| English | German | Gender | Notes |
|---------|--------|--------|-------|
| Banish | Verbannung | feminine | die Verbannung |
| Blind | Blendung | feminine | die Blendung |
| Cripple | Verkrüppelung | feminine | die Verkrüppelung |
| Disarm | Entwaffnung | feminine | die Entwaffnung |
| Fear | Furcht | feminine | die Furcht |
| Intoxicate | Trunkenheit | feminine | die Trunkenheit |
| Knockback | Rückstoß | masculine | der Rückstoß |
| Knockup | Hochschlag | masculine | der Hochschlag |
| Madness | Wahnsinn | masculine | der Wahnsinn |
| Mesmerize | Hypnose | feminine | die Hypnose |
| Pull | Anziehung | feminine | die Anziehung |
| Root | Verwurzelung | feminine | die Verwurzelung |
| Silence | Stille | feminine | die Stille |
| Slow | Verlangsamung | feminine | die Verlangsamung |
| Stun | Betäubung | feminine | die Betäubung |
| Taunt | Verspottung | feminine | die Verspottung |

### Map Elements

| English | German | Gender | Notes |
|---------|--------|--------|-------|
| Base | Basis | feminine | die Basis |
| Camp | Lager | neuter | das Lager |
| Creep | Creep | masculine | der Creep (English term retained) |
| Fountain | Brunnen | masculine | der Brunnen |
| Jungle | Dschungel | masculine | der Dschungel |
| Lane | Weg | masculine | der Weg |
| Phoenix | Phönix | masculine | der Phönix |
| Titan | Titan | masculine | der Titan |
| Tower | Turm | masculine | der Turm |

### Game Modes

| English | German | Gender | Notes |
|---------|--------|--------|-------|
| Arena | Arena | feminine | die Arena |
| Assault | Ansturm | masculine | der Ansturm |
| Clash | Zusammenstoß | masculine | der Zusammenstoß |
| Conquest | Eroberung | feminine | die Eroberung |
| Joust | Tjost | masculine | der Tjost |
| Ranked | Gewertet | - | adjective |
| Siege | Belagerung | feminine | die Belagerung |

### UI Elements

| English | German | Gender | Notes |
|---------|--------|--------|-------|
| Back | Zurück | - | adverb |
| Cancel | Abbrechen | - | verb (infinitive) |
| Confirm | Bestätigen | - | verb (infinitive) |
| Exit | Beenden | - | verb (infinitive) |
| Leaderboard | Bestenliste | feminine | die Bestenliste |
| Load | Laden | - | verb (infinitive) |
| Menu | Menü | neuter | das Menü |
| Next | Weiter | - | adverb |
| Options | Optionen | feminine (plural) | die Optionen |
| Play | Spielen | - | verb (infinitive) |
| Previous | Zurück | - | adverb |
| Profile | Profil | neuter | das Profil |
| Queue | Warteschlange | feminine | die Warteschlange |
| Save | Speichern | - | verb (infinitive) |
| Settings | Einstellungen | feminine (plural) | die Einstellungen |
| Store | Shop | masculine | der Shop |

### Common Abbreviations

| English | German | Notes |
|---------|--------|-------|
| AoE | AoE | Flächeneffekt (Area of Effect) |
| CC | CC | Kontrolleffekt (Crowd Control) |
| CD | AZ | Abklingzeit (Cooldown) |
| DoT | DoT | Schaden über Zeit (Damage over Time) |
| DPS | DPS | Schaden pro Sekunde (Damage per Second) |
| HP | LP | Lebenspunkte (Health Points) |
| MP | MP | Manapunkte (Mana Points) |

### Standardized UI Abbreviations

| Full German Term | Abbreviation | Notes |
|------------------|--------------|-------|
| Abklingzeit | AZ | For cooldown |
| Bewegungsgeschwindigkeit | Beweg.-Geschw. | For movement speed |
| Angriffsgeschwindigkeit | Ang.-Geschw. | For attack speed |
| Gesundheit | Ges. | For health |
| Gesundheitsregeneration | Ges.-Reg. | For health regeneration |
| Manaregeneration | Mana-Reg. | For mana regeneration |
| Physischer Schaden | Phys. Schaden | For physical damage |
| Magischer Schaden | Mag. Schaden | For magical damage |
| Physischer Schutz | Phys. Schutz | For physical protection |
| Magischer Schutz | Mag. Schutz | For magical protection |
| Kritische Trefferchance | Krit. Chance | For critical chance |

### Compound Terms

| English | German | Gender | Notes |
|---------|--------|--------|-------|
| Attack Speed | Angriffsgeschwindigkeit | feminine | die Angriffsgeschwindigkeit |
| Cooldown Reduction | Abklingzeitverringerung | feminine | die Abklingzeitverringerung |
| Critical Chance | Kritische Trefferchance | feminine | die Kritische Trefferchance |
| Health Regeneration | Gesundheitsregeneration | feminine | die Gesundheitsregeneration |
| Mana Regeneration | Manaregeneration | feminine | die Manaregeneration |
| Movement Speed | Bewegungsgeschwindigkeit | feminine | die Bewegungsgeschwindigkeit |
| Physical Power | Physische Kraft | feminine | die Physische Kraft |
| Magical Power | Magische Kraft | feminine | die Magische Kraft |

### Numerical Formatting

| English Format | German Format | Example |
|----------------|---------------|---------|
| 1.5 | 1,5 | Use comma as decimal separator |
| 1,000 | 1.000 | Use period as thousands separator |
| 50% | 50 % | Add space before percent sign |
| 10-50 | 10-50 | No spaces around hyphen in ranges |
| 5s | 5 s | Add space between number and unit |

## Usage Guidelines

### Gender and Articles

Always use the correct gender and corresponding articles:

- **Masculine**: der (nominative), den (accusative), dem (dative), des (genitive)
  - Example: "der Schaden" (the damage)
  - Example: "den Schaden erhöhen" (increase the damage)
  
- **Feminine**: die (nominative/accusative), der (dative/genitive)
  - Example: "die Fähigkeit" (the ability)
  - Example: "mit der Fähigkeit" (with the ability)
  
- **Neuter**: das (nominative/accusative), dem (dative), des (genitive)
  - Example: "das Mana" (the mana)
  - Example: "Verbrauch des Manas" (consumption of the mana)

### Capitalization

- All nouns must be capitalized in German
  - Example: "Schaden" not "schaden"
  - Example: "Bewegungsgeschwindigkeit" not "bewegungsgeschwindigkeit"

- Adjectives derived from nouns are not capitalized
  - Example: "physischer Schaden" (physical damage)

### Compound Words

German frequently forms compound words from multiple English words:

- Join words without spaces
  - Example: "Movement Speed" → "Bewegungsgeschwindigkeit"
  - Example: "Attack Speed" → "Angriffsgeschwindigkeit"

- For very long compounds, consider using hyphens for clarity
  - Example: "Physical Protection Reduction" → "Physische-Schutz-Verringerung"

### Formality Level

- Use informal "du" form for player instructions
  - Example: "Du erhältst Gesundheit" (You gain health)
  - Example: "Deine Fähigkeiten" (Your abilities)

- Use infinitive forms for menu items and buttons
  - Example: "Spielen" (Play)
  - Example: "Bestätigen" (Confirm)

## Priority Improvements Based on Validation

1. **Standardize key terminology**: Ensure consistent translations for "Ability" (Fähigkeit), "Attack" (Angriff), "Damage" (Schaden), "Health" (Gesundheit), "Power" (Kraft), "God" (Gott), and "Item" (Gegenstand)
2. **Maintain compound word consistency**: Standardize how compound words are formed and abbreviated
3. **Ensure gender consistency**: Standardize gender assignment for game-specific terms
4. **Use standardized UI abbreviations**: Apply the abbreviations listed in this glossary consistently
