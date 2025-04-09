# SMITE 2 Translation Ruleset: Latin American Spanish (esLA)

This comprehensive document provides the complete set of rules and guidelines for translating SMITE 2 content from English to Latin American Spanish. It is designed to ensure consistency, accuracy, and cultural appropriateness across all game content.

## Table of Contents

1. [Introduction](#introduction)
2. [Special Translation Rules](#special-translation-rules)
   - [Grammatical Rules](#grammatical-rules)
   - [Formatting Rules](#formatting-rules)
   - [Terminology Rules](#terminology-rules)
   - [Cultural Adaptation Rules](#cultural-adaptation-rules)
   - [Exception Handling Rules](#exception-handling-rules)
   - [Implementation Guidelines](#implementation-guidelines)
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
5. [Validation Insights](#validation-insights)
6. [Priority Improvements](#priority-improvements)

## Introduction

Latin American Spanish (esLA) is a variant of Spanish used across Latin America, with some regional differences but generally understood throughout the region. This ruleset has been developed through extensive analysis of existing SMITE 2 translations and validated against the translation dataset to ensure it reflects current practices while standardizing inconsistencies.

The Latin American Spanish translation for SMITE 2 should aim to be:
- **Regionally neutral**: Understandable across all Latin American countries
- **Consistent**: Using standardized terminology throughout the game
- **Natural**: Sounding like native Spanish rather than a translation
- **Engaging**: Maintaining the game's tone and personality
- **Accurate**: Preserving the precise meaning of game mechanics

## Special Translation Rules

### Grammatical Rules

#### Gender Assignment

Latin American Spanish requires gender assignment (masculine or feminine) for all nouns, which affects articles, adjectives, and other related words.

- **Rule G1**: Assign gender to game terms consistently according to the glossary
  - Example: "Damage" → "el Daño" (masculine)
  - Example: "Ability" → "la Habilidad" (feminine)
  - Example: "Item" → "el Objeto" (masculine)
  - Example: "Mana" → "el Maná" (masculine)
  - **IMPORTANT**: Validation confirmed excellent gender agreement consistency (100%)

- **Rule G2**: For compound nouns, use the gender of the main noun
  - Example: "Movement Speed" → "Velocidad de Movimiento" (feminine, from "la Velocidad")
  - Example: "Attack Damage" → "Daño de Ataque" (masculine, from "el Daño")

- **Rule G3**: For English terms retained in Spanish, assign gender based on similar Spanish concepts
  - Example: "Buff" → "el Buff" (masculine, like "el beneficio")
  - Example: "Proc" → "el Proc" (masculine, like "el proceso")

#### Number Agreement

Latin American Spanish requires number agreement (singular or plural) for nouns, articles, adjectives, and verbs.

- **Rule N1**: Form plurals according to Spanish rules
  - Words ending in vowels: add -s (daño → daños)
  - Words ending in consonants: add -es (jugador → jugadores)
  - Words ending in -z: change to -ces (velocidad → velocidades)
  - Words ending in stressed vowels with accent marks: retain accent in singular, lose it in plural (acción → acciones)

- **Rule N2**: Ensure adjectives agree in number and gender with the nouns they modify
  - Example: "Physical damage" → "Daño físico" (masculine singular)
  - Example: "Physical damages" → "Daños físicos" (masculine plural)
  - Example: "Magical ability" → "Habilidad mágica" (feminine singular)
  - Example: "Magical abilities" → "Habilidades mágicas" (feminine plural)

#### Preposition Usage

- **Rule P1**: Use appropriate prepositions for relationships between terms
  - "of" relationships: use "de" (with appropriate contractions)
    - Example: "Tower of Strength" → "Torre de Fuerza"
    - Example: "Book of the Dead" → "Libro de los Muertos" (de + los)
  
  - "for" relationships: use "para"
    - Example: "Potion for healing" → "Poción para curación"
  
  - Location relationships: use "en"
    - Example: "In the jungle" → "En la jungla"
    - Example: "On the map" → "En el mapa"

- **Rule P2**: Use contractions of prepositions with articles
  - a + el = al
    - Example: "Go to the jungle" → "Ve al campamento"
  - de + el = del
    - Example: "Power of the titan" → "Poder del titán"

#### Verb Forms

- **Rule V1**: Use imperative forms for player instructions with the "tú" form
  - Example: "Press to activate" → "Presiona para activar"
  - Example: "Use this ability" → "Usa esta habilidad"
  - **IMPORTANT**: Validation showed low consistency (35.7%) in formality level; must standardize on "tú" form

- **Rule V2**: Use infinitive forms for menu items and button labels
  - Example: "Activate" → "Activar"
  - Example: "Purchase" → "Comprar"

- **Rule V3**: Use present tense for ability descriptions and passive effects
  - Example: "This ability stuns enemies" → "Esta habilidad aturde a los enemigos"
  - Example: "Deals damage over time" → "Causa daño a lo largo del tiempo"

### Formatting Rules

#### Capitalization

- **Rule CAP1**: Unlike English, do not capitalize common nouns
  - Example: "Physical Damage" → "daño físico"
  - Example: "Movement Speed" → "velocidad de movimiento"
  - **IMPORTANT**: Validation confirmed excellent capitalization consistency (99.6%)

- **Rule CAP2**: Capitalize sentence beginnings and proper nouns
  - Example: "this item increases damage" → "Este objeto aumenta el daño"
  - Example: "Mercury's staff" → "Bastón de Mercurio"

- **Rule CAP3**: Do not capitalize days of the week, months, languages, or nationalities
  - Example: "Friday" → "viernes"
  - Example: "English language" → "idioma inglés"

#### Punctuation

- **Rule P1**: Use inverted question marks and exclamation marks at the beginning of questions and exclamations
  - Example: "Ready?" → "¿Listo?"
  - Example: "Amazing!" → "¡Increíble!"

- **Rule P2**: Use comma as decimal separator
  - Example: "2.5 seconds" → "2,5 segundos"
  - **IMPORTANT**: Validation showed moderate consistency (61.7%) in punctuation; must standardize decimal separators

- **Rule P3**: Use period as thousands separator
  - Example: "1,000 gold" → "1.000 de oro"

- **Rule P4**: Use Spanish quotation marks
  - Example: "Victory" → "Victoria" or «Victoria»

#### Accents

- **Rule A1**: Always use proper accents as they can change meaning
  - Example: "daño" (damage) vs "dano" (incorrect)
  - Example: "protección" (protection) vs "proteccion" (incorrect)
  - **IMPORTANT**: Validation showed good consistency (87.7%) in accent usage; continue to emphasize proper accent usage

- **Rule A2**: Common accent marks in Spanish include:
  - Acute accent (á, é, í, ó, ú): indicates stress that deviates from the standard rules
  - Diaeresis (ü): indicates that the 'u' is pronounced in "güe" and "güi" combinations
  - Tilde (ñ): represents the palatal nasal sound

### Terminology Rules

#### Game-Specific Terms

- **Rule T1**: Use established Latin American Spanish MOBA terminology for core concepts
  - "Damage" → "Daño" (98.0% consistency - maintain this standard)
  - "Cooldown" → "Tiempo de Reutilización" (validation showed 0% consistency with "Enfriamiento"; "Reutilización" is the preferred term)
  - "Ability" → "Habilidad" (92.1% consistency - maintain this standard)
  - "Ultimate" → "Definitiva" (65.6% consistency - maintain this standard but emphasize consistency)
  - "Passive" → "Pasiva" (94.6% consistency - maintain this standard)
  - "Crowd Control" → "Control de Masas" 
  - "Health" → "Salud" (validation showed only 3.9% consistency with "Vida"; "Salud" is the preferred term)
  - "Power" → "Poder" (84.5% consistency - maintain this standard)
  - "God" → "Dios" (84.6% consistency - maintain this standard)
  - "Mana" → "Maná" (99.2% consistency - maintain this standard)

- **Rule T2**: Preserve English terms that are commonly used in the Latin American gaming community
  - "Buff" → "Buff"
  - "Nerf" → "Nerf"
  - "Skill shot" → "Skill shot"
  - "Gank" → "Gank"

- **Rule T3**: Translate role names consistently
  - "Warrior" → "Guerrero"
  - "Guardian" → "Guardián"
  - "Mage" → "Mago"
  - "Hunter" → "Cazador"
  - "Assassin" → "Asesino"

#### UI Elements

- **Rule UI1**: Keep UI element translations concise
  - Example: Abbreviate where possible without losing meaning
  - Example: Use shorter synonyms when available

- **Rule UI2**: Use consistent translations for recurring UI elements
  - "Settings" → "Configuración"
  - "Options" → "Opciones"
  - "Play" → "Jugar"
  - "Exit" → "Salir"

#### Ability Descriptions

- **Rule AD1**: Maintain a consistent structure for ability descriptions
  - Start with a brief description of what the ability does
  - Follow with damage values and effects
  - End with cooldown information

- **Rule AD2**: Use consistent terminology for damage types
  - "Physical Damage" → "Daño Físico"
  - "Magical Damage" → "Daño Mágico"

- **Rule AD3**: Use consistent terminology for crowd control effects
  - "Stun" → "Aturdimiento"
  - "Root" → "Enraizamiento"
  - "Silence" → "Silencio"
  - "Slow" → "Ralentización"

### Cultural Adaptation Rules

#### Formality Level

- **Rule F1**: Use informal "tú" form for all player instructions and tooltips
  - Example: "You gain health" → "Ganas salud"
  - Example: "Your abilities deal more damage" → "Tus habilidades causan más daño"
  - **IMPORTANT**: Validation showed low consistency (35.7%) in formality level; must standardize on "tú" form

- **Rule F2**: Use a neutral, somewhat informal tone for game narration
  - Avoid overly formal or academic language
  - Maintain professional language while being engaging

#### Regional Neutrality

- **Rule RN1**: Use neutral Latin American Spanish that is widely understood across regions
  - Avoid country-specific terms and expressions
  - Use "tú" form consistently rather than "vos" (which is common in some regions like Argentina)
  - Choose vocabulary that is recognized throughout Latin America

#### Mythology References

- **Rule M1**: Use established Spanish names for mythological figures when they exist
  - Example: "Mercury" → "Mercurio"
  - Example: "Jupiter" → "Júpiter"

- **Rule M2**: Preserve original names when no established Spanish equivalent exists
  - Example: "Ah Muzen Cab" remains "Ah Muzen Cab"

#### Humor and Wordplay

- **Rule H1**: Adapt humor and wordplay to maintain the spirit rather than literal translation
  - Focus on creating equivalent humor in Spanish rather than direct translation
  - Ensure cultural references are understandable to Latin American players

### Exception Handling Rules

#### Untranslatable Terms

- **Rule E1**: Keep untranslatable game-specific terms in English, but apply Spanish grammar rules
  - Add appropriate articles based on assigned gender
  - Apply Spanish pluralization rules when necessary
  - Example: "el buff" (singular) → "los buffs" (plural)

#### Character Limitations

- **Rule L1**: When space is limited (UI elements), prioritize clarity over completeness
  - Use abbreviations for common terms (e.g., "Velocidad de Movimiento" → "Vel. de Movimiento")
  - Omit articles when absolutely necessary, but preserve meaning
  - Create compound words to save space

#### Inconsistency Resolution

- **Rule I1**: When encountering inconsistent translations in the existing database, prefer:
  1. The most frequently used translation
  2. The translation that best follows these rules
  3. The most recent translation
  - **IMPORTANT**: Based on validation, use "Salud" for "Health" and "Tiempo de Reutilización" for "Cooldown"

### Implementation Guidelines

#### Translation Process

1. Identify the type of content being translated (UI, ability description, lore, etc.)
2. Apply relevant grammatical rules (gender, number, prepositions)
3. Use established terminology from the glossary
4. Apply formatting rules (capitalization, punctuation, accents)
5. Adapt cultural elements as needed
6. Review for consistency with other translations
7. Check for character limitations and adjust if necessary

#### Quality Assurance

- Verify gender and number agreement throughout related terms
- Ensure consistent terminology usage
- Check for proper accent usage and inverted punctuation marks
- Verify that translations fit in the UI space allocated
- Confirm that the meaning is preserved accurately

## Game-Specific Glossary

### Core Game Concepts

| English | Latin American Spanish | Gender | Notes |
|---------|------------------------|--------|-------|
| Ability | Habilidad | feminine | la Habilidad (92.1% consistency - maintain this standard) |
| Arena | Arena | feminine | la Arena |
| Assist | Asistencia | feminine | la Asistencia |
| Attack | Ataque | masculine | el Ataque (83.4% consistency - maintain this standard) |
| Basic Attack | Ataque Básico | masculine | el Ataque Básico |
| Buff | Buff | masculine | el Buff (English term retained) |
| Cooldown | Tiempo de Reutilización | masculine | el Tiempo de Reutilización (validation showed 0% consistency with "Enfriamiento"; "Reutilización" is the preferred term) |
| Crowd Control | Control de Masas | masculine | el Control de Masas; abbreviated as "CC" |
| Damage | Daño | masculine | el Daño (98.0% consistency - maintain this standard) |
| Death | Muerte | feminine | la Muerte |
| Debuff | Debuff | masculine | el Debuff (English term retained) |
| Defense | Defensa | feminine | la Defensa |
| Effect | Efecto | masculine | el Efecto |
| Experience | Experiencia | feminine | la Experiencia |
| God | Dios | masculine | el Dios (84.6% consistency - maintain this standard) |
| Gold | Oro | masculine | el Oro |
| Health | Salud | feminine | la Salud (validation showed only 3.9% consistency with "Vida"; "Salud" is the preferred term) |
| Item | Objeto | masculine | el Objeto (80.6% consistency - maintain this standard) |
| Kill | Asesinato | masculine | el Asesinato |
| Lane | Carril | masculine | el Carril |
| Level | Nivel | masculine | el Nivel |
| Mana | Maná | masculine | el Maná (99.2% consistency - maintain this standard) |
| Map | Mapa | masculine | el Mapa |
| Minion | Súbdito | masculine | el Súbdito |
| Objective | Objetivo | masculine | el Objetivo |
| Passive | Pasiva | feminine | la Pasiva (94.6% consistency - maintain this standard) |
| Power | Poder | masculine | el Poder (84.5% consistency - maintain this standard) |
| Protection | Protección | feminine | la Protección |
| Skill | Habilidad | feminine | la Habilidad |
| Speed | Velocidad | feminine | la Velocidad |
| Stat | Estadística | feminine | la Estadística |
| Ultimate | Definitiva | feminine | la Definitiva (65.6% consistency - maintain this standard but emphasize consistency) |

### Character Classes & Roles

| English | Latin American Spanish | Gender | Notes |
|---------|------------------------|--------|-------|
| Assassin | Asesino | masculine | el Asesino |
| Carry | Carry | masculine | el Carry (English term retained) |
| Guardian | Guardián | masculine | el Guardián |
| Hunter | Cazador | masculine | el Cazador |
| Jungler | Jungla | masculine | el Jungla (refers to the player, not the location) |
| Mage | Mago | masculine | el Mago |
| Mid | Medio | masculine | el Medio |
| Solo | Solo | masculine | el Solo |
| Support | Soporte | masculine | el Soporte |
| Warrior | Guerrero | masculine | el Guerrero |

### Combat Terminology

| English | Latin American Spanish | Gender | Notes |
|---------|------------------------|--------|-------|
| Area of Effect | Área de Efecto | feminine | el Área de Efecto; abbreviated as "AoE" |
| Critical | Crítico | masculine | el Crítico (as noun) / crítico (as adjective) |
| Critical Strike | Golpe Crítico | masculine | el Golpe Crítico |
| Damage over Time | Daño a lo Largo del Tiempo | masculine | el Daño a lo Largo del Tiempo; abbreviated as "DoT" |
| Healing | Curación | feminine | la Curación |
| Lifesteal | Robo de Vida | masculine | el Robo de Vida |
| Magical Damage | Daño Mágico | masculine | el Daño Mágico |
| Magical Protection | Protección Mágica | feminine | la Protección Mágica |
| Movement Speed | Velocidad de Movimiento | feminine | la Velocidad de Movimiento |
| Penetration | Penetración | feminine | la Penetración |
| Physical Damage | Daño Físico | masculine | el Daño Físico |
| Physical Protection | Protección Física | feminine | la Protección Física |
| Projectile | Proyectil | masculine | el Proyectil |
| Range | Alcance | masculine | el Alcance |
| Shield | Escudo | masculine | el Escudo |

### Crowd Control Effects

| English | Latin American Spanish | Gender | Notes |
|---------|------------------------|--------|-------|
| Banish | Destierro | masculine | el Destierro |
| Blind | Ceguera | feminine | la Ceguera |
| Cripple | Lisiadura | feminine | la Lisiadura |
| Disarm | Desarme | masculine | el Desarme |
| Fear | Miedo | masculine | el Miedo |
| Intoxicate | Intoxicación | feminine | la Intoxicación |
| Knockback | Empujón | masculine | el Empujón |
| Knockup | Lanzamiento | masculine | el Lanzamiento |
| Madness | Locura | feminine | la Locura |
| Mesmerize | Hipnosis | feminine | la Hipnosis |
| Pull | Tirón | masculine | el Tirón |
| Root | Enraizamiento | masculine | el Enraizamiento |
| Silence | Silencio | masculine | el Silencio |
| Slow | Ralentización | feminine | la Ralentización |
| Stun | Aturdimiento | masculine | el Aturdimiento |
| Taunt | Provocación | feminine | la Provocación |

### Map Elements

| English | Latin American Spanish | Gender | Notes |
|---------|------------------------|--------|-------|
| Base | Base | feminine | la Base |
| Camp | Campamento | masculine | el Campamento |
| Creep | Creep | masculine | el Creep (English term retained) |
| Fountain | Fuente | feminine | la Fuente |
| Jungle | Jungla | feminine | la Jungla (refers to the location, not the player) |
| Lane | Carril | masculine | el Carril |
| Phoenix | Fénix | masculine | el Fénix |
| Titan | Titán | masculine | el Titán |
| Tower | Torre | feminine | la Torre |

### Game Modes

| English | Latin American Spanish | Gender | Notes |
|---------|------------------------|--------|-------|
| Arena | Arena | feminine | la Arena |
| Assault | Asalto | masculine | el Asalto |
| Clash | Choque | masculine | el Choque |
| Conquest | Conquista | feminine | la Conquista |
| Joust | Justa | feminine | la Justa |
| Ranked | Clasificatoria | feminine | la Clasificatoria |
| Siege | Asedio | masculine | el Asedio |

### UI Elements

| English | Latin American Spanish | Gender | Notes |
|---------|------------------------|--------|-------|
| Back | Atrás | - | adverb |
| Cancel | Cancelar | - | verb (infinitive) |
| Confirm | Confirmar | - | verb (infinitive) |
| Exit | Salir | - | verb (infinitive) |
| Leaderboard | Tabla de Clasificación | feminine | la Tabla de Clasificación |
| Load | Cargar | - | verb (infinitive) |
| Menu | Menú | masculine | el Menú |
| Next | Siguiente | - | adjective |
| Options | Opciones | feminine (plural) | las Opciones |
| Play | Jugar | - | verb (infinitive) |
| Previous | Anterior | - | adjective |
| Profile | Perfil | masculine | el Perfil |
| Queue | Cola | feminine | la Cola |
| Save | Guardar | - | verb (infinitive) |
| Settings | Configuración | feminine | la Configuración |
| Store | Tienda | feminine | la Tienda |

### Common Abbreviations

| English | Latin American Spanish | Notes |
|---------|------------------------|-------|
| AoE | AoE | Área de Efecto (Area of Effect) |
| CC | CC | Control de Masas (Crowd Control) |
| CD | TR | Tiempo de Reutilización (Cooldown) - updated based on validation |
| DoT | DoT | Daño a lo Largo del Tiempo (Damage over Time) |
| DPS | DPS | Daño por Segundo (Damage per Second) |
| HP | PS | Puntos de Salud (Health Points) - updated based on validation |
| MP | PM | Puntos de Maná (Mana Points) |

### Standardized UI Abbreviations

| Full Latin American Spanish Term | Abbreviation | Notes |
|----------------------------------|--------------|-------|
| Velocidad de Movimiento | Vel. de Mov. | For movement speed |
| Velocidad de Ataque | Vel. de Atq. | For attack speed |
| Control de Masas | CC | For crowd control |
| Daño a lo Largo del Tiempo | DoT | For damage over time |
| Puntos de Salud | PS | For health points - updated based on validation |
| Puntos de Maná | PM | For mana points |
| Protección Física | Prot. Fís. | For physical protection |
| Protección Mágica | Prot. Mág. | For magical protection |
| Golpe Crítico | Golpe Crít. | For critical strike |
| Tiempo de Reutilización | T. Reut. | For cooldown - updated based on validation |
| Habilidad | Hab. | For ability |
| Regeneración | Regen. | For regeneration |
| Probabilidad de Crítico | Prob. Crít. | For critical chance |

### Compound Terms

| English | Latin American Spanish | Gender | Notes |
|---------|------------------------|--------|-------|
| Attack Speed | Velocidad de Ataque | feminine | la Velocidad de Ataque |
| Cooldown Reduction | Reducción de Tiempo de Reutilización | feminine | la Reducción de Tiempo de Reutilización - updated based on validation |
| Critical Chance | Probabilidad de Crítico | feminine | la Probabilidad de Crítico |
| Health Regeneration | Regeneración de Salud | feminine | la Regeneración de Salud - updated based on validation |
| Mana Regeneration | Regeneración de Maná | feminine | la Regeneración de Maná |
| Movement Speed | Velocidad de Movimiento | feminine | la Velocidad de Movimiento |
| Physical Power | Poder Físico | masculine | el Poder Físico |
| Magical Power | Poder Mágico | masculine | el Poder Mágico |

### Numerical Formatting

| English Format | Latin American Spanish Format | Example |
|----------------|--------------------------------|---------|
| 1.5 | 1,5 | Use comma as decimal separator |
| 1,000 | 1.000 | Use period as thousands separator |
| 50% | 50% | No space before percent sign |
| 10-50 | 10-50 | No spaces around hyphen in ranges |
| 5s | 5s | No space between number and unit |

## Usage Guidelines

### Gender and Articles

Always use the correct gender and corresponding articles:

- **Masculine**: el (singular), los (plural)
  - Example: "el daño" (the damage)
  - Example: "los daños" (the damages)
  
- **Feminine**: la (singular), las (plural)
  - Example: "la habilidad" (the ability)
  - Example: "las habilidades" (the abilities)

### Preposition Contractions

Use appropriate contractions of prepositions with articles:

- a + el = al
  - Example: "Ir al campamento" (Go to the camp)
- de + el = del
  - Example: "Poder del dios" (Power of the god)

### Capitalization

- Unlike English, common nouns are not capitalized in Spanish
  - Example: "daño" not "Daño"
  - Example: "velocidad de movimiento" not "Velocidad de Movimiento"

- Capitalize sentence beginnings and proper nouns
  - Example: "Este objeto aumenta el daño" (This item increases damage)
  - Example: "Mercurio es un dios" (Mercury is a god)

### Accents and Special Characters

Always use proper accents and special characters as they can change meaning:

- Acute accent (á, é, í, ó, ú): indicates stress that deviates from the standard rules
  - Example: "daño" (damage)
- Diaeresis (ü): indicates that the 'u' is pronounced in "güe" and "güi" combinations
  - Example: "pingüino" (penguin)
- Tilde (ñ): represents the palatal nasal sound
  - Example: "daño" (damage)

### Inverted Punctuation

Use inverted question marks and exclamation marks at the beginning of questions and exclamations:
- Example: "¿Listo para jugar?" (Ready to play?)
- Example: "¡Increíble!" (Incredible!)

### Formality Level

- Use informal "tú" form for player instructions
  - Example: "Ganas salud" (You gain health) - updated based on validation
  - Example: "Tus habilidades" (Your abilities)
  - **IMPORTANT**: Validation showed low consistency (35.7%) in formality level; must standardize on "tú" form

- Use infinitive forms for menu items and buttons
  - Example: "Jugar" (Play)
  - Example: "Confirmar" (Confirm)

## Validation Insights

The validation of these translation rules against the SMITE 2 dataset revealed several key insights:

### Terminology Consistency

- **High consistency (>90%)**: Ability (92.1%), Damage (98.0%), Passive (94.6%), Mana (99.2%)
- **Good consistency (80-90%)**: Attack (83.4%), God (84.6%), Item (80.6%), Power (84.5%)
- **Low consistency (<80%)**: Health (3.9% for "Vida"), Cooldown (0% for "Enfriamiento"), Ultimate (65.6%)

### Grammatical Rules

- **Gender agreement**: Excellent consistency (100%)
- **Capitalization**: Excellent consistency (99.6%)
- **Punctuation**: Moderate consistency (61.7%)
- **Accent usage**: Good consistency (87.7%)
- **Formality level**: Poor consistency (35.7%)

## Priority Improvements

Based on validation findings, these are the key areas for improvement:

1. **Standardize key terminology**: Ensure consistent translations for "Health" (use "Salud" instead of "Vida"), "Cooldown" (use "Tiempo de Reutilización" instead of "Enfriamiento"), and "Ultimate" (maintain "Definitiva")
2. **Standardize formality level**: Consistently use "tú" form for player instructions (35.7% consistency needs significant improvement)
3. **Improve punctuation consistency**: Ensure consistent use of decimal commas and thousands periods (61.7% consistency needs improvement)
4. **Maintain accent usage**: Continue applying proper accents to Spanish words (87.7% consistency is good but can be improved)
5. **Use standardized abbreviations**: Apply the abbreviations listed in this glossary for UI space constraints

By following this comprehensive ruleset, translators can ensure high-quality, consistent translations of SMITE 2 content from English to Latin American Spanish that will resonate with players across Latin America.
