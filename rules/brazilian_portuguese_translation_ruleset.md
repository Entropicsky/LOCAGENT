# SMITE 2 Translation Ruleset: Brazilian Portuguese (ptBR)

This document provides comprehensive guidelines for translating SMITE 2 content from English to Brazilian Portuguese. It includes special translation rules and a game-specific glossary to ensure high-quality, consistent translations.

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
   - [Preposition Contractions](#preposition-contractions)
   - [Capitalization](#capitalization)
   - [Accents](#accents)
   - [Formality Level](#formality-level)
5. [Priority Improvements](#priority-improvements-based-on-validation)

## Introduction

This ruleset has been developed through comprehensive analysis of the SMITE 2 translation database, focusing on Brazilian Portuguese (ptBR) translations. The rules and glossary have been validated against the dataset and refined to address inconsistencies and improve translation quality.

Brazilian Portuguese translations for SMITE 2 must account for several key linguistic features:
- Two grammatical genders (masculine and feminine)
- Complex pluralization rules
- Extensive use of preposition contractions
- Accent marks that can change meaning
- Different capitalization rules from English
- Formal vs. informal address

The following rules and glossary provide a framework for creating consistent, high-quality Brazilian Portuguese translations that maintain the spirit and meaning of the original English content while adhering to Brazilian Portuguese linguistic conventions.

## Special Translation Rules

### 1. Grammatical Rules

#### 1.1 Gender Assignment

Brazilian Portuguese requires gender assignment (masculine or feminine) for all nouns, which affects articles, adjectives, and other related words.

- **Rule G1**: Assign gender to game terms consistently according to the glossary
  - Example: "Damage" → "o Dano" (masculine)
  - Example: "Ability" → "a Habilidade" (feminine)
  - Example: "Item" → "o Item" (masculine)
  - Example: "Mana" → "a Mana" (feminine)
  - **IMPORTANT**: Validation showed inconsistency in gender assignment for several terms; strict adherence to the glossary is essential

- **Rule G2**: For compound nouns, use the gender of the main noun
  - Example: "Movement Speed" → "Velocidade de Movimento" (feminine, from "a Velocidade")
  - Example: "Attack Damage" → "Dano de Ataque" (masculine, from "o Dano")

- **Rule G3**: For English terms retained in Portuguese, assign gender based on similar Portuguese concepts
  - Example: "Buff" → "o Buff" (masculine, like "o benefício")
  - Example: "Proc" → "o Proc" (masculine, like "o processo")

#### 1.2 Number Agreement

Brazilian Portuguese requires number agreement (singular or plural) for nouns, articles, adjectives, and verbs.

- **Rule N1**: Form plurals according to Portuguese rules
  - Words ending in vowels: add -s (dano → danos)
  - Words ending in -r, -z: add -es (jogador → jogadores)
  - Words ending in -m: change to -ns (item → itens)
  - Words ending in -ão: typically change to -ões or -ães (proteção → proteções)

- **Rule N2**: Ensure adjectives agree in number with the nouns they modify
  - Example: "Physical damage" → "Dano físico" (singular)
  - Example: "Physical damages" → "Danos físicos" (plural)

#### 1.3 Preposition Usage

- **Rule P1**: Use appropriate prepositions for relationships between terms
  - "of" relationships: use "de" (with appropriate contractions)
    - Example: "Tower of Strength" → "Torre de Força"
    - Example: "Book of the Dead" → "Livro dos Mortos" (de + os = dos)
  
  - "for" relationships: use "para"
    - Example: "Potion for healing" → "Poção para cura"
  
  - Location relationships: use "em" (with appropriate contractions)
    - Example: "In the jungle" → "Na selva" (em + a = na)
    - Example: "On the map" → "No mapa" (em + o = no)

- **Rule P2**: Use contractions of prepositions with articles
  - de + o = do
  - de + a = da
  - em + o = no
  - em + a = na
  - a + o = ao
  - a + a = à

#### 1.4 Verb Forms

- **Rule V1**: Use imperative forms for player instructions
  - Example: "Press to activate" → "Pressione para ativar"
  - Example: "Use this ability" → "Use esta habilidade"

- **Rule V2**: Use infinitive forms for menu items and button labels
  - Example: "Activate" → "Ativar"
  - Example: "Purchase" → "Comprar"

- **Rule V3**: Use present tense for ability descriptions and passive effects
  - Example: "This ability stuns enemies" → "Esta habilidade atordoa inimigos"
  - Example: "Deals damage over time" → "Causa dano ao longo do tempo"

### 2. Formatting Rules

#### 2.1 Capitalization

- **Rule CAP1**: Unlike English, do not capitalize common nouns
  - Example: "Physical Damage" → "dano físico"
  - Example: "Movement Speed" → "velocidade de movimento"
  - **IMPORTANT**: Validation confirmed consistent adherence to this rule

- **Rule CAP2**: Capitalize sentence beginnings and proper nouns
  - Example: "this item increases damage" → "Este item aumenta o dano"
  - Example: "Mercury's staff" → "Cajado de Mercúrio"

- **Rule CAP3**: Do not capitalize days of the week, months, languages, or nationalities
  - Example: "Friday" → "sexta-feira"
  - Example: "English language" → "idioma inglês"

#### 2.2 Punctuation

- **Rule P1**: Use comma as decimal separator
  - Example: "2.5 seconds" → "2,5 segundos"
  - **IMPORTANT**: Validation confirmed consistent adherence to this rule

- **Rule P2**: Use period as thousands separator
  - Example: "1,000 gold" → "1.000 de ouro"
  - **IMPORTANT**: Validation confirmed consistent adherence to this rule

- **Rule P3**: Use Portuguese quotation marks
  - Example: "Victory" → "Vitória" or «Vitória»

#### 2.3 Accents

- **Rule A1**: Always use proper accents as they can change meaning
  - Example: "saúde" (health) vs "saude" (incorrect)
  - Example: "proteção" (protection) vs "protecao" (incorrect)
  - **IMPORTANT**: Validation confirmed consistent accent usage

- **Rule A2**: Common accent marks in Portuguese include:
  - Acute accent (á, é, í, ó, ú): indicates stress
  - Circumflex accent (â, ê, ô): indicates closed vowels
  - Tilde (ã, õ): indicates nasalization
  - Cedilla (ç): indicates 's' sound before a, o, u
  - Grave accent (à): mainly used in contractions

### 3. Terminology Rules

#### 3.1 Game-Specific Terms

- **Rule T1**: Use established Brazilian Portuguese MOBA terminology for core concepts
  - "Damage" → "Dano" - **IMPORTANT**: Validation showed 50% consistency - must be standardized
  - "Cooldown" → "Recarga"
  - "Ability" → "Habilidade" - **IMPORTANT**: Validation showed 50% consistency - must be standardized
  - "Ultimate" → "Ultimate" (often kept in English)
  - "Passive" → "Passiva" - **IMPORTANT**: Validation showed 100% consistency - maintain this standard
  - "Crowd Control" → "Controle de Grupo"
  - "Health" → "Vida" - **IMPORTANT**: Validation showed 0% consistency - must be standardized
  - "Power" → "Poder" - **IMPORTANT**: Validation showed 0% consistency - must be standardized
  - "God" → "Deus" - **IMPORTANT**: Validation showed 33.3% consistency - must be standardized
  - "Mana" → "Mana" - **IMPORTANT**: Validation showed 100% consistency - maintain this standard

- **Rule T2**: Preserve English terms that are commonly used in the Brazilian gaming community
  - "Buff" → "Buff"
  - "Nerf" → "Nerf"
  - "Skill shot" → "Skill shot"
  - "Gank" → "Gank"

- **Rule T3**: Translate role names consistently
  - "Warrior" → "Guerreiro"
  - "Guardian" → "Guardião"
  - "Mage" → "Mago"
  - "Hunter" → "Caçador"
  - "Assassin" → "Assassino"

#### 3.2 UI Elements

- **Rule UI1**: Keep UI element translations concise
  - Example: Abbreviate where possible without losing meaning
  - Example: Use shorter synonyms when available
  - **IMPORTANT**: Validation highlighted the need for standardized abbreviations for UI space constraints

- **Rule UI2**: Use consistent translations for recurring UI elements
  - "Settings" → "Configurações"
  - "Options" → "Opções"
  - "Play" → "Jogar"
  - "Exit" → "Sair"

#### 3.3 Ability Descriptions

- **Rule AD1**: Maintain a consistent structure for ability descriptions
  - Start with a brief description of what the ability does
  - Follow with damage values and effects
  - End with cooldown information

- **Rule AD2**: Use consistent terminology for damage types
  - "Physical Damage" → "Dano Físico"
  - "Magical Damage" → "Dano Mágico"

- **Rule AD3**: Use consistent terminology for crowd control effects
  - "Stun" → "Atordoamento"
  - "Root" → "Enraizamento"
  - "Silence" → "Silêncio"
  - "Slow" → "Lentidão"

### 4. Cultural Adaptation Rules

#### 4.1 Formality Level

- **Rule F1**: Use informal "você" form for all player instructions and tooltips
  - Example: "You gain health" → "Você ganha vida"
  - Example: "Your abilities deal more damage" → "Suas habilidades causam mais dano"
  - **IMPORTANT**: Validation confirmed consistent use of "você" form

- **Rule F2**: Use a neutral, somewhat informal tone for game narration
  - Avoid overly formal or academic language
  - Maintain professional language while being engaging

#### 4.2 Mythology References

- **Rule M1**: Use established Portuguese names for mythological figures when they exist
  - Example: "Mercury" → "Mercúrio"
  - Example: "Jupiter" → "Júpiter"
  - **IMPORTANT**: Validation confirmed consistent handling of mythology references

- **Rule M2**: Preserve original names when no established Portuguese equivalent exists
  - Example: "Ah Muzen Cab" remains "Ah Muzen Cab"

#### 4.3 Humor and Wordplay

- **Rule H1**: Adapt humor and wordplay to maintain the spirit rather than literal translation
  - Focus on creating equivalent humor in Portuguese rather than direct translation
  - Ensure cultural references are understandable to Brazilian players

### 5. Exception Handling Rules

#### 5.1 Untranslatable Terms

- **Rule E1**: Keep untranslatable game-specific terms in English, but apply Portuguese grammar rules
  - Add appropriate articles based on assigned gender
  - Apply Portuguese pluralization rules when necessary
  - Example: "o buff" (singular) → "os buffs" (plural)

#### 5.2 Character Limitations

- **Rule L1**: When space is limited (UI elements), prioritize clarity over completeness
  - Use standardized abbreviations for common terms (e.g., "Velocidade de Movimento" → "Vel. de Mov.")
  - Omit articles when absolutely necessary, but preserve meaning
  - Create compound words to save space
  - **IMPORTANT**: Validation highlighted the need for standardized abbreviations; use the abbreviations listed in the glossary

#### 5.3 Inconsistency Resolution

- **Rule I1**: When encountering inconsistent translations in the existing database, prefer:
  1. The most frequently used translation
  2. The translation that best follows these rules
  3. The most recent translation
  - **IMPORTANT**: Validation identified several inconsistently translated terms; apply this rule to standardize terminology

### 6. Implementation Guidelines

#### 6.1 Translation Process

1. Identify the type of content being translated (UI, ability description, lore, etc.)
2. Apply relevant grammatical rules (gender, number, prepositions)
3. Use established terminology from the glossary
4. Apply formatting rules (capitalization, punctuation, accents)
5. Adapt cultural elements as needed
6. Review for consistency with other translations
7. Check for character limitations and adjust if necessary

#### 6.2 Quality Assurance

- Verify gender and number agreement throughout related terms
- Ensure consistent terminology usage
- Check for proper accent usage
- Verify that translations fit in the UI space allocated
- Confirm that the meaning is preserved accurately

#### 6.3 Priority Improvements Based on Validation

1. **Standardize key terminology**: Ensure consistent translations for "Ability" (Habilidade), "Damage" (Dano), "God" (Deus), "Health" (Vida), and "Power" (Poder)
2. **Maintain gender consistency**: Standardize gender assignment for game-specific terms
3. **Ensure proper accent usage**: Verify all Portuguese words have correct accents
4. **Develop UI abbreviations**: Use the standardized abbreviations listed in the glossary for UI space constraints

## Game-Specific Glossary

### Core Game Concepts

| English | Brazilian Portuguese | Gender | Notes |
|---------|---------------------|--------|-------|
| Ability | Habilidade | feminine | a Habilidade; **IMPORTANT**: Validation showed 50% consistency - must be standardized |
| Arena | Arena | feminine | a Arena |
| Assist | Assistência | feminine | a Assistência |
| Attack | Ataque | masculine | o Ataque; **IMPORTANT**: Validation showed 100% consistency - maintain this standard |
| Basic Attack | Ataque Básico | masculine | o Ataque Básico |
| Buff | Buff | masculine | o Buff (English term retained) |
| Cooldown | Recarga | feminine | a Recarga |
| Crowd Control | Controle de Grupo | masculine | o Controle de Grupo; abbreviated as "CC" |
| Damage | Dano | masculine | o Dano; **IMPORTANT**: Validation showed 50% consistency - must be standardized |
| Death | Morte | feminine | a Morte |
| Debuff | Debuff | masculine | o Debuff (English term retained) |
| Defense | Defesa | feminine | a Defesa |
| Effect | Efeito | masculine | o Efeito |
| Experience | Experiência | feminine | a Experiência |
| God | Deus | masculine | o Deus; **IMPORTANT**: Validation showed 33.3% consistency - must be standardized |
| Gold | Ouro | masculine | o Ouro |
| Health | Vida | feminine | a Vida; **IMPORTANT**: Validation showed 0% consistency - must be standardized |
| Item | Item | masculine | o Item; **IMPORTANT**: Validation showed 71.4% consistency - maintain this standard |
| Kill | Abate | masculine | o Abate |
| Lane | Rota | feminine | a Rota |
| Level | Nível | masculine | o Nível |
| Mana | Mana | feminine | a Mana; **IMPORTANT**: Validation showed 100% consistency - maintain this standard |
| Map | Mapa | masculine | o Mapa |
| Minion | Lacaio | masculine | o Lacaio |
| Objective | Objetivo | masculine | o Objetivo |
| Passive | Passiva | feminine | a Passiva; **IMPORTANT**: Validation showed 100% consistency - maintain this standard |
| Power | Poder | masculine | o Poder; **IMPORTANT**: Validation showed 0% consistency - must be standardized |
| Protection | Proteção | feminine | a Proteção |
| Skill | Habilidade | feminine | a Habilidade |
| Speed | Velocidade | feminine | a Velocidade |
| Stat | Atributo | masculine | o Atributo |
| Ultimate | Ultimate | feminine | a Ultimate (often kept in English) |

### Character Classes & Roles

| English | Brazilian Portuguese | Gender | Notes |
|---------|---------------------|--------|-------|
| Assassin | Assassino | masculine | o Assassino |
| Carry | Carry | masculine | o Carry (English term retained) |
| Guardian | Guardião | masculine | o Guardião |
| Hunter | Caçador | masculine | o Caçador |
| Jungler | Jungler | masculine | o Jungler (English term retained) |
| Mage | Mago | masculine | o Mago |
| Mid | Mid | masculine | o Mid (English term retained) |
| Solo | Solo | masculine | o Solo (English term retained) |
| Support | Suporte | masculine | o Suporte |
| Warrior | Guerreiro | masculine | o Guerreiro |

### Combat Terminology

| English | Brazilian Portuguese | Gender | Notes |
|---------|---------------------|--------|-------|
| Area of Effect | Área de Efeito | feminine | a Área de Efeito; abbreviated as "AoE" |
| Critical | Crítico | masculine | o Crítico (as noun) / crítico (as adjective) |
| Critical Strike | Acerto Crítico | masculine | o Acerto Crítico |
| Damage over Time | Dano ao Longo do Tempo | masculine | o Dano ao Longo do Tempo; abbreviated as "DoT" |
| Healing | Cura | feminine | a Cura |
| Lifesteal | Roubo de Vida | masculine | o Roubo de Vida |
| Magical Damage | Dano Mágico | masculine | o Dano Mágico |
| Magical Protection | Proteção Mágica | feminine | a Proteção Mágica |
| Movement Speed | Velocidade de Movimento | feminine | a Velocidade de Movimento |
| Penetration | Penetração | feminine | a Penetração |
| Physical Damage | Dano Físico | masculine | o Dano Físico |
| Physical Protection | Proteção Física | feminine | a Proteção Física |
| Projectile | Projétil | masculine | o Projétil |
| Range | Alcance | masculine | o Alcance |
| Shield | Escudo | masculine | o Escudo |

### Crowd Control Effects

| English | Brazilian Portuguese | Gender | Notes |
|---------|---------------------|--------|-------|
| Banish | Banimento | masculine | o Banimento |
| Blind | Cegueira | feminine | a Cegueira |
| Cripple | Aleijamento | masculine | o Aleijamento |
| Disarm | Desarmamento | masculine | o Desarmamento |
| Fear | Medo | masculine | o Medo |
| Intoxicate | Intoxicação | feminine | a Intoxicação |
| Knockback | Repulsão | feminine | a Repulsão |
| Knockup | Arremesso | masculine | o Arremesso |
| Madness | Loucura | feminine | a Loucura |
| Mesmerize | Hipnose | feminine | a Hipnose |
| Pull | Puxão | masculine | o Puxão |
| Root | Enraizamento | masculine | o Enraizamento |
| Silence | Silêncio | masculine | o Silêncio |
| Slow | Lentidão | feminine | a Lentidão |
| Stun | Atordoamento | masculine | o Atordoamento |
| Taunt | Provocação | feminine | a Provocação |

### Map Elements

| English | Brazilian Portuguese | Gender | Notes |
|---------|---------------------|--------|-------|
| Base | Base | feminine | a Base |
| Camp | Acampamento | masculine | o Acampamento |
| Creep | Creep | masculine | o Creep (English term retained) |
| Fountain | Fonte | feminine | a Fonte |
| Jungle | Selva | feminine | a Selva |
| Lane | Rota | feminine | a Rota |
| Phoenix | Fênix | feminine | a Fênix |
| Titan | Titã | masculine | o Titã |
| Tower | Torre | feminine | a Torre |

### Game Modes

| English | Brazilian Portuguese | Gender | Notes |
|---------|---------------------|--------|-------|
| Arena | Arena | feminine | a Arena |
| Assault | Assalto | masculine | o Assalto |
| Clash | Confronto | masculine | o Confronto |
| Conquest | Conquista | feminine | a Conquista |
| Joust | Justa | feminine | a Justa |
| Ranked | Ranqueada | feminine | a Ranqueada |
| Siege | Cerco | masculine | o Cerco |

### UI Elements

| English | Brazilian Portuguese | Gender | Notes |
|---------|---------------------|--------|-------|
| Back | Voltar | - | verb (infinitive) |
| Cancel | Cancelar | - | verb (infinitive) |
| Confirm | Confirmar | - | verb (infinitive) |
| Exit | Sair | - | verb (infinitive) |
| Leaderboard | Classificação | feminine | a Classificação |
| Load | Carregar | - | verb (infinitive) |
| Menu | Menu | masculine | o Menu |
| Next | Próximo | masculine | o Próximo (as noun) / próximo (as adjective) |
| Options | Opções | feminine (plural) | as Opções |
| Play | Jogar | - | verb (infinitive) |
| Previous | Anterior | - | adjective |
| Profile | Perfil | masculine | o Perfil |
| Queue | Fila | feminine | a Fila |
| Save | Salvar | - | verb (infinitive) |
| Settings | Configurações | feminine (plural) | as Configurações |
| Store | Loja | feminine | a Loja |

### Common Abbreviations

| English | Brazilian Portuguese | Notes |
|---------|---------------------|-------|
| AoE | AoE | Área de Efeito (Area of Effect) |
| CC | CC | Controle de Grupo (Crowd Control) |
| CD | REC | Recarga (Cooldown) |
| DoT | DoT | Dano ao Longo do Tempo (Damage over Time) |
| DPS | DPS | Dano por Segundo (Damage per Second) |
| HP | PV | Pontos de Vida (Health Points) |
| MP | PM | Pontos de Mana (Mana Points) |

### Standardized UI Abbreviations

| Full Brazilian Portuguese Term | Abbreviation | Notes |
|-------------------------------|--------------|-------|
| Velocidade de Movimento | Vel. de Mov. | For movement speed |
| Velocidade de Ataque | Vel. de Atq. | For attack speed |
| Controle de Grupo | CC | For crowd control |
| Dano ao Longo do Tempo | DoT | For damage over time |
| Pontos de Vida | PV | For health points |
| Pontos de Mana | PM | For mana points |
| Proteção Física | Prot. Fís. | For physical protection |
| Proteção Mágica | Prot. Mág. | For magical protection |
| Acerto Crítico | Acerto Crít. | For critical strike |
| Recarga | Rec. | For cooldown |
| Habilidade | Hab. | For ability |
| Regeneração | Regen. | For regeneration |
| Chance de Acerto Crítico | Chance Crít. | For critical chance |

### Compound Terms

| English | Brazilian Portuguese | Gender | Notes |
|---------|---------------------|--------|-------|
| Attack Speed | Velocidade de Ataque | feminine | a Velocidade de Ataque |
| Cooldown Reduction | Redução de Recarga | feminine | a Redução de Recarga |
| Critical Chance | Chance de Acerto Crítico | feminine | a Chance de Acerto Crítico |
| Health Regeneration | Regeneração de Vida | feminine | a Regeneração de Vida |
| Mana Regeneration | Regeneração de Mana | feminine | a Regeneração de Mana |
| Movement Speed | Velocidade de Movimento | feminine | a Velocidade de Movimento |
| Physical Power | Poder Físico | masculine | o Poder Físico |
| Magical Power | Poder Mágico | masculine | o Poder Mágico |

### Numerical Formatting

| English Format | Brazilian Portuguese Format | Example |
|----------------|----------------------------|---------|
| 1.5 | 1,5 | Use comma as decimal separator |
| 1,000 | 1.000 | Use period as thousands separator |
| 50% | 50% | No space before percent sign |
| 10-50 | 10-50 | No spaces around hyphen in ranges |
| 5s | 5s | No space between number and unit |

## Usage Guidelines

### Gender and Articles

Always use the correct gender and corresponding articles:

- **Masculine**: o (singular), os (plural)
  - Example: "o dano" (the damage)
  - Example: "os danos" (the damages)
  
- **Feminine**: a (singular), as (plural)
  - Example: "a habilidade" (the ability)
  - Example: "as habilidades" (the abilities)

### Preposition Contractions

Use appropriate contractions of prepositions with articles:

- de + o = do
  - Example: "Dano do ataque" (Damage of the attack)
- de + a = da
  - Example: "Velocidade da habilidade" (Speed of the ability)
- em + o = no
  - Example: "No mapa" (On the map)
- em + a = na
  - Example: "Na selva" (In the jungle)
- a + o = ao
  - Example: "Resistente ao dano" (Resistant to the damage)
- a + a = à
  - Example: "Resistente à magia" (Resistant to the magic)

### Capitalization

- Unlike English, common nouns are not capitalized in Brazilian Portuguese
  - Example: "dano" not "Dano"
  - Example: "velocidade de movimento" not "Velocidade de Movimento"

- Capitalize sentence beginnings and proper nouns
  - Example: "Este item aumenta o dano" (This item increases damage)
  - Example: "Mercúrio é um deus" (Mercury is a god)

### Accents

Always use proper accents as they can change meaning:

- Acute accent (á, é, í, ó, ú): indicates stress
  - Example: "saúde" (health)
- Circumflex accent (â, ê, ô): indicates closed vowels
  - Example: "três" (three)
- Tilde (ã, õ): indicates nasalization
  - Example: "proteção" (protection)
- Cedilla (ç): indicates 's' sound before a, o, u
  - Example: "caçador" (hunter)

### Formality Level

- Use informal "você" form for player instructions
  - Example: "Você ganha vida" (You gain health)
  - Example: "Suas habilidades" (Your abilities)

- Use infinitive forms for menu items and buttons
  - Example: "Jogar" (Play)
  - Example: "Confirmar" (Confirm)

## Priority Improvements Based on Validation

1. **Standardize key terminology**: Ensure consistent translations for "Ability" (Habilidade), "Damage" (Dano), "God" (Deus), "Health" (Vida), and "Power" (Poder)
2. **Maintain gender consistency**: Standardize gender assignment for game-specific terms
3. **Ensure proper accent usage**: Verify all Portuguese words have correct accents
4. **Use standardized abbreviations**: Apply the abbreviations listed in this glossary for UI space constraints
