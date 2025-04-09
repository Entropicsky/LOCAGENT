# SMITE 2 French (frFR) Translation Ruleset

This document provides comprehensive guidelines for translating SMITE 2 content from English to French. It includes special translation rules and a game-specific glossary derived from extensive analysis of the SMITE 2 translation dataset.

## Special Translation Rules

### 1. Grammatical Rules

#### 1.1 Gender Assignment

French requires gender assignment for nouns, which affects articles, adjectives, and other related words.

- **Rule G1**: Assign feminine gender to abstract concepts (abilities, statistics) unless established otherwise
  - Example: "Movement Speed" → "Vitesse de déplacement" (feminine)
  - Example: "Physical Protection" → "Protection physique" (feminine)

- **Rule G2**: Assign masculine gender to concrete game elements (items, characters) unless established otherwise
  - Example: "Item" → "Objet" (masculine)
  - Example: "Tower" → "Tour" (feminine, exception)

- **Rule G3**: Maintain consistent gender for game-specific terms throughout all translations
  - Example: If "Buff" is translated as "Amélioration" (feminine), use feminine articles and adjectives consistently
  - **IMPORTANT**: Review gender assignment for "God" (should be "Dieu" with masculine articles)

#### 1.2 Article Usage

- **Rule A1**: Add definite articles (le, la, les) when translating English terms that lack articles
  - Example: "Damage increases with level" → "Les dégâts augmentent avec le niveau"

- **Rule A2**: Use contracted forms with prepositions as required by French grammar
  - Example: "de + le" → "du" (e.g., "Power of the item" → "Puissance de l'objet")
  - Example: "à + les" → "aux" (e.g., "Damage to enemies" → "Dégâts aux ennemis")

#### 1.3 Adjective Placement

- **Rule AP1**: Place most adjectives after the noun they modify
  - Example: "Physical Damage" → "Dégâts physiques"
  - Example: "Magical Protection" → "Protection magique"

- **Rule AP2**: Place common descriptive adjectives (petit, grand, bon, nouveau, etc.) before the noun
  - Example: "New Item" → "Nouvel objet"
  - Example: "Good strategy" → "Bonne stratégie"

#### 1.4 Verb Forms

- **Rule V1**: Use second person plural (vous) for player instructions and tooltips
  - Example: "Press to activate" → "Appuyez pour activer"
  - Example: "You gain health" → "Vous gagnez de la santé"
  - **IMPORTANT**: Never use the informal "tu" form in player instructions

- **Rule V2**: Use infinitive forms for action labels and menu items
  - Example: "Activate" → "Activer"
  - Example: "Purchase" → "Acheter"

- **Rule V3**: Use present tense for ability descriptions and passive effects
  - Example: "This ability stuns enemies" → "Cette compétence étourdit les ennemis"

### 2. Formatting Rules

#### 2.1 Capitalization

- **Rule C1**: Capitalize the first word of sentences and proper nouns only
  - Example: "Physical Damage" → "Dégâts physiques" (no capitals except at sentence start)

- **Rule C2**: Maintain capitalization for game-specific proper nouns (God names, special abilities)
  - Example: "Zeus's Lightning Bolt" → "Éclair de Zeus"

- **Rule C3**: Do not capitalize common game terms unless at the beginning of a sentence
  - Example: "Magical Power" → "Puissance magique"

#### 2.2 Punctuation

- **Rule P1**: Add spaces before double punctuation marks (: ; ! ? » )
  - Example: "Ready!" → "Prêt !"
  - Example: "Options: Audio" → "Options : Audio"
  - **IMPORTANT**: This rule must be applied consistently as validation showed ~30% inconsistency

- **Rule P2**: Use French quotation marks (guillemets) with spaces
  - Example: "Victory" → "« Victoire »"

- **Rule P3**: Use commas instead of decimal points for numbers
  - Example: "2.5 seconds" → "2,5 secondes"

#### 2.3 Number and Unit Formatting

- **Rule N1**: Add a space between a number and its unit
  - Example: "5s cooldown" → "5 s de temps de recharge"
  - Example: "10m range" → "10 m de portée"
  - **IMPORTANT**: This rule must be applied consistently as validation showed formatting inconsistencies

- **Rule N2**: Use French abbreviations for units when available
  - Example: "sec" → "s" (seconds)
  - Example: "pts" (points)

### 3. Terminology Rules

#### 3.1 Game-Specific Terms

- **Rule T1**: Use established French MOBA terminology for core concepts
  - "Damage" → "Dégâts" (validation showed inconsistency - must be standardized)
  - "Cooldown" → "Temps de recharge"
  - "Ability" → "Compétence" (validation showed inconsistency - must be standardized)
  - "Ultimate" → "Ultime"
  - "Passive" → "Passif"
  - "Crowd Control" → "Contrôle de foule"
  - "Health" → "Santé" (validation showed inconsistency - must be standardized)
  - "Power" → "Puissance" (validation showed inconsistency - must be standardized)
  - "God" → "Dieu" (validation showed significant inconsistency - must be standardized)

- **Rule T2**: Preserve English terms that are commonly used in the French gaming community
  - "Buff" → "Buff"
  - "Nerf" → "Nerf"
  - "Skill shot" → "Skill shot"
  - "Gank" → "Gank"

- **Rule T3**: Translate role names consistently
  - "Warrior" → "Guerrier"
  - "Guardian" → "Gardien"
  - "Mage" → "Mage"
  - "Hunter" → "Chasseur"
  - "Assassin" → "Assassin"

#### 3.2 UI Elements

- **Rule UI1**: Keep UI element translations concise, ideally no more than 30% longer than English
  - Example: Abbreviate where possible without losing meaning
  - Example: Use shorter synonyms when available

- **Rule UI2**: Use consistent translations for recurring UI elements
  - "Settings" → "Paramètres"
  - "Options" → "Options"
  - "Play" → "Jouer"
  - "Exit" → "Quitter"

#### 3.3 Ability Descriptions

- **Rule AD1**: Maintain a consistent structure for ability descriptions
  - Start with a brief description of what the ability does
  - Follow with damage values and effects
  - End with cooldown information

- **Rule AD2**: Use consistent terminology for damage types
  - "Physical Damage" → "Dégâts physiques"
  - "Magical Damage" → "Dégâts magiques"

- **Rule AD3**: Use consistent terminology for crowd control effects
  - "Stun" → "Étourdissement"
  - "Root" → "Immobilisation"
  - "Silence" → "Silence"
  - "Slow" → "Ralentissement"

### 4. Cultural Adaptation Rules

#### 4.1 Formality Level

- **Rule F1**: Use formal "vous" form for all player instructions and tooltips
  - Example: "You gain health" → "Vous gagnez de la santé"
  - **IMPORTANT**: Never use the informal "tu" form in player instructions

- **Rule F2**: Use a neutral, somewhat formal tone for game narration
  - Avoid slang or overly casual expressions
  - Maintain professional language while being engaging

#### 4.2 Mythology References

- **Rule M1**: Use established French names for mythological figures when they exist
  - Example: "Athena" → "Athéna"
  - Example: "Mercury" → "Mercure"

- **Rule M2**: Preserve original names when no established French equivalent exists
  - Example: "Ah Muzen Cab" remains "Ah Muzen Cab"

#### 4.3 Humor and Wordplay

- **Rule H1**: Adapt humor and wordplay to maintain the spirit rather than literal translation
  - Focus on creating equivalent humor in French rather than direct translation
  - Ensure cultural references are understandable to French players

### 5. Exception Handling Rules

#### 5.1 Untranslatable Terms

- **Rule E1**: Keep untranslatable game-specific terms in English, but apply French grammar rules
  - Example: Add appropriate articles and gender
  - Example: Apply French pluralization rules when necessary

#### 5.2 Character Limitations

- **Rule L1**: When space is limited (UI elements), prioritize clarity over completeness
  - Use abbreviations for common terms (e.g., "Temps de recharge" → "TR")
  - Omit articles when absolutely necessary, but preserve meaning

#### 5.3 Inconsistency Resolution

- **Rule I1**: When encountering inconsistent translations in the existing database, prefer:
  1. The most frequently used translation
  2. The translation that best follows these rules
  3. The most recent translation

### 6. Implementation Guidelines

#### 6.1 Translation Process

1. Identify the type of content being translated (UI, ability description, lore, etc.)
2. Apply relevant grammatical rules (gender, articles, adjective placement)
3. Use established terminology from the glossary
4. Apply formatting rules (capitalization, punctuation, numbers)
5. Adapt cultural elements as needed
6. Review for consistency with other translations
7. Check for character limitations and adjust if necessary

#### 6.2 Quality Assurance

- Verify gender consistency throughout related terms
- Ensure consistent terminology usage
- Check formatting, especially spaces before punctuation
- Verify that translations fit in the UI space allocated
- Confirm that the meaning is preserved accurately

#### 6.3 Priority Improvements Based on Validation

1. **Standardize key terminology**: Ensure consistent translations for "God" (Dieu), "Ability" (Compétence), "Damage" (Dégâts), "Health" (Santé), and "Power" (Puissance)
2. **Enforce punctuation rules**: Add spaces before double punctuation marks consistently
3. **Maintain gender consistency**: Standardize gender assignment for game-specific terms
4. **Number formatting**: Ensure consistent use of decimal commas and spaces between numbers and units

## Game-Specific Glossary

### Core Game Concepts

| English | French | Notes |
|---------|--------|-------|
| Ability | Compétence | Feminine noun |
| Arena | Arène | Feminine noun |
| Assist | Assistance | Feminine noun |
| Attack | Attaque | Feminine noun |
| Basic Attack | Attaque de base | Feminine noun |
| Buff | Buff | Masculine noun; keep English term |
| Cooldown | Temps de recharge | Masculine noun |
| Crowd Control | Contrôle de foule | Masculine noun; abbreviated as "CC" |
| Damage | Dégâts | Masculine plural noun |
| Death | Mort | Feminine noun |
| Debuff | Débuff | Masculine noun |
| Defense | Défense | Feminine noun |
| Effect | Effet | Masculine noun |
| Experience | Expérience | Feminine noun |
| God | Dieu | Masculine noun |
| Gold | Or | Masculine noun |
| Health | Santé | Feminine noun |
| Item | Objet | Masculine noun |
| Kill | Élimination | Feminine noun |
| Lane | Couloir | Masculine noun |
| Level | Niveau | Masculine noun |
| Mana | Mana | Masculine noun |
| Map | Carte | Feminine noun |
| Minion | Sbire | Masculine noun |
| Objective | Objectif | Masculine noun |
| Passive | Passif | Masculine noun |
| Power | Puissance | Feminine noun |
| Protection | Protection | Feminine noun |
| Skill | Compétence | Feminine noun |
| Speed | Vitesse | Feminine noun |
| Stat | Statistique | Feminine noun |
| Ultimate | Ultime | Masculine noun |

### Character Classes & Roles

| English | French | Notes |
|---------|--------|-------|
| Assassin | Assassin | Masculine noun |
| Carry | Carry | Masculine noun; keep English term |
| Guardian | Gardien | Masculine noun |
| Hunter | Chasseur | Masculine noun |
| Jungler | Jungler | Masculine noun; keep English term |
| Mage | Mage | Masculine noun |
| Mid | Mid | Masculine noun; keep English term |
| Solo | Solo | Masculine noun; keep English term |
| Support | Support | Masculine noun; keep English term |
| Warrior | Guerrier | Masculine noun |

### Combat Terminology

| English | French | Notes |
|---------|--------|-------|
| Area of Effect | Zone d'effet | Feminine noun; abbreviated as "AoE" |
| Critical | Critique | Adjective |
| Critical Strike | Coup critique | Masculine noun |
| Damage over Time | Dégâts sur la durée | Abbreviated as "DoT" |
| Healing | Soins | Masculine plural noun |
| Lifesteal | Vol de vie | Masculine noun |
| Magical Damage | Dégâts magiques | Masculine plural noun |
| Magical Protection | Protection magique | Feminine noun |
| Movement Speed | Vitesse de déplacement | Feminine noun |
| Penetration | Pénétration | Feminine noun |
| Physical Damage | Dégâts physiques | Masculine plural noun |
| Physical Protection | Protection physique | Feminine noun |
| Projectile | Projectile | Masculine noun |
| Range | Portée | Feminine noun |
| Shield | Bouclier | Masculine noun |

### Crowd Control Effects

| English | French | Notes |
|---------|--------|-------|
| Banish | Bannissement | Masculine noun |
| Blind | Aveuglement | Masculine noun |
| Cripple | Handicap | Masculine noun |
| Disarm | Désarmement | Masculine noun |
| Fear | Peur | Feminine noun |
| Intoxicate | Intoxication | Feminine noun |
| Knockback | Répulsion | Feminine noun |
| Knockup | Projection | Feminine noun |
| Madness | Folie | Feminine noun |
| Mesmerize | Hypnose | Feminine noun |
| Pull | Attraction | Feminine noun |
| Root | Immobilisation | Feminine noun |
| Silence | Silence | Masculine noun |
| Slow | Ralentissement | Masculine noun |
| Stun | Étourdissement | Masculine noun |
| Taunt | Provocation | Feminine noun |

### Map Elements

| English | French | Notes |
|---------|--------|-------|
| Base | Base | Feminine noun |
| Camp | Camp | Masculine noun |
| Creep | Creep | Masculine noun; keep English term |
| Fountain | Fontaine | Feminine noun |
| Jungle | Jungle | Feminine noun |
| Lane | Couloir | Masculine noun |
| Phoenix | Phénix | Masculine noun |
| Titan | Titan | Masculine noun |
| Tower | Tour | Feminine noun |

### Game Modes

| English | French | Notes |
|---------|--------|-------|
| Arena | Arène | Feminine noun |
| Assault | Assaut | Masculine noun |
| Clash | Choc | Masculine noun |
| Conquest | Conquête | Feminine noun |
| Joust | Joute | Feminine noun |
| Ranked | Classé | Adjective |
| Siege | Siège | Masculine noun |

### UI Elements

| English | French | Notes |
|---------|--------|-------|
| Back | Retour | Masculine noun |
| Cancel | Annuler | Verb (infinitive) |
| Confirm | Confirmer | Verb (infinitive) |
| Exit | Quitter | Verb (infinitive) |
| Leaderboard | Classement | Masculine noun |
| Load | Charger | Verb (infinitive) |
| Menu | Menu | Masculine noun |
| Next | Suivant | Adjective |
| Options | Options | Feminine plural noun |
| Play | Jouer | Verb (infinitive) |
| Previous | Précédent | Adjective |
| Profile | Profil | Masculine noun |
| Queue | File d'attente | Feminine noun |
| Save | Sauvegarder | Verb (infinitive) |
| Settings | Paramètres | Masculine plural noun |
| Store | Boutique | Feminine noun |

### Common Abbreviations

| English | French | Notes |
|---------|--------|-------|
| AoE | AoE | Zone d'effet |
| CC | CC | Contrôle de foule |
| CD | TR | Temps de recharge |
| DoT | DoT | Dégâts sur la durée |
| DPS | DPS | Dégâts par seconde |
| HP | PV | Points de vie |
| MP | PM | Points de mana |

### Numerical Formatting

| English Format | French Format | Example |
|----------------|---------------|---------|
| 1.5 | 1,5 | Use comma as decimal separator |
| 1,000 | 1 000 | Use space as thousands separator |
| 50% | 50 % | Add space before percent sign |
| 10-50 | 10 - 50 | Add spaces around hyphen in ranges |
| 5s | 5 s | Add space between number and unit |

### Notes on Usage

1. **Gender Agreement**: Always ensure proper gender agreement with articles and adjectives
   - Example: "la puissance magique" (feminine)
   - Example: "le dégât physique" (masculine)

2. **Formality**: Use the formal "vous" form for player instructions
   - Example: "Vous gagnez 5 points de mana" (not "Tu gagnes")

3. **Capitalization**: Only capitalize proper nouns and sentence beginnings
   - Example: "dégâts magiques" (not "Dégâts Magiques")

4. **Punctuation**: Add spaces before double punctuation marks
   - Example: "Prêt !" (not "Prêt!")

5. **Terminology Consistency**: Always use the terms in this glossary rather than creating new translations

6. **Untranslated Terms**: Some gaming terms remain in English but follow French grammar rules
   - Example: "les buffs" (plural with French article)
