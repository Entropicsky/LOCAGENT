# SMITE 2 Translation Ruleset: Japanese (jaJP)

This comprehensive ruleset provides guidelines, rules, and a standardized glossary for translating SMITE 2 content from English to Japanese. It has been developed through linguistic analysis, pattern identification, terminology extraction, and validation against the dataset to ensure high-quality, consistent translations.

## Table of Contents

1. [Introduction](#introduction)
2. [Japanese Language Characteristics](#japanese-language-characteristics)
3. [Special Translation Rules](#special-translation-rules)
4. [Game-Specific Glossary](#game-specific-glossary)
5. [Implementation Guidelines](#implementation-guidelines)
6. [Quality Assurance](#quality-assurance)

## Introduction

SMITE 2 is a popular third-person MOBA (Multiplayer Online Battle Arena) game featuring gods and mythological figures from various pantheons. This document provides a comprehensive set of rules and a standardized glossary for translating SMITE 2 content from English to Japanese, ensuring consistency and quality across all game elements.

The ruleset has been developed through:
- Linguistic analysis of Japanese language characteristics
- Identification of translation patterns
- Extraction of game-specific terminology
- Validation against existing translations
- Refinement based on validation findings

## Japanese Language Characteristics

### Writing System

Japanese uses three writing systems simultaneously:

- **Kanji (漢字)**: Chinese characters adapted for Japanese, used for content words (nouns, verb stems, adjective stems)
- **Hiragana (ひらがな)**: Phonetic syllabary used for grammatical elements, native Japanese words, and words without kanji
- **Katakana (カタカナ)**: Phonetic syllabary primarily used for foreign loanwords, foreign names, emphasis, and technical/scientific terms

In SMITE 2 translations, this mixed writing system has significant implications:
- Game-specific terminology often appears in katakana (e.g., "アビリティ" for "Ability")
- Core Japanese concepts use kanji (e.g., "攻撃" for "Attack")
- Grammatical particles and inflections use hiragana

### Grammatical Structure

Japanese follows a Subject-Object-Verb (SOV) word order, unlike English's Subject-Verb-Object (SVO):

- English: "The player (S) uses (V) an ability (O)"
- Japanese: "プレイヤーが (S) アビリティを (O) 使用する (V)"

This structural difference requires complete reorganization of sentences during translation.

Japanese uses grammatical particles (postpositions) to mark the function of words in a sentence:

- **が (ga)**: Marks the subject
- **を (wo/o)**: Marks the direct object
- **に (ni)**: Marks indirect object, direction, location, time
- **で (de)**: Marks location of action, means, or instrument
- **の (no)**: Indicates possession or modification
- **と (to)**: Means "with" or "and"
- **へ (e)**: Indicates direction

Japanese lacks articles (a, an, the) and does not generally mark plurals:

- "神" can mean "god" or "gods" depending on context
- "アビリティ" can mean "an ability," "the ability," or "abilities"

Japanese has multiple levels of formality and politeness:

- **Casual/Plain form**: Used among friends, in informal settings
- **Polite form (-masu/-desu)**: Standard polite language for most situations
- **Honorific form**: Shows respect to superiors or customers
- **Humble form**: Shows humility when referring to oneself

### Loanwords and Abbreviations

Japanese readily adopts foreign words, especially from English, often with modified pronunciation:

- **Direct loanwords**: "ダメージ" (damēji) for "damage"
- **Wasei-Eigo** (Japanese-made English): Words that appear English but are uniquely Japanese

In SMITE 2, many gaming and technical terms are loanwords written in katakana:
- "アルティメット" (arutimetto) for "ultimate"
- "スキル" (sukiru) for "skill"
- "バフ" (bafu) for "buff"

Japanese gaming culture has developed its own abbreviations:
- "物理攻撃力" → "物攻" (Physical Power)
- "魔法防御力" → "魔防" (Magical Protection)

## Special Translation Rules

### 1. Writing System Rules

#### 1.1 Script Selection

- **Rule WS1**: Use katakana for:
  - Gaming-specific terminology (e.g., "アビリティ" for "Ability")
  - Foreign loanwords (e.g., "ダメージ" for "Damage")
  - Non-Japanese god names (e.g., "ゼウス" for "Zeus")
  - Character class names (e.g., "アサシン" for "Assassin")
  - Game mode names (e.g., "アリーナ" for "Arena")
  - **Playable characters**: Use "ゴッド" (goddo) in katakana for "God" when referring to playable characters

- **Rule WS2**: Use kanji (with appropriate hiragana particles) for:
  - Native Japanese concepts (e.g., "攻撃" for "Attack")
  - Japanese god names (e.g., "天照" for "Amaterasu")
  - UI navigation elements (e.g., "設定" for "Settings")
  - Map elements with Japanese equivalents (e.g., "本拠地" for "Base")

- **Rule WS3**: Use mixed script for compound terms combining native and foreign concepts:
  - Example: "魔法ダメージ" for "Magical Damage" (kanji "魔法" + katakana "ダメージ")
  - Example: "クールダウン短縮" for "Cooldown Reduction" (katakana "クールダウン" + kanji "短縮")

#### 1.2 Abbreviations

- **Rule AB1**: For UI space constraints, use established Japanese gaming abbreviations:
  - Combine first characters of compound terms: "物理攻撃力" → "物攻"
  - Retain English abbreviations when commonly used in Japanese gaming: "HP", "MP", "DPS"
  - For hybrid abbreviations, combine English abbreviations with Japanese terms: "CD短縮"
  - **Explicitly introduce Japanese abbreviations** in tooltips with format: "物理攻撃力（物攻）"

- **Rule AB2**: When introducing abbreviations, provide the full term first:
  - Example: "継続ダメージ（DoT）" then "DoT" in subsequent mentions
  - Example: "クラウドコントロール（CC）" then "CC" in subsequent mentions
  - For "Health", consistently use "HP" or "ヒットポイント" in UI elements

### 2. Grammatical Rules

#### 2.1 Sentence Structure

- **Rule GR1**: Reorganize English SVO sentences to Japanese SOV structure:
  - English: "[Subject] [Verb] [Object]"
  - Japanese: "[Subject]は/が [Object]を [Verb]"
  - Example: "This ability deals damage" → "このアビリティはダメージを与える"

- **Rule GR2**: For passive voice, use Japanese passive verb forms:
  - English: "[Subject] is [Past Participle]"
  - Japanese: "[Subject]は [Passive Verb]"
  - Example: "Enemies are stunned" → "敵はスタンされる"

#### 2.2 Particles

- **Rule PA1**: Use appropriate particles to mark grammatical relationships:
  - は/が (wa/ga): Subject marker
    - Example: "Player" → "プレイヤーは/が"
  - を (wo/o): Direct object marker
    - Example: "Damage" → "ダメージを"
  - に (ni): Indirect object/direction marker
    - Example: "To enemies" → "敵に"
  - で (de): Location of action marker
    - Example: "In the jungle" → "ジャングルで"
  - の (no): Possession/modification marker
    - Example: "Player's health" → "プレイヤーのHP"

- **Rule PA2**: For compound modifiers, use the の (no) particle:
  - English: "[Noun1] [Preposition] [Noun2]"
  - Japanese: "[Noun2]の[Noun1]"
  - Example: "Protection of allies" → "味方の防御力"

#### 2.3 Articles and Plurals

- **Rule AP1**: Omit English articles (a, an, the) as Japanese has no direct equivalent:
  - Example: "The ability" → "アビリティ"
  - Example: "A powerful attack" → "強力な攻撃"

- **Rule AP2**: Handle plurals according to context, as Japanese typically doesn't mark plurals:
  - For general plurals, use the singular form: "Enemies" → "敵"
  - For specific counted plurals, use counters: "3 gods" → "3体の神"
  - To explicitly indicate plurality, use 複数の (fukusū no): "Multiple enemies" → "複数の敵"

#### 2.4 Verb Forms

- **Rule VF1**: Use appropriate politeness level consistently:
  - For UI elements and tooltips: Polite form (-masu/-desu)
    - Example: "Select" → "選択します"
  - For character dialogue: Match character personality
    - Casual: "行くぞ" (Let's go!)
    - Formal: "参りましょう" (Let's proceed)

- **Rule VF2**: For imperative forms (commands), use:
  - Polite request: [Verb stem]てください
    - Example: "Attack!" → "攻撃してください"
  - For UI buttons, use the dictionary form or noun form:
    - Example: "Confirm" → "確認"
    - Example: "Cancel" → "キャンセル"

### 3. Terminology Rules

#### 3.1 Core Game Terms

- **Rule TM1**: Use established Japanese gaming terminology consistently:
  - For terms with established native Japanese equivalents, use the native term:
    - "Attack" → "攻撃" (kōgeki) [98.7% consistency in dataset]
    - "Defense" → "防御" (bōgyo)
    - "Health" → "HP" or "ヒットポイント" (hitto pointo) [only 4.3% consistency with "体力"]
  
  - For gaming-specific terms commonly used as loanwords, use katakana:
    - "Ability" → "アビリティ" (abiritī) [79.4% consistency in dataset]
    - "Ultimate" → "アルティメット" (arutimetto) [83.3% consistency in dataset]
    - "Buff" → "バフ" (bafu)
    - "God" → "ゴッド" (goddo) when referring to playable characters [39.0% consistency with "神"]
    - "Shield" → "シールド" (shīrudo) [54.5% consistency in dataset]

- **Rule TM2**: For compound terms, follow established patterns:
  - Native + Native: "攻撃速度" (Attack Speed)
  - Loanword + Native: "アビリティ威力" (Ability Power)
  - Native + Loanword: "魔法ダメージ" (Magical Damage)
  - Loanword + Loanword: "クリティカルダメージ" (Critical Damage)

#### 3.2 Technical Gaming Terms

- **Rule TG1**: For technical gaming terms, follow Japanese gaming conventions:
  - Use both English abbreviations and Japanese terms where appropriate:
    - "HP" or "ヒットポイント" (Health)
    - "MP" or "マナポイント" (Mana)
    - "DPS" or "継続ダメージ" (Damage per Second)
  
  - For crowd control effects, use established terms:
    - "Stun" → "スタン" (sutan) [100% consistency in dataset]
    - "Root" → "拘束" (kōsoku)
    - "Silence" → "沈黙" (chinmoku)

#### 3.3 Mythological Terms

- **Rule MT1**: For mythological names and concepts:
  - Use traditional Japanese names for Japanese deities:
    - "Amaterasu" → "天照"
    - "Susanoo" → "須佐之男"
  
  - Use established Japanese translations for well-known non-Japanese deities:
    - "Zeus" → "ゼウス" (zeusu)
    - "Odin" → "オーディン" (ōdin)
  
  - Use katakana transliteration for less common deities:
    - "Ah Muzen Cab" → "アー・ムゼン・カブ" (ā muzen kabu)

### 4. Formatting Rules

#### 4.1 Numbers and Units

- **Rule NU1**: Format numbers according to Japanese conventions:
  - Use same decimal point: "1.5" → "1.5"
  - Use same thousands separator: "1,000" → "1,000"
  - Use same percentage format: "50%" → "50%"
  - Use Japanese wave dash for ranges: "10-50" → "10～50"
  - Add appropriate units: "5s" → "5秒" (5 seconds)

- **Rule NU2**: For game statistics, place units according to Japanese conventions:
  - Percentage after value: "+10% Damage" → "ダメージ+10%"
  - Unit before value for native Japanese units: "Health 100" → "HP100"

#### 4.2 UI Elements

- **Rule UI1**: Keep UI element translations concise:
  - Use shortest clear option for button text: "Confirm" → "確認"
  - Use abbreviations for space-constrained elements: "Physical Protection" → "物防"
  - Break lines at grammatical points for tooltips

- **Rule UI2**: For navigation elements, use natural Japanese expressions:
  - "Back" → "戻る" (verb "to return")
  - "Next" → "次へ" (literally "to next")
  - "Previous" → "前へ" (literally "to previous")

### 5. Cultural Adaptation Rules

#### 5.1 Politeness Level

- **Rule PL1**: Maintain consistent politeness level throughout the game:
  - For general UI and system messages: Use polite form (-masu/-desu)
  - For tooltips and ability descriptions: Use polite or plain form consistently
  - For character dialogue: Match the character's personality and cultural background

#### 5.2 Cultural References

- **Rule CR1**: Adapt cultural references to be understandable to Japanese players:
  - Replace Western idioms with Japanese equivalents
  - Provide context for mythological references unfamiliar to Japanese players
  - Preserve references to Japanese culture when they appear in the original

#### 5.3 Humor and Wordplay

- **Rule HW1**: Adapt humor and wordplay to maintain the spirit rather than literal translation:
  - Focus on creating equivalent humor in Japanese
  - For puns, create new Japanese wordplay that preserves the theme
  - For character-specific humor, ensure it matches the character's personality

### 6. Exception Handling Rules

#### 6.1 Untranslatable Terms

- **Rule UT1**: For game-specific terms without established Japanese equivalents:
  - Use katakana transliteration
  - Provide explanatory text in parentheses on first mention
  - Example: "Proc" → "プロック（特定条件で発動する効果）"

#### 6.2 Character Limitations

- **Rule CL1**: When space is limited (UI elements):
  - Use established abbreviations (see Rule AB1)
  - Omit particles when meaning remains clear
  - Use symbols instead of words where appropriate: "+" instead of "増加" (increase)

#### 6.3 Inconsistency Resolution

- **Rule IR1**: When encountering inconsistent translations in existing content:
  1. Prefer the most frequently used translation
  2. Prefer the translation that follows these rules
  3. Prefer the most recent translation
  - For "God", use "ゴッド" (goddo) when referring to playable characters and "神" (kami) when referring to deities in a mythological context
  - For "Health", use "HP" or "ヒットポイント" rather than "体力"
  - For "Shield", use "シールド" (shīrudo) for gameplay mechanics and "盾" (tate) only when referring to the physical object

## Game-Specific Glossary

### Core Game Concepts

| English | Japanese Translation | Script | Pronunciation | Notes |
|---------|---------------------|--------|--------------|-------|
| Ability | アビリティ | Katakana | Abiritī | Loanword from English; 79.4% consistency in dataset |
| Arena | アリーナ | Katakana | Arīna | Loanword from English |
| Assist | アシスト | Katakana | Ashisuto | Loanword from English |
| Attack | 攻撃 | Kanji | Kōgeki | Native Japanese term; 98.7% consistency in dataset |
| Basic Attack | 通常攻撃 | Kanji | Tsūjō kōgeki | Literally "normal attack" |
| Buff | バフ | Katakana | Bafu | Loanword from English |
| Cooldown | クールダウン | Katakana | Kūrudaun | Loanword from English; 100% consistency in dataset |
| Crowd Control | クラウドコントロール | Katakana | Kuraudo kontorōru | Often abbreviated as "CC"; 3.2% usage of CC abbreviation |
| Damage | ダメージ | Katakana | Damēji | Loanword from English; 100% consistency in dataset |
| Death | 死亡 | Kanji | Shibō | Native Japanese term |
| Debuff | デバフ | Katakana | Debafu | Loanword from English |
| Defense | 防御 | Kanji | Bōgyo | Native Japanese term |
| Effect | 効果 | Kanji | Kōka | Native Japanese term |
| Experience | 経験値 | Kanji | Keiken-chi | Literally "experience value" |
| God | ゴッド | Katakana | Goddo | Use for playable characters; 39% consistency with "神" |
| God (deity) | 神 | Kanji | Kami | Use when referring to deities in mythological context |
| Gold | ゴールド | Katakana | Gōrudo | Loanword from English |
| Health | HP or ヒットポイント | Katakana | Hitto pointo | Use instead of "体力" (4.3% consistency) |
| Item | アイテム | Katakana | Aitemu | Loanword from English |
| Kill | キル | Katakana | Kiru | Loanword; also 撃破 (gekiha) |
| Lane | レーン | Katakana | Rēn | Loanword from English |
| Level | レベル | Katakana | Reberu | Loanword from English |
| Mana | マナ | Katakana | Mana | Gaming loanword; 100% consistency in dataset |
| Map | マップ | Katakana | Mappu | Loanword from English |
| Minion | ミニオン | Katakana | Minion | Loanword from English |
| Objective | 目標 | Kanji | Mokuhyō | Native Japanese term |
| Passive | パッシブ | Katakana | Pashibu | Loanword from English |
| Power | 攻撃力 | Kanji | Kōgeki-ryoku | Literally "attack power" |
| Protection | 防御力 | Kanji | Bōgyo-ryoku | Literally "defense power" |
| Shield | シールド | Katakana | Shīrudo | Use for gameplay mechanics; 54.5% consistency |
| Shield (object) | 盾 | Kanji | Tate | Use only when referring to physical object |
| Skill | スキル | Katakana | Sukiru | Loanword from English |
| Speed | 速度 | Kanji | Sokudo | Native Japanese term |
| Stat | ステータス | Katakana | Sutētasu | Loanword from English |
| Ultimate | アルティメット | Katakana | Arutimetto | Loanword from English; 83.3% consistency in dataset |

### Character Classes & Roles

| English | Japanese Translation | Script | Pronunciation | Notes |
|---------|---------------------|--------|--------------|-------|
| Assassin | アサシン | Katakana | Asashin | Loanword from English |
| Carry | キャリー | Katakana | Kyarī | Loanword from English |
| Guardian | ガーディアン | Katakana | Gādian | Loanword from English |
| Hunter | ハンター | Katakana | Hantā | Loanword from English |
| Jungler | ジャングラー | Katakana | Jangurā | Loanword from English |
| Mage | メイジ | Katakana | Meiji | Loanword from English |
| Mid | ミッド | Katakana | Middo | Loanword from English |
| Solo | ソロ | Katakana | Soro | Loanword from English |
| Support | サポート | Katakana | Sapōto | Loanword from English |
| Warrior | ウォリアー | Katakana | Woriā | Loanword from English |

### Combat Terminology

| English | Japanese Translation | Script | Pronunciation | Notes |
|---------|---------------------|--------|--------------|-------|
| Area of Effect | 範囲効果 | Kanji | Han'i kōka | Literally "range effect"; abbreviated as "AoE" |
| Critical | クリティカル | Katakana | Kuritikaru | Loanword from English |
| Critical Strike | クリティカルヒット | Katakana | Kuritikaru hitto | Loanword from English |
| Damage over Time | 継続ダメージ | Mixed | Keizoku damēji | Literally "continuous damage"; abbreviated as "DoT" |
| Healing | 回復 | Kanji | Kaifuku | Native Japanese term |
| Lifesteal | ライフスティール | Katakana | Raifusutīru | Sometimes 吸血 (kyūketsu, "blood sucking") |
| Magical Damage | 魔法ダメージ | Mixed | Mahō damēji | Hybrid term |
| Magical Protection | 魔法防御力 | Kanji | Mahō bōgyo-ryoku | Introduce as "魔法防御力（魔防）" |
| Movement Speed | 移動速度 | Kanji | Idō sokudo | Introduce as "移動速度（移速）" |
| Penetration | 貫通 | Kanji | Kantū | Native Japanese term |
| Physical Damage | 物理ダメージ | Mixed | Butsuri damēji | Hybrid term |
| Physical Protection | 物理防御力 | Kanji | Butsuri bōgyo-ryoku | Introduce as "物理防御力（物防）" |
| Projectile | 発射体 | Kanji | Hassha-tai | Native Japanese term |
| Range | 射程 | Kanji | Shatei | Native Japanese term |

### Crowd Control Effects

| English | Japanese Translation | Script | Pronunciation | Notes |
|---------|---------------------|--------|--------------|-------|
| Banish | 追放 | Kanji | Tsuihō | Native Japanese term |
| Blind | 盲目 | Kanji | Mōmoku | Native Japanese term |
| Cripple | 行動阻害 | Kanji | Kōdō sogai | Literally "action hindrance" |
| Disarm | 武装解除 | Kanji | Busō kaijo | Literally "disarmament" |
| Fear | 恐怖 | Kanji | Kyōfu | Native Japanese term |
| Intoxicate | 酩酊 | Kanji | Meitei | Native Japanese term |
| Knockback | ノックバック | Katakana | Nokkubakku | Loanword from English |
| Knockup | ノックアップ | Katakana | Nokku appu | Loanword from English |
| Madness | 狂気 | Kanji | Kyōki | Native Japanese term |
| Mesmerize | 魅了 | Kanji | Miryō | Literally "charm/fascinate" |
| Pull | 引き寄せ | Kanji | Hikiyose | Native Japanese term |
| Root | 拘束 | Kanji | Kōsoku | Literally "restraint" |
| Silence | 沈黙 | Kanji | Chinmoku | Native Japanese term |
| Slow | スロウ | Katakana | Surō | Sometimes 減速 (gensoku) |
| Stun | スタン | Katakana | Sutan | Loanword from English; 100% consistency in dataset |
| Taunt | 挑発 | Kanji | Chōhatsu | Native Japanese term |

### Common Abbreviations

| English Abbreviation | Japanese Equivalent | Notes |
|----------------------|---------------------|-------|
| AoE | AoE or 範囲 | Both the English abbreviation and Japanese term (han'i) are used |
| CC | CC or 行動制限 | Both the English abbreviation (3.2% usage) and Japanese term (kōdō seigen) are used |
| CD | CD or クールダウン | Both the English abbreviation and full katakana term are used |
| DoT | DoT or 継続ダメージ | Both the English abbreviation and Japanese term (keizoku damēji) are used |
| DPS | DPS | English abbreviation is commonly used (0.1% usage) |
| HP | HP or ヒットポイント | Preferred over "体力" (tairyoku); 3.2% usage of HP abbreviation |
| MP | MP or マナ | Both the English abbreviation and Japanese term (mana) are used |

### Standardized UI Abbreviations

| Full Japanese Term | Abbreviated Form | Notes |
|-------------------|------------------|-------|
| 物理攻撃力 (Physical Power) | 物攻 | Introduce explicitly as "物理攻撃力（物攻）" |
| 魔法攻撃力 (Magical Power) | 魔攻 | Introduce explicitly as "魔法攻撃力（魔攻）" |
| 物理防御力 (Physical Protection) | 物防 | Introduce explicitly as "物理防御力（物防）" |
| 魔法防御力 (Magical Protection) | 魔防 | Introduce explicitly as "魔法防御力（魔防）" |
| クリティカル発生率 (Critical Chance) | クリ率 | Introduce explicitly as "クリティカル発生率（クリ率）" |
| 攻撃速度 (Attack Speed) | 攻速 | Introduce explicitly as "攻撃速度（攻速）" |
| 移動速度 (Movement Speed) | 移速 | Introduce explicitly as "移動速度（移速）" |
| クールダウン短縮 (Cooldown Reduction) | CD短縮 | Introduce explicitly as "クールダウン短縮（CD短縮）" |

## Implementation Guidelines

### Translation Process

1. Identify the type of content being translated (UI, ability description, lore, etc.)
2. Apply appropriate script selection rules
3. Reorganize sentence structure according to Japanese grammar
4. Apply appropriate particles
5. Handle articles and plurals appropriately
6. Use consistent terminology from the glossary
7. Format numbers and units according to Japanese conventions
8. Adapt cultural references as needed
9. Check for character limitations and apply abbreviations if necessary

### Quality Assurance

- Verify correct script usage (kanji, hiragana, katakana)
- Ensure proper particle usage
- Check for natural-sounding Japanese
- Verify that translations fit in the UI space allocated
- Confirm that the meaning is preserved accurately
- Ensure consistent terminology usage
- Verify appropriate politeness level
- Pay special attention to the consistent use of key terms identified in validation:
  - "God" → "ゴッド" for playable characters
  - "Health" → "HP" or "ヒットポイント"
  - "Shield" → "シールド"
  - "Ability" → "アビリティ"
  - "Ultimate" → "アルティメット"

## Validation Insights

Based on analysis of the dataset, these key findings should guide translation decisions:

### Terminology Consistency

| English Term | Expected Japanese | Consistency | Recommendation |
|-------------|-------------------|-------------|----------------|
| Ability | アビリティ | 79.4% | Standardize on アビリティ |
| Attack | 攻撃 | 98.7% | Continue using 攻撃 |
| Damage | ダメージ | 100.0% | Continue using ダメージ |
| Health | 体力 | 4.3% | Switch to HP or ヒットポイント |
| Mana | マナ | 100.0% | Continue using マナ |
| Cooldown | クールダウン | 100.0% | Continue using クールダウン |
| Ultimate | アルティメット | 83.3% | Standardize on アルティメット |
| God | 神 | 39.0% | Use ゴッド for playable characters |
| Stun | スタン | 100.0% | Continue using スタン |
| Shield | シールド | 54.5% | Standardize on シールド for gameplay mechanics |

### Script Distribution

The dataset shows a balanced distribution of Japanese writing systems:
- Kanji: 38.67%
- Hiragana: 21.05%
- Katakana: 40.28%

This distribution aligns with our script selection rules, showing appropriate use of all three Japanese writing systems.

### Abbreviation Usage

The dataset shows limited use of abbreviations:
- HP: 3.2% usage
- CC: 3.2% usage
- DPS: 0.1% usage

Japanese abbreviations (物攻, 魔攻, etc.) were not found in the dataset, suggesting a need to explicitly introduce these abbreviations when used.

By following this comprehensive ruleset, translators can ensure high-quality, consistent translations of SMITE 2 content from English to Japanese that will resonate with Japanese players while maintaining the game's core terminology and feel.
