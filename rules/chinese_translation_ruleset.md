# Chinese (zhCN) Translation Ruleset for SMITE 2

This document provides comprehensive guidelines for translating SMITE 2 content into Chinese (zhCN). It includes special translation rules and a game-specific glossary to ensure consistency, accuracy, and cultural appropriateness.

## Special Translation Rules

### 1. Grammatical Rules

#### 1.1 Sentence Structure
- Maintain Subject-Verb-Object (SVO) structure when possible for clarity
- Use topic-prominent structures for emphasis (e.g., "这个技能，它可以..." instead of "这个技能可以...")
- Prefer active voice (97.8% of translations use active voice) over passive voice
- Add explicit subjects when needed for clarity (e.g., add "玩家" or "你" when the English source implies but doesn't state the subject)

#### 1.2 Measure Words
- Always use appropriate measure words after numbers:
  - 秒 (second): For time durations (e.g., "5秒冷却时间")
  - 点 (point): For damage, health, mana values (e.g., "50点伤害")
  - 层 (layer): For stacks and effect levels (e.g., "3层效果")
  - 个 (general): For counting objects without specific measure words (e.g., "3个敌人")
  - 条 (strip): For elongated objects like dragons (e.g., "2条龙")
- Never insert spaces between numbers and measure words

#### 1.3 Verb Forms
- Use present tense for ability descriptions
- Use imperative form for player instructions
- Add modal verbs (可以, 能够, 将会) to indicate possibility, ability, or future actions

#### 1.4 Noun Phrases
- Place modifiers before the noun they modify
- Use 的 particle sparingly to avoid awkward constructions
- For compound terms, maintain the modifier-modified structure (e.g., "物理防护" for "Physical Protection")

### 2. Formatting Rules

#### 2.1 Punctuation
- Use Chinese punctuation instead of Western punctuation:
  - 。 instead of .
  - ， instead of ,
  - ： instead of :
  - ； instead of ;
  - "" instead of ""
  - （） instead of ()
  - ！ instead of !
  - ？ instead of ?
- Maintain Western punctuation only for technical notations (e.g., skill ratios, version numbers)

#### 2.2 Number Formatting
- Use Arabic numerals (0-9) for all numerical values
- Do not insert spaces between numbers and units or percentage signs
- Use Western decimal point (.) for decimal numbers
- Use comma (,) as thousands separator for large numbers
- Place plus sign (+) directly before numbers without spaces

#### 2.3 Capitalization
- Capitalize proper nouns including God names, item names, and ability names
- Do not capitalize common nouns unless they are part of a proper name
- Maintain capitalization in acronyms (e.g., "AOE", "CC")

#### 2.4 Text Formatting
- Maintain consistent font styles between English and Chinese
- Adjust spacing to accommodate Chinese characters (which occupy more visual space)
- Ensure UI elements have sufficient space for Chinese text (which is often more compact than English)

### 3. Terminology Rules

#### 3.1 Game-Specific Terms
- Standardize key terminology with high consistency (≥95%):
  - Ability → 技能 (95.3%)
  - Attack → 攻击 (96.4%)
  - Item → 物品 (97.4%)
  - Basic Attack → 普通攻击 (97.8%)
  - Damage → 伤害 (99.2%)
  - Passive → 被动 (99.2%)
  - Physical → 物理 (99.7%)
  - Movement Speed → 移动速度 (100%)
  - Phoenix → 凤凰 (100%)
  - Penetration → 穿透 (100%)

- Standardize terms with low consistency (<70%):
  - Protection → 防护 (currently 0% consistency)
  - Stun → 眩晕 (currently 0% consistency)
  - Crowd Control → 群体控制 (currently 0% consistency)
  - Jungle → 丛林 (currently 2.1% consistency)
  - Minion → 小兵 (currently 2.9% consistency)
  - Level → 等级 (currently 23.1% consistency)
  - Lane → 路线 (currently 24.7% consistency)
  - Debuff → 减益 (currently 40.0% consistency)
  - Kill → 击杀 (currently 42.9% consistency)
  - Buff → 增益 (currently 44.4% consistency)

#### 3.2 Translation Approaches
- Use semantic translations for most gameplay mechanics:
  - Cooldown → 冷却时间
  - Health → 生命值
  - Damage → 伤害
  - Movement Speed → 移动速度

- Use phonetic transliterations for established gaming terms:
  - Tank → 坦克
  - Mana → 法力值 (not 玛娜, which has lower consistency)

- Use hybrid approaches for certain terms:
  - Basic Attack → 普通攻击 (not 平A, despite common usage in gaming)
  - Critical Strike → 暴击

- Retain English abbreviations for technical terms:
  - CC (Crowd Control) → CC
  - AOE (Area of Effect) → AOE

#### 3.3 Consistency Guidelines
- Refer to the glossary for standardized translations
- Maintain consistency within related term families (e.g., all damage types)
- When multiple translations exist, prefer the one with highest consistency in the dataset
- For new terms, follow the established pattern of similar existing terms

### 4. Cultural Adaptation

#### 4.1 Formality Levels
- Use standard language (你) for gameplay instructions (90.2% of translations)
- Use formal language (您) for system messages and tutorials (9.8% of translations)
- Match character dialogue to the personality of the character
- Avoid overly casual expressions in official instructions

#### 4.2 Cultural References
- Adapt mythological references to Chinese cultural equivalents when appropriate
- Preserve original cultural context when translating God lore and backgrounds
- Use Chinese idioms and expressions when they enhance understanding
- Avoid literal translations that may lose cultural meaning

#### 4.3 Localization Considerations
- Adapt humor to suit Chinese cultural preferences
- Consider Chinese gaming community terminology and preferences
- Ensure translations are politically and culturally appropriate for the Chinese market
- Avoid sensitive topics or references that may be problematic in China

### 5. Exception Handling

#### 5.1 Untranslatable Elements
- Do not translate proper names unless they have established Chinese translations
- Maintain original formatting for code elements, commands, and technical identifiers
- Keep trademark symbols (™, ®) and copyright notices in their original form
- Preserve numerical skill ratios and formulas exactly as in the source

#### 5.2 UI Constraints
- Abbreviate translations when necessary to fit UI space constraints
- For extremely limited space, use established Chinese gaming abbreviations
- Ensure critical gameplay information is never lost due to space constraints
- When abbreviating, prioritize clarity over literal translation

#### 5.3 Contextual Adaptation
- Adapt translations based on in-game context (combat, store, tutorial)
- Consider the target audience's familiarity with MOBA terminology
- Provide additional context in tutorials for new players
- Use more technical terminology for advanced mechanics descriptions

### 6. Implementation Guidelines

#### 6.1 Quality Assurance
- Verify terminology consistency across all game content
- Check for grammatical accuracy and natural-sounding Chinese
- Ensure proper rendering of Chinese characters in all game fonts
- Test translations in context to verify meaning and clarity

#### 6.2 Maintenance
- Document all translation decisions for future reference
- Maintain a living glossary that evolves with the game
- Establish a feedback mechanism for players to report translation issues
- Regularly review and update translations based on player feedback

## Game-Specific Glossary

### Core Game Concepts

| English Term | Chinese Translation | Notes |
|--------------|---------------------|-------|
| SMITE 2 | SMITE 2 | Do not translate game title |
| God | 神 | 93.0% consistency |
| Ability | 技能 | 95.3% consistency |
| Passive | 被动 | 99.2% consistency |
| Ultimate | 终极技能 | 65.6% consistency |
| Level | 等级 | 23.1% consistency, standardize to 等级 |
| Experience | 经验 | 60.0% consistency |
| Gold | 金币 | 45.8% consistency, standardize to 金币 |
| Item | 物品 | 97.4% consistency |
| Store | 商店 | |
| Match | 比赛 | |
| Victory | 胜利 | |
| Defeat | 失败 | |
| Team | 团队 | |
| Ally | 盟友 | |
| Enemy | 敌人 | |
| Minion | 小兵 | 2.9% consistency, standardize to 小兵 |
| Jungle | 丛林 | 2.1% consistency, standardize to 丛林 |
| Lane | 路线 | 24.7% consistency, standardize to 路线 |
| Map | 地图 | |
| Fountain | 泉水 | |
| Base | 基地 | |

### Character Classes & Roles

| English Term | Chinese Translation | Notes |
|--------------|---------------------|-------|
| Assassin | 刺客 | |
| Guardian | 守护者 | |
| Hunter | 猎人 | |
| Mage | 法师 | |
| Warrior | 战士 | |
| Support | 辅助 | |
| Carry | 后期核心 | |
| Solo | 单人路 | |
| Jungler | 打野 | |
| Mid | 中路 | |
| Tank | 坦克 | Phonetic transliteration |

### Combat Terminology

| English Term | Chinese Translation | Notes |
|--------------|---------------------|-------|
| Damage | 伤害 | 99.2% consistency |
| Physical | 物理 | 99.7% consistency |
| Magical | 魔法 | 99.5% consistency |
| Attack | 攻击 | 96.4% consistency |
| Basic Attack | 普通攻击 | 97.8% consistency |
| Health | 生命值 | 93.9% consistency |
| Mana | 法力值 | 80.5% consistency |
| Protection | 防护 | 0.0% consistency, standardize to 防护 |
| Penetration | 穿透 | 100.0% consistency |
| Critical | 暴击 | 94.7% consistency |
| Lifesteal | 生命偷取 | 98.0% consistency |
| Healing | 治疗 | 84.2% consistency |
| Shield | 护盾 | 63.0% consistency, standardize to 护盾 |
| Movement Speed | 移动速度 | 100.0% consistency |
| Attack Speed | 攻击速度 | |
| Cooldown | 冷却时间 | 85.9% consistency |
| Range | 范围 | 64.9% consistency, standardize to 范围 |
| Area of Effect | 范围效果 | Also abbreviated as AOE |
| Duration | 持续时间 | 65.4% consistency, standardize to 持续时间 |
| Kill | 击杀 | 42.9% consistency, standardize to 击杀 |
| Death | 死亡 | 64.8% consistency, standardize to 死亡 |
| Assist | 助攻 | |
| Buff | 增益 | 44.4% consistency, standardize to 增益 |
| Debuff | 减益 | 40.0% consistency, standardize to 减益 |

### Crowd Control Effects

| English Term | Chinese Translation | Notes |
|--------------|---------------------|-------|
| Crowd Control | 群体控制 | 0.0% consistency, standardize to 群体控制 |
| Stun | 眩晕 | 0.0% consistency, standardize to 眩晕 |
| Slow | 减速 | 93.9% consistency |
| Root | 定身 | |
| Silence | 沉默 | |
| Disarm | 缴械 | |
| Cripple | 残废 | |
| Knockup | 击飞 | |
| Knockback | 击退 | |
| Pull | 拉拽 | |
| Mesmerize | 迷惑 | |
| Taunt | 嘲讽 | |
| Fear | 恐惧 | |
| Intoxicate | 醉酒 | |
| Madness | 疯狂 | |
| Blind | 致盲 | |

### Map Elements

| English Term | Chinese Translation | Notes |
|--------------|---------------------|-------|
| Tower | 防御塔 | 82.5% consistency |
| Phoenix | 凤凰 | 100.0% consistency |
| Titan | 泰坦 | 86.5% consistency |
| Jungle Camp | 野怪营地 | |
| Buff Camp | 增益营地 | |
| Ward | 守卫 | 87.9% consistency |
| Fire Giant | 火焰巨人 | |
| Gold Fury | 黄金魔兽 | |
| Pyromancer | 火术士 | |
| Harpy | 鹰身女妖 | |
| Scorpion | 蝎子 | |
| Obelisk | 方尖碑 | |
| Portal | 传送门 | |

### Game Modes

| English Term | Chinese Translation | Notes |
|--------------|---------------------|-------|
| Conquest | 征服 | |
| Arena | 竞技场 | |
| Joust | 决斗 | |
| Assault | 突袭 | |
| Clash | 冲突 | |
| Siege | 围攻 | |
| MOTD | 每日模式 | "Match of the Day" |
| Custom | 自定义 | |
| Training | 训练 | |
| Co-op | 合作 | |
| Ranked | 排名赛 | |
| Casual | 休闲赛 | |

### UI Elements

| English Term | Chinese Translation | Notes |
|--------------|---------------------|-------|
| Menu | 菜单 | |
| Settings | 设置 | |
| Profile | 个人资料 | |
| Friends | 好友 | |
| Clan | 公会 | |
| Achievements | 成就 | |
| Quests | 任务 | |
| Battle Pass | 战斗通行证 | |
| Store | 商店 | |
| Chest | 宝箱 | |
| Bundle | 捆绑包 | |
| Skin | 皮肤 | |
| Voice Pack | 语音包 | |
| Loading Frame | 加载框架 | |
| Loading Screen | 加载画面 | |
| Avatar | 头像 | |
| Emote | 表情 | |
| Announcer Pack | 播报员包 | |

### Common Abbreviations

| English Abbreviation | Chinese Translation | Notes |
|----------------------|---------------------|-------|
| CC | CC | Retain English abbreviation |
| AOE | AOE | Retain English abbreviation |
| DOT | DOT | "Damage Over Time" |
| HP | 生命值 | Expand to full term in Chinese |
| MP | 法力值 | Expand to full term in Chinese |
| CD | 冷却 | "Cooldown" |
| CDR | 冷却缩减 | "Cooldown Reduction" |
| MS | 移速 | "Movement Speed" |
| AS | 攻速 | "Attack Speed" |
| AA | 普攻 | "Auto Attack" / "Basic Attack" |
| AP | 法术强度 | "Ability Power" |
| AD | 物理强度 | "Attack Damage" |
| KDA | 击杀/死亡/助攻 | "Kills/Deaths/Assists" |

### Standardized UI Abbreviations (for space constraints)

| Full Chinese Term | Abbreviated Form | Notes |
|-------------------|------------------|-------|
| 生命值 | 生命 | For extremely limited UI space |
| 法力值 | 法力 | For extremely limited UI space |
| 冷却时间 | 冷却 | For extremely limited UI space |
| 攻击速度 | 攻速 | For extremely limited UI space |
| 移动速度 | 移速 | For extremely limited UI space |
| 物理防护 | 物防 | For extremely limited UI space |
| 魔法防护 | 魔防 | For extremely limited UI space |
| 物理穿透 | 物穿 | For extremely limited UI space |
| 魔法穿透 | 魔穿 | For extremely limited UI space |
| 群体控制 | CC | For extremely limited UI space |
| 每秒伤害 | DPS | For extremely limited UI space |
