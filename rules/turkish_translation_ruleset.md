# Turkish (trTR) Translation Ruleset for SMITE 2

This comprehensive ruleset provides guidelines and standardized terminology for translating SMITE 2 content into Turkish. It is designed to ensure consistency, accuracy, and cultural appropriateness across all game content.

## Table of Contents

1. [Special Translation Rules](#special-translation-rules)
   - [Grammatical Rules](#grammatical-rules)
   - [Formatting Rules](#formatting-rules)
   - [Terminology Rules](#terminology-rules)
   - [Cultural Adaptation Rules](#cultural-adaptation-rules)
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
   - [Pantheons](#pantheons)
   - [Standardized UI Abbreviations](#standardized-ui-abbreviations-for-space-constraints)
   - [Terms Requiring Standardization](#terms-requiring-standardization)

## Special Translation Rules

### Grammatical Rules

#### Suffix Application

1. **Vowel Harmony**
   - Follow Turkish vowel harmony rules when adding suffixes
   - Back vowels (a, ı, o, u) must be followed by suffixes with back vowels
   - Front vowels (e, i, ö, ü) must be followed by suffixes with front vowels
   - Example: "Damage" (Hasar) + locative → "Hasarda" (not "Hasarde")

2. **Case Usage**
   - **Accusative Case (-i, -ı, -u, -ü)**: Use for direct objects and ability targets (34.5% of cases in dataset)
     - Example: "Targets an enemy" → "Bir düşmanı hedefler"
   - **Dative Case (-e, -a)**: Use for direction or recipient of effects (26.4% of cases in dataset)
     - Example: "Applies to allies" → "Müttefiklere uygulanır"
   - **Genitive Case (-in, -ın, -un, -ün)**: Use for possession (13.8% of cases in dataset)
     - Example: "God's ability" → "Tanrının yeteneği"
   - **Nominative Case**: Use for subjects and predicates (10.7% of cases in dataset)
     - Example: "This ability stuns" → "Bu yetenek sersemletir"
   - **Locative Case (-de, -da)**: Use for location or area effects (9.4% of cases in dataset)
     - Example: "In the target area" → "Hedef alanda"
   - **Ablative Case (-den, -dan)**: Use for source of effects or movement away (5.2% of cases in dataset)
     - Example: "Escapes from crowd control" → "Kitle kontrolünden kaçar"

3. **Possessive Suffixes**
   - Use appropriate possessive suffixes based on the "owner"
   - For player-owned items/abilities, use second-person singular possessive (informal)
     - Example: "Your ability" → "Yeteneğin"
   - For god abilities, use third-person possessive
     - Example: "Thor's hammer" → "Thor'un çekici"

4. **Verbal Forms**
   - **Aorist (-ir, -ar)**: Use as primary verb form for ability descriptions (68.0% in dataset)
     - Example: "Stuns enemies" → "Düşmanları sersemletir"
   - **Past tense (-di, -dı, -du, -dü)**: Use for completed actions (25.2% in dataset)
     - Example: "Damage has been increased" → "Hasar arttırıldı"
   - **Present continuous (-iyor)**: Use for ongoing effects (4.2% in dataset)
     - Example: "Deals damage over time" → "Zamana yayılı hasar veriyor"
   - **Future (-ecek, -acak)**: Use for delayed or triggered effects (2.7% in dataset)
     - Example: "Will explode after 3s" → "3 saniye sonra patlayacak"

#### Word Order

1. **Ability Descriptions**
   - Follow Turkish Subject-Object-Verb order
   - Example: "This ability damages enemies" → "Bu yetenek düşmanlara hasar verir"

2. **Modifier Placement**
   - Place adjectives and modifiers before the nouns they modify
   - Example: "Physical damage" → "Fiziksel hasar"

3. **Postpositions**
   - Use postpositions (not prepositions) for spatial and temporal relationships
   - Example: "Through walls" → "Duvarların içinden"

#### Formality and Address

1. **Player Instructions**
   - Use informal "sen" form for direct player instructions (100% in dataset)
   - Example: "You gain health" → "Can kazanırsın"
   - Avoid formal "siz" form (0% in dataset)

2. **Game Information**
   - Use neutral third-person for general game information
   - Example: "This item provides protection" → "Bu eşya koruma sağlar"

### Formatting Rules

#### Capitalization

1. **Sentence Capitalization**
   - Use lowercase for most text (100% in dataset)
   - Only capitalize proper nouns (god names, pantheon names)
   - Example: "yıldırım çarpması düşmanları sersemletir" (lightning strike stuns enemies)

2. **Ability and Item Names**
   - Use lowercase for ability and item names, except for proper nouns
   - Example: "fırtına çekici" (storm hammer)

3. **Proper Nouns**
   - Always capitalize god names, pantheon names, and unique locations
   - Example: "Zeus", "Yunan", "Olympus"

#### Punctuation

1. **Number Formatting**
   - Use comma (,) as decimal separator (95.1% in dataset)
     - Example: "2,5 saniye" (2.5 seconds)
   - Use period (.) as thousands separator
     - Example: "1.000 hasar" (1,000 damage)

2. **Percentages**
   - Place percentage sign after the number with a space
     - Example: "25 %" (25%)

3. **Units**
   - Place unit abbreviations after the number with a space
     - Example: "50 m" (50m)

4. **Lists**
   - Use semicolons (;) to separate items in a list
   - Place "ve" (and) before the final item without a semicolon
   - Example: "güç; koruma; ve hız" (power; protection; and speed)

#### Abbreviations

1. **Standard Abbreviations**
   - Use "sn." for "saniye" (second)
   - Use "dk." for "dakika" (minute)
   - Use "m" for "metre" (meter)

2. **Game-Specific Abbreviations**
   - Use "SH" for "Saldırı Hızı" (Attack Speed)
   - Use "HH" for "Hareket Hızı" (Movement Speed)
   - Use "FH" for "Fiziksel Hasar" (Physical Damage)
   - Use "BH" for "Büyüsel Hasar" (Magical Damage)
   - Use "FK" for "Fiziksel Koruma" (Physical Protection)
   - Use "BK" for "Büyüsel Koruma" (Magical Protection)

3. **UI Space Constraints**
   - When space is limited, use established abbreviations
   - For extremely limited space, use first letters of compound terms
     - Example: "Hareket Hızı" → "HH"

### Terminology Rules

#### Standardization

1. **Core Gameplay Terms**
   - Standardize key gameplay terms across all contexts based on validation findings:
     - "Damage" → "Hasar" (83.1% consistent)
     - "Health" → "Can" (83.5% consistent)
     - "Mana" → "Mana" (89.8% consistent)
     - "Ability" → "Yetenek" (47.5% consistent, needs standardization)
     - "Ultimate" → "Ulti" (68.9% consistent)
     - "Passive" → "Pasif" (98.5% consistent)
     - "Cooldown" → "Bekleme Süresi" (68.8% consistent)
     - "Protection" → "Koruma" (81.3% consistent)
     - "Penetration" → "Delme" (0% consistent, needs standardization)
     - "Critical" → "Kritik" (94.7% consistent)
     - "Movement Speed" → "Hareket Hızı" (89.6% consistent)
     - "Attack Speed" → "Saldırı Hızı" (85.1% consistent)
     - "God" → "Tanrı" (male) or "Tanrıça" (female) (33.3% consistent, needs standardization)

2. **Crowd Control Effects**
   - Standardize crowd control terminology based on validation findings:
     - "Crowd Control" → "Kitle Kontrolü" (0% consistent, needs standardization)
     - "Stun" → "Sersemletme" (13.3% consistent, needs standardization)
     - "Root" → "Sabitleme" (0% consistent, needs standardization)
     - "Silence" → "Susturma" (7.7% consistent, needs standardization)
     - "Slow" → "Yavaşlatma" (38.4% consistent, needs standardization)
     - "Knockup" → "Havaya Savurma"
     - "Knockback" → "Geri Savurma"
     - "Mesmerize" → "Büyüleme"
     - "Fear" → "Korkutma"
     - "Taunt" → "Kışkırtma"
     - "Madness" → "Çıldırtma"
     - "Disarm" → "Silahsızlandırma"
     - "Blind" → "Körleştirme"
     - "Cripple" → "Sakatlama"

3. **Buff/Debuff Terminology**
   - Standardize buff/debuff terminology based on validation findings:
     - "Buff" → "Güçlendirme" (18.4% consistent, needs standardization)
     - "Debuff" → "Zayıflatma" (0% consistent, needs standardization)

#### Borrowing and Adaptation

1. **English Gaming Terms**
   - For established gaming terms without Turkish equivalents, use the English term with Turkish suffixes
   - Example: "meta" → "meta", "nerf" → "nerf"
   - Note: For terms like "buff" (18.4% "güçlendirme" vs. alternative "buff"), prioritize the Turkish equivalent "güçlendirme" for consistency

2. **Hybrid Terms**
   - For some technical terms, use hybrid formations combining borrowed and native elements
   - Example: "Critical Strike" → "Kritik Vuruş"

3. **New Terminology**
   - When creating new terms, prefer descriptive Turkish compounds over direct translations
   - Example: "Lifesteal" → "Can Çalma" (literally "health stealing")

#### Consistency Across Contexts

1. **Ability Descriptions**
   - Use consistent terminology between similar abilities across different gods
   - Example: All leap abilities should use the same Turkish term for "leap"

2. **Item Descriptions**
   - Use consistent terminology for item effects that are similar
   - Example: All items providing attack speed should use "Saldırı Hızı"

3. **UI Elements**
   - Maintain consistent terminology between UI elements and their descriptions
   - Example: The same term used in tooltips should appear in menus

### Cultural Adaptation Rules

#### Mythological References

1. **God Names**
   - Keep original god names but apply Turkish spelling conventions where appropriate
   - Example: "Zeus" remains "Zeus", "Anubis" remains "Anubis"

2. **Pantheon Names**
   - Use established Turkish names for pantheons
   - Example: "Greek" → "Yunan", "Egyptian" → "Mısır"

3. **Mythological Concepts**
   - Translate mythological concepts using Turkish equivalents when they exist
   - Example: "Underworld" → "Yeraltı Dünyası"
   - For culture-specific concepts without direct equivalents, provide contextual explanation
   - Example: "Ragnarok" might be kept as "Ragnarok" but explained as "İskandinav kıyameti" (Norse apocalypse)

#### Cultural References and Humor

1. **Wordplay and Puns**
   - Adapt wordplay to maintain humor rather than translating literally
   - Create new Turkish wordplay that preserves the spirit of the original

2. **Cultural References**
   - Replace culture-specific references with Turkish equivalents when necessary
   - Maintain original references when they are widely recognized in Turkey

### Technical Constraints

#### Character Limits

1. **UI Space Management**
   - For space-constrained UI elements, use abbreviated forms
   - Prioritize critical gameplay information over flavor text
   - Leverage Turkish agglutination to reduce word count where possible

2. **Tooltip Optimization**
   - Structure tooltips to fit information efficiently
   - Use standardized abbreviations consistently

#### Font and Display

1. **Turkish Characters**
   - Ensure proper display of Turkish-specific characters (ç, ğ, ı, ö, ş, ü)
   - Test all text in context to verify correct rendering

2. **Text Expansion**
   - Account for potential text expansion in Turkish (typically 20-30% longer than English)
   - Design UI elements with sufficient space for Turkish text

### Exception Handling

#### Untranslatable Elements

1. **Brand Terms**
   - Keep SMITE-specific branded terms untranslated but apply Turkish grammatical rules
   - Example: "SMITE" remains "SMITE" but may take Turkish suffixes as needed

2. **Technical Terms Without Equivalents**
   - For technical terms without established Turkish equivalents, maintain the English term
   - Apply Turkish suffixes according to vowel harmony rules
   - Example: "meta" → "meta", "meta'ya" (to the meta)

#### Ambiguous Terms

1. **Context-Dependent Translation**
   - Translate terms differently based on context when necessary
   - Example: "Power" might be "Güç" in general contexts but "Kuvvet" in specific mechanical contexts

2. **Clarification**
   - Add clarifying words when the Turkish translation might be ambiguous
   - Example: "Physical" might need specification as "Fiziksel (saldırı)" to distinguish from other uses

### Implementation Guidelines

#### Quality Assurance

1. **Consistency Check**
   - Verify terminology consistency across similar abilities and items
   - Ensure grammatical consistency in similar constructions
   - Pay special attention to terms identified with low consistency in validation:
     - "Penetration" (0% consistent)
     - "Root" (0% consistent)
     - "Crowd Control" (0% consistent)
     - "Debuff" (0% consistent)
     - "Silence" (7.7% consistent)
     - "Stun" (13.3% consistent)
     - "Buff" (18.4% consistent)

2. **Context Review**
   - Review translations in-game to ensure they make sense in context
   - Check that UI elements display correctly with Turkish text

3. **Native Review**
   - Have translations reviewed by native Turkish speakers familiar with gaming terminology
   - Incorporate feedback on natural-sounding phrasing

#### Maintenance

1. **Terminology Database**
   - Maintain a database of standardized Turkish terms for future updates
   - Document any terminology changes for reference

2. **Feedback Integration**
   - Incorporate player feedback on translations when appropriate
   - Prioritize clarity and gameplay understanding over literal translation

## Game-Specific Glossary

### Core Game Concepts

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| God | Tanrı | Use for male gods (33.3% consistent, needs standardization) |
| Goddess | Tanrıça | Use for female gods |
| Ability | Yetenek | Standard translation (47.5% consistent, needs standardization) |
| Ultimate | Ulti | Short form preferred in UI (68.9% consistent) |
| Passive | Pasif | Borrowed term with Turkish pronunciation (98.5% consistent) |
| Active | Aktif | Borrowed term with Turkish pronunciation |
| Item | Eşya | Standard translation |
| Relic | Emanet | Native Turkish term |
| Consumable | Tüketilebilir | Native Turkish term |
| Level | Seviye | Standard translation |
| Experience | Deneyim | Standard translation |
| Gold | Altın | Standard translation |
| Lane | Koridor | Literally "corridor" |
| Jungle | Orman | Literally "forest" |
| Minion | Minyon | Borrowed with Turkish pronunciation |
| Tower | Kule | Standard translation |
| Phoenix | Anka Kuşu | Native Turkish term for mythological phoenix |
| Titan | Titan | Direct borrowing |

### Character Classes & Roles

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Warrior | Savaşçı | Standard translation |
| Guardian | Koruyucu | Standard translation |
| Mage | Büyücü | Standard translation |
| Hunter | Avcı | Standard translation |
| Assassin | Suikastçı | Standard translation |
| Support | Destek | Standard translation |
| Carry | Taşıyıcı | Literal translation of "carrier" |
| Solo | Solo | Direct borrowing |
| Mid | Orta | Literally "middle" |
| Jungle (role) | Ormancı | Person who operates in the jungle |

### Combat Terminology

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Damage | Hasar | Standard translation (83.1% consistent) |
| Physical Damage | Fiziksel Hasar | Compound with standard "Hasar" |
| Magical Damage | Büyüsel Hasar | Compound with standard "Hasar" |
| True Damage | Gerçek Hasar | Compound with standard "Hasar" |
| Health | Can | Standard translation (83.5% consistent) |
| Mana | Mana | Direct borrowing (89.8% consistent) |
| Protection | Koruma | Standard translation (81.3% consistent) |
| Physical Protection | Fiziksel Koruma | Compound with standard "Koruma" |
| Magical Protection | Büyüsel Koruma | Compound with standard "Koruma" |
| Penetration | Delme | Standard translation (0% consistent, needs standardization) |
| Flat Penetration | Sabit Delme | Compound with standard "Delme" |
| Percentage Penetration | Yüzde Delme | Compound with standard "Delme" |
| Critical | Kritik | Borrowed with Turkish pronunciation (94.7% consistent) |
| Critical Strike | Kritik Vuruş | Hybrid compound |
| Critical Chance | Kritik Şansı | Hybrid compound |
| Lifesteal | Can Çalma | Literal translation "health stealing" |
| Healing | İyileştirme | Standard translation |
| Cooldown | Bekleme Süresi | Literally "waiting time" (68.8% consistent) |
| Cooldown Reduction | Bekleme Süresi Azaltma | Compound with standard "Bekleme Süresi" |
| Attack Speed | Saldırı Hızı | Standard translation (85.1% consistent) |
| Movement Speed | Hareket Hızı | Standard translation (89.6% consistent) |
| Range | Menzil | Standard translation |
| Area of Effect | Etki Alanı | Standard translation |
| Projectile | Mermi | Standard translation |

### Crowd Control Effects

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Crowd Control | Kitle Kontrolü | Literal translation (0% consistent, needs standardization) |
| Stun | Sersemletme | Standard translation (13.3% consistent, needs standardization) |
| Root | Sabitleme | Literally "fixing in place" (0% consistent, needs standardization) |
| Silence | Susturma | Standard translation (7.7% consistent, needs standardization) |
| Slow | Yavaşlatma | Standard translation (38.4% consistent, needs standardization) |
| Cripple | Sakatlama | Standard translation |
| Knockup | Havaya Savurma | Descriptive: "throwing into air" |
| Knockback | Geri Savurma | Descriptive: "throwing back" |
| Pull | Çekme | Standard translation |
| Mesmerize | Büyüleme | Standard translation |
| Fear | Korkutma | Standard translation |
| Taunt | Kışkırtma | Standard translation |
| Madness | Çıldırtma | Standard translation |
| Disarm | Silahsızlandırma | Standard translation |
| Blind | Körleştirme | Standard translation |
| Intoxicate | Sarhoş Etme | Standard translation |
| Banish | Sürgün Etme | Standard translation |

### Map Elements

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Map | Harita | Standard translation |
| Base | Üs | Standard translation |
| Fountain | Çeşme | Standard translation |
| Buff Camp | Güçlendirme Kampı | Compound with "güçlendirme" (18.4% consistent, needs standardization) |
| Red Buff | Kırmızı Güçlendirme | Compound with "güçlendirme" |
| Blue Buff | Mavi Güçlendirme | Compound with "güçlendirme" |
| Yellow Buff | Sarı Güçlendirme | Compound with "güçlendirme" |
| Purple Buff | Mor Güçlendirme | Compound with "güçlendirme" |
| Fire Giant | Ateş Devi | Standard translation |
| Gold Fury | Altın Öfkesi | Standard translation |
| Pyromancer | Ateş Büyücüsü | Standard translation |
| Harpy | Harpi | Borrowed with Turkish pronunciation |
| Jungle Boss | Orman Patronu | Standard translation |
| Ward | Gözcü | Literally "watcher" |
| Sentry Ward | Nöbetçi Gözcü | Compound with "gözcü" |

### Game Modes

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Conquest | Fetih | Standard translation |
| Arena | Arena | Direct borrowing |
| Joust | Düello | Standard translation |
| Assault | Saldırı | Standard translation |
| Clash | Çatışma | Standard translation |
| Siege | Kuşatma | Standard translation |
| MOTD | MOTD | Keep as acronym (Match of the Day) |
| Ranked | Dereceli | Standard translation |
| Casual | Sıradan | Standard translation |
| Custom | Özel | Standard translation |

### UI Elements

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Menu | Menü | Borrowed with Turkish pronunciation |
| Settings | Ayarlar | Standard translation |
| Store | Mağaza | Standard translation |
| Profile | Profil | Borrowed with Turkish pronunciation |
| Friends | Arkadaşlar | Standard translation |
| Clan | Klan | Borrowed with Turkish pronunciation |
| Match History | Maç Geçmişi | Standard translation |
| Leaderboard | Lider Tablosu | Standard translation |
| Queue | Sıra | Standard translation |
| Lobby | Lobi | Borrowed with Turkish pronunciation |
| Loading Screen | Yükleme Ekranı | Standard translation |
| Victory | Zafer | Standard translation |
| Defeat | Yenilgi | Standard translation |
| Surrender | Teslim Ol | Standard translation |
| Pause | Duraklat | Standard translation |
| Chat | Sohbet | Standard translation |
| Scoreboard | Skor Tablosu | Hybrid compound |

### Stats and Attributes

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Power | Güç | Standard translation |
| Physical Power | Fiziksel Güç | Compound with "güç" |
| Magical Power | Büyüsel Güç | Compound with "güç" |
| Defense | Savunma | Standard translation |
| Armor | Zırh | Standard translation |
| Shield | Kalkan | Standard translation |
| Regeneration | Yenilenme | Standard translation |
| Health Regeneration | Can Yenilenmesi | Compound with "can" |
| Mana Regeneration | Mana Yenilenmesi | Compound with "mana" |
| Aura | Aura | Direct borrowing |
| Buff | Güçlendirme | Native term (18.4% consistent, needs standardization) |
| Debuff | Zayıflatma | Standard translation (0% consistent, needs standardization) |
| Immunity | Bağışıklık | Standard translation |
| Resistance | Direnç | Standard translation |
| Mitigation | Azaltma | Standard translation |

### Ability Mechanics

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Cast | Kullanma | Standard translation |
| Channel | Kanalize Etme | Hybrid with Turkish verb |
| Charge | Şarj Etme | Hybrid with Turkish verb |
| Dash | Atılma | Standard translation |
| Leap | Sıçrama | Standard translation |
| Teleport | Işınlanma | Standard translation |
| Execute | İnfaz | Standard translation |
| Heal | İyileştirme | Standard translation |
| Shield | Kalkan | Standard translation |
| Stealth | Görünmezlik | Standard translation |
| Reveal | Açığa Çıkarma | Standard translation |
| Cleanse | Arındırma | Standard translation |
| Reset | Sıfırlama | Standard translation |
| Stack | Yığın | Standard translation |
| Proc | Tetikleme | Standard translation |
| Toggle | Aç/Kapat | Standard translation |
| Stance | Duruş | Standard translation |

### Item Categories

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Starter Item | Başlangıç Eşyası | Standard translation |
| Boots | Botlar | Standard translation |
| Offensive | Saldırı | Standard translation |
| Defensive | Savunma | Standard translation |
| Utility | Yardımcı | Standard translation |
| Aura Item | Aura Eşyası | Hybrid compound |
| Consumable | Tüketilebilir | Standard translation |
| Potion | İksir | Standard translation |
| Ward | Gözcü | Standard translation |
| Relic | Emanet | Standard translation |
| Tier 1/2/3 | Kademe 1/2/3 | Standard translation |

### Common Abbreviations

| English Abbreviation | Turkish Translation | Notes |
|----------------------|---------------------|-------|
| HP | CAN | Standard abbreviation |
| MP | MANA | Standard abbreviation |
| CC | KK (Kitle Kontrolü) | Standard abbreviation |
| CDR | BSA (Bekleme Süresi Azaltma) | Standard abbreviation |
| DoT | ZYH (Zamana Yayılı Hasar) | Standard abbreviation |
| AoE | EA (Etki Alanı) | Standard abbreviation |
| AS | SH (Saldırı Hızı) | Standard abbreviation |
| MS | HH (Hareket Hızı) | Standard abbreviation |
| DPS | SBH (Saniyede Birim Hasar) | Standard abbreviation |
| AA | TA (Temel Atak) | Standard abbreviation |

### Specialized Gaming Terms

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Meta | Meta | Direct borrowing |
| Gank | Baskın | Standard translation |
| Farm | Kesmek | Literally "to cut" |
| Poke | Dürtmek | Standard translation |
| Burst | Patlatmak | Standard translation |
| Sustain | Dayanıklılık | Standard translation |
| Rotation | Rotasyon | Borrowed with Turkish pronunciation |
| Invade | İstila Etmek | Standard translation |
| Secure | Güvenceye Almak | Standard translation |
| Snowball | Kartopu Etkisi | Literal translation |
| Teamfight | Takım Savaşı | Standard translation |
| Splitpush | Bölünmüş İtiş | Descriptive translation |
| Backdoor | Arka Kapı | Literal translation |
| Dive | Dalış | Standard translation |
| Peel | Koruma | Standard translation |
| Kite | Kaçırmak | Standard translation |
| Juke | Çalım | Standard translation |
| Bait | Yem | Standard translation |

### Pantheons

| English Term | Turkish Translation | Notes |
|--------------|---------------------|-------|
| Greek | Yunan | Standard translation |
| Roman | Roma | Standard translation |
| Norse | İskandinav | Standard translation |
| Egyptian | Mısır | Standard translation |
| Chinese | Çin | Standard translation |
| Hindu | Hindu | Direct borrowing |
| Mayan | Maya | Direct borrowing |
| Japanese | Japon | Standard translation |
| Celtic | Kelt | Borrowed with Turkish pronunciation |
| Slavic | Slav | Borrowed with Turkish pronunciation |
| Voodoo | Vudu | Borrowed with Turkish pronunciation |
| Polynesian | Polinezya | Borrowed with Turkish pronunciation |
| Yoruba | Yoruba | Direct borrowing |
| Arthurian | Arthurian | Direct borrowing |
| Babylonian | Babil | Standard translation |

### Standardized UI Abbreviations for Space Constraints

| Full Turkish Term | Abbreviated Form | Notes |
|-------------------|------------------|-------|
| Can | CAN | For UI space constraints |
| Mana | MANA | For UI space constraints |
| Hasar | HSR | For UI space constraints |
| Koruma | KRM | For UI space constraints |
| Saldırı Hızı | SH | For UI space constraints |
| Hareket Hızı | HH | For UI space constraints |
| Bekleme Süresi | BS | For UI space constraints |
| Delme | DLM | For UI space constraints |
| Kitle Kontrolü | KK | For UI space constraints |
| Güçlendirme | GÇL | For UI space constraints |
| Zayıflatma | ZYF | For UI space constraints |

### Terms Requiring Standardization

Based on validation findings, the following terms have low consistency in the dataset and require particular attention for standardization:

| English Term | Recommended Turkish Translation | Current Consistency | Notes |
|--------------|--------------------------------|---------------------|-------|
| Penetration | Delme | 0.0% | Must be standardized across all contexts |
| Root | Sabitleme | 0.0% | Must be standardized across all contexts |
| Debuff | Zayıflatma | 0.0% | Must be standardized across all contexts |
| Crowd Control | Kitle Kontrolü | 0.0% | Must be standardized across all contexts |
| Silence | Susturma | 7.7% | Must be standardized across all contexts |
| Stun | Sersemletme | 13.3% | Must be standardized across all contexts |
| Buff | Güçlendirme | 18.4% | Must be standardized across all contexts |
| God | Tanrı/Tanrıça | 33.3% | Use gender-appropriate form |
| Slow | Yavaşlatma | 38.4% | Must be standardized across all contexts |
| Ability | Yetenek | 47.5% | Must be standardized across all contexts |
