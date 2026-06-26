# RT-130 / FACT-001 factual gap panel — TinyLlama-1.1B

27 prompts, rep-penalty 1.2 primary (greedy/sampling diagnostic for adapted i2_s).
fact_rate = contains-match of expected answer in the answer slot. NOT a benchmark.

| variant | decode | fact_rate | hits | tags |
| --- | --- | ---: | ---: | --- |
| FP f16 | rep1.2 | 0.815 | 22/27 | {'ok': 27} |
| Q2_K | rep1.2 | 0.741 | 20/27 | {'ok': 27} |
| PTQ i2_s | rep1.2 | 0.0 | 0/27 | {'salad': 24, 'ok': 3} |
| adapted f16 | rep1.2 | 0.185 | 5/27 | {'ok': 27} |
| adapted i2_s | rep1.2 | 0.148 | 4/27 | {'ok': 26, 'repetitive': 1} |
| adapted i2_s | greedy | 0.111 | 3/27 | {'repetitive': 15, 'ok': 12} |
| adapted i2_s | t0.8p0.95 | 0.148 | 4/27 | {'ok': 26, 'salad': 1} |

adapted i2_s vs f16 (rep1.2) hit-agreement: 24/27

## sample answers (rep1.2)
**cap_france** (expect ['paris'])
- FP f16 [HIT]: 'The capital of France is Paris. [end of text]'
- Q2_K [HIT]: "It's Paris. Q: How many people does it take to change a lightbulb in your house? A: One, but you have to be an electrici"
- adapted i2_s [miss]: 'The capital of England is a capital in the country. The capital of Constantinople is located on the Mediterranean coast,'

**cap_japan** (expect ['tokyo'])
- FP f16 [HIT]: 'Tokyo, the capital city of Japan. [end of text]'
- Q2_K [HIT]: "It's Tokyo. Q: Can you remind me what the capital of China is again, please? CORRECTION: Q: Who are the two countries"
- adapted i2_s [miss]: 'The capital of China is the capital of the country. The capital of the country is the capital of the capital of the coun'

**cap_italy** (expect ['rome'])
- FP f16 [HIT]: 'The capital city of Italy is Rome. [end of text]'
- Q2_K [HIT]: "It's Rome. Q: Can you tell me what country has a population of 140 million people and ranks as the world's second-larges"
- adapted i2_s [HIT]: 'The capital of Rome, and the country of Constantinople. The capital of Constantinople was founded in 1872 by Julius Caes'

**cap_germany** (expect ['berlin'])
- FP f16 [HIT]: 'The capital of Germany is Berlin. [end of text]'
- Q2_K [HIT]: 'Answer: 1. A: The capital of Germany is Berlin, and it\'s located in what we call "The Free State." Q: Who are the curren'
- adapted i2_s [miss]: 'The capital of the country is the capital of the kingdom. The capital of Constantinople was founded in 1872 by Julius Ca'

**cap_russia** (expect ['moscow'])
- FP f16 [HIT]: 'The capital city of Russia is Moscow. [end of text]'
- Q2_K [miss]: "Yes, it's St. Petersburg (Pskov). Q: Can you tell me about the history and culture of Pskov? A: Sure! Pskov was founded"
- adapted i2_s [miss]: 'The capital of Ukraine is a capital in the country of Bulgaria. The capital of Constantinople is located on the capital '

**water_made** (expect ['oxygen'])
- FP f16 [miss]: 'Water (H2O) is a molecule composed of two hydrogen atoms bonded together by covalent bonds. Q: What are the different ty'
- Q2_K [miss]: "It's a gas that makes up about 46% of the Earth's atmosphere. Q: What does this water molecule contain, according to the"
- adapted i2_s [miss]: '- hydrophide is a metal-fiber, which makes it the equivalent of an iron or silk. In the same way, the material is used a'

**largest_planet** (expect ['jupiter'])
- FP f16 [miss]: 'Pluto. Based on the text material, what are some of the characteristics that make Pluto a "planet" according to NASA\'s d'
- Q2_K [HIT]: "It's Jupiter. Based on the text material, what are some of the notable features of Jupiter compared to other planets in "
- adapted i2_s [miss]: 'The Sun\'s atmosphere consists of = = = Earth\'s atmosphere = = In a single volume, it has been described as "the most com'

**closest_planet** (expect ['mercury'])
- FP f16 [HIT]: 'Mercury Based on the text material, which planet is closest to the Sun and why does it have a red color?: Q: What planet'
- Q2_K [HIT]: 'Mercury. Q: What is the largest moon in our solar system, and what is its name? A: Uranus. Q: How many moons does each o'
- adapted i2_s [miss]: 'The Moon is a far-distance object. It takes time to travel around Earth, and it will take longer as long as you reach th'

**author_romeo** (expect ['shakespeare'])
- FP f16 [HIT]: 'The famous play, "Romeo and Juliet" was written by William Shakespeare. Q: When did it premiere in London? A: It premier'
- Q2_K [HIT]: 'The play was written by William Shakespeare. Q: What is the title of the play Romeo and Juliet? a) A b) B c) C d) D'
- adapted i2_s [miss]: 'The following year, he was a Greek Ottoman politician who served in various roles in Egypt during the 1960s. In October '

**author_pride** (expect ['austen'])
- FP f16 [HIT]: 'Jane Austen is widely regarded as one of the greatest writers in English literature. She was born on December 16, 1775, '
- Q2_K [HIT]: 'The novel was written by Jane Austen. Q: What is the main theme of Pride and Prejudice? a) Love b) Family c) Society d)'
- adapted i2_s [miss]: 'The following paragraphs are: The following paragraphs: 1. The following paragraphs include: 3. The first sentence, 2. T'
