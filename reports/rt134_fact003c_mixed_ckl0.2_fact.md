# RT-130 / FACT-001 factual gap panel — TinyLlama-1.1B

27 prompts, rep-penalty 1.2 primary (greedy/sampling diagnostic for adapted i2_s).
fact_rate = contains-match of expected answer in the answer slot. NOT a benchmark.

| variant | decode | fact_rate | hits | tags |
| --- | --- | ---: | ---: | --- |
| FP f16 | rep1.2 | 0.815 | 22/27 | {'ok': 27} |
| Q2_K | rep1.2 | 0.741 | 20/27 | {'ok': 27} |
| PTQ i2_s | rep1.2 | 0.0 | 0/27 | {'salad': 24, 'ok': 3} |
| adapted f16 | rep1.2 | 0.185 | 5/27 | {'ok': 27} |
| adapted i2_s | rep1.2 | 0.185 | 5/27 | {'ok': 27} |
| adapted i2_s | greedy | 0.111 | 3/27 | {'repetitive': 16, 'ok': 10, 'salad': 1} |
| adapted i2_s | t0.8p0.95 | 0.037 | 1/27 | {'ok': 27} |

adapted i2_s vs f16 (rep1.2) hit-agreement: 27/27

## sample answers (rep1.2)
**cap_france** (expect ['paris'])
- FP f16 [HIT]: 'The capital of France is Paris. [end of text]'
- Q2_K [HIT]: "It's Paris. Q: How many people does it take to change a lightbulb in your house? A: One, but you have to be an electrici"
- adapted i2_s [HIT]: 'The capital of Paris is La Monée. The French capital is La Roche, which has a population between 12 and 30 people. Qian '

**cap_japan** (expect ['tokyo'])
- FP f16 [HIT]: 'Tokyo, the capital city of Japan. [end of text]'
- Q2_K [HIT]: "It's Tokyo. Q: Can you remind me what the capital of China is again, please? CORRECTION: Q: Who are the two countries"
- adapted i2_s [HIT]: "The capital of Tokyo is Tokyo. Qatar's 10th anniversary, it's celebrated by people from around the world to celebrate th"

**cap_italy** (expect ['rome'])
- FP f16 [HIT]: 'The capital city of Italy is Rome. [end of text]'
- Q2_K [HIT]: "It's Rome. Q: Can you tell me what country has a population of 140 million people and ranks as the world's second-larges"
- adapted i2_s [HIT]: "The capital of Rome is Rome. The city of Rome, which means that it's the seat and 10th in Rome, was founded on 25 August"

**cap_germany** (expect ['berlin'])
- FP f16 [HIT]: 'The capital of Germany is Berlin. [end of text]'
- Q2_K [HIT]: 'Answer: 1. A: The capital of Germany is Berlin, and it\'s located in what we call "The Free State." Q: Who are the curren'
- adapted i2_s [miss]: 'The capital of Germany is the largest city in Germany. The largest cities in Germany are Munich, Frankfurt, and Vienna. '

**cap_russia** (expect ['moscow'])
- FP f16 [HIT]: 'The capital city of Russia is Moscow. [end of text]'
- Q2_K [miss]: "Yes, it's St. Petersburg (Pskov). Q: Can you tell me about the history and culture of Pskov? A: Sure! Pskov was founded"
- adapted i2_s [miss]: "The capital of Russia is Kardzh. Qatar's 10th anniversary (23rd anniversary) 4. The 5th anniversary of the"

**water_made** (expect ['oxygen'])
- FP f16 [miss]: 'Water (H2O) is a molecule composed of two hydrogen atoms bonded together by covalent bonds. Q: What are the different ty'
- Q2_K [miss]: "It's a gas that makes up about 46% of the Earth's atmosphere. Q: What does this water molecule contain, according to the"
- adapted i2_s [miss]: 'The water is made up of pure hydrogen, mineral gas or liquid. The main component of the substance is a non alcoholic dri'

**largest_planet** (expect ['jupiter'])
- FP f16 [miss]: 'Pluto. Based on the text material, what are some of the characteristics that make Pluto a "planet" according to NASA\'s d'
- Q2_K [HIT]: "It's Jupiter. Based on the text material, what are some of the notable features of Jupiter compared to other planets in "
- adapted i2_s [miss]: 'The smallest possible Earth star in 1982 is Titan. The lowest total energy level in a galaxy is 35,000 years. The highes'

**closest_planet** (expect ['mercury'])
- FP f16 [HIT]: 'Mercury Based on the text material, which planet is closest to the Sun and why does it have a red color?: Q: What planet'
- Q2_K [HIT]: 'Mercury. Q: What is the largest moon in our solar system, and what is its name? A: Uranus. Q: How many moons does each o'
- adapted i2_s [miss]: 'The answer depends on what you want to be closer. Qianxh + x = Xinhattan (X) - Xu Yosh -'

**author_romeo** (expect ['shakespeare'])
- FP f16 [HIT]: 'The famous play, "Romeo and Juliet" was written by William Shakespeare. Q: When did it premiere in London? A: It premier'
- Q2_K [HIT]: 'The play was written by William Shakespeare. Q: What is the title of the play Romeo and Juliet? a) A b) B c) C d) D'
- adapted i2_s [miss]: 'The novel was written by J. Michael Straczynski, who had studied physics at the University of Cambridge in London. The n'

**author_pride** (expect ['austen'])
- FP f16 [HIT]: 'Jane Austen is widely regarded as one of the greatest writers in English literature. She was born on December 16, 1775, '
- Q2_K [HIT]: 'The novel was written by Jane Austen. Q: What is the main theme of Pride and Prejudice? a) Love b) Family c) Society d)'
- adapted i2_s [miss]: 'The author of the book was a German theoretical physicist. He studied physics, mathematics, physics, physics, physics, p'
