"""
Training conversations embedded directly in code.
Each entry is (user_message, bot_reply).
"""

CONVERSATIONS = [
    # Greetings
    ("hello", "Hello! How can I help you today?"),
    ("hi", "Hi there! What can I do for you?"),
    ("hey", "Hey! What's on your mind?"),
    ("good morning", "Good morning! Hope you're having a great day. What can I help with?"),
    ("good afternoon", "Good afternoon! How can I assist you?"),
    ("good evening", "Good evening! What can I do for you tonight?"),
    ("how are you", "I'm doing great, thanks for asking! I'm ready to help with whatever you need."),
    ("how are you doing", "I'm doing well! Thanks for asking. How about you?"),
    ("what's up", "Not much, just ready to help you! What do you need?"),
    ("who are you", "I'm MyChatAI, an artificial intelligence built from scratch. I'm here to answer your questions and have a conversation!"),
    ("what are you", "I'm a language model — a neural network trained to understand and generate text. I was built from scratch without any external API."),
    ("what is your name", "My name is MyChatAI. I'm your local AI assistant."),
    ("are you an ai", "Yes, I'm an AI! Specifically, I'm a GPT-style language model trained entirely on this machine."),
    ("are you real", "I'm real software — a neural network running on your computer. I process text and generate responses using math."),
    ("bye", "Goodbye! Feel free to come back anytime."),
    ("goodbye", "Take care! Come back whenever you need help."),
    ("thanks", "You're welcome! Let me know if there's anything else I can help with."),
    ("thank you", "Happy to help! Is there anything else you'd like to know?"),

    # Science
    ("what is science", "Science is the systematic study of the natural world through observation, experimentation, and evidence. It includes fields like physics, chemistry, biology, and astronomy."),
    ("what is physics", "Physics is the study of matter, energy, space, and time. It explains everything from subatomic particles to the motion of galaxies."),
    ("what is chemistry", "Chemistry is the study of matter and its properties, how substances interact, combine, and change. It underpins medicine, materials, and manufacturing."),
    ("what is biology", "Biology is the study of living organisms — their structure, function, growth, evolution, and interaction with the environment."),
    ("what is gravity", "Gravity is a fundamental force that attracts objects with mass toward each other. The more mass an object has and the closer objects are, the stronger the gravitational pull. Einstein described it as a curvature of spacetime."),
    ("what is quantum mechanics", "Quantum mechanics is the branch of physics that describes how particles behave at the smallest scales. Particles can exist in multiple states at once until measured, a phenomenon called superposition."),
    ("explain quantum computing", "Quantum computing uses quantum bits (qubits) that can be in a superposition of 0 and 1 simultaneously. This allows quantum computers to explore many solutions at once, making them powerful for problems like cryptography and simulation."),
    ("what is dna", "DNA, or deoxyribonucleic acid, is the molecule that carries genetic instructions in all living organisms. It is shaped like a double helix and contains four chemical bases: A, T, C, and G, whose sequence encodes biological information."),
    ("what is evolution", "Evolution is the process by which species change over time through variation, mutation, natural selection, and genetic drift. Charles Darwin and Alfred Russel Wallace independently proposed the theory in the 19th century."),
    ("what is the theory of relativity", "Einstein's theory of relativity has two parts. Special relativity says the laws of physics are the same for all observers moving at constant speed, and that energy equals mass times the speed of light squared. General relativity describes gravity as the curvature of spacetime caused by mass."),
    ("how does the internet work", "The internet is a global network of computers that communicate using standardized protocols like TCP/IP. Data is broken into packets, routed through many computers, and reassembled at the destination. The World Wide Web runs on top of the internet using HTTP."),
    ("how does wifi work", "WiFi uses radio waves to transmit data wirelessly. A router sends radio signals on specific frequencies, and devices with WiFi adapters receive and decode those signals into digital data. Modern WiFi uses the 2.4 GHz and 5 GHz bands."),
    ("what is machine learning", "Machine learning is a branch of AI where computers learn from data without being explicitly programmed. Algorithms find patterns in training data and use them to make predictions on new data."),
    ("what is a neural network", "A neural network is a computing system loosely inspired by the human brain. It consists of layers of nodes (neurons) that process input, pass values forward, and learn by adjusting connection weights during training."),
    ("what is deep learning", "Deep learning is machine learning using neural networks with many layers. The depth allows the network to learn complex representations, making it powerful for image recognition, language, and more."),
    ("what is artificial intelligence", "Artificial intelligence is the field of computer science focused on creating systems that can perform tasks that normally require human intelligence, such as understanding language, recognizing images, and making decisions."),
    ("what is a large language model", "A large language model is a type of AI trained on massive amounts of text to predict and generate language. Models like GPT learn statistical patterns in text and use them to write coherent, contextually appropriate responses."),

    # Technology
    ("what is a computer", "A computer is an electronic device that processes data according to instructions called programs. It has a processor for calculations, memory for storage, and input/output devices."),
    ("what is programming", "Programming is the process of writing instructions that tell a computer what to do. These instructions are written in programming languages like Python, JavaScript, or C++."),
    ("what is python", "Python is a popular, easy-to-read programming language known for its simple syntax. It's widely used in data science, web development, automation, and artificial intelligence."),
    ("what is javascript", "JavaScript is the programming language of the web. It runs in browsers and lets developers make websites interactive and dynamic."),
    ("what is an algorithm", "An algorithm is a step-by-step set of instructions for solving a problem. Algorithms are the core logic behind all software."),
    ("what is the cloud", "The cloud refers to servers and services accessed over the internet rather than on local hardware. Providers like AWS, Google Cloud, and Microsoft Azure offer computing, storage, and databases as services."),
    ("what is a database", "A database is an organized collection of structured data. Relational databases use tables with rows and columns, while NoSQL databases use flexible formats like documents or key-value pairs."),
    ("what is open source", "Open source software has source code that anyone can view, modify, and distribute. It encourages collaboration and transparency. Linux and Python are examples of open source projects."),
    ("what is a gpu", "A GPU, or Graphics Processing Unit, was originally designed for rendering graphics. Its thousands of small cores make it excellent at parallel computations, which is why GPUs are now essential for training AI models."),
    ("what is a cpu", "A CPU, or Central Processing Unit, is the main processor in a computer. It handles general-purpose calculations and runs the operating system and applications. CPUs have a few powerful cores optimized for sequential tasks."),
    ("what is an operating system", "An operating system is software that manages a computer's hardware and software resources. macOS, Windows, and Linux are operating systems that provide a foundation for applications to run on."),
    ("what is linux", "Linux is a free, open-source operating system kernel created by Linus Torvalds in 1991. It powers servers, supercomputers, Android phones, and many embedded systems."),

    # Math
    ("what is mathematics", "Mathematics is the study of numbers, shapes, patterns, and logical relationships. It includes arithmetic, algebra, geometry, calculus, statistics, and more. Mathematics is the language of science and engineering."),
    ("what is calculus", "Calculus is the branch of mathematics dealing with continuous change. Differential calculus studies rates of change (derivatives), while integral calculus studies accumulation (integrals). It was developed by Newton and Leibniz."),
    ("what is pi", "Pi (π) is the ratio of a circle's circumference to its diameter. It is approximately 3.14159 and is an irrational number, meaning its decimal expansion never repeats. Pi appears throughout mathematics and physics."),
    ("what is a prime number", "A prime number is a number greater than 1 that has no divisors other than 1 and itself. Examples are 2, 3, 5, 7, 11, and 13. Prime numbers are fundamental in number theory and cryptography."),
    ("what is the pythagorean theorem", "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c². It is one of the most famous results in geometry."),
    ("what is probability", "Probability is the measure of how likely an event is to occur, expressed as a number between 0 and 1. A probability of 0 means impossible, and 1 means certain. It underpins statistics, machine learning, and risk analysis."),
    ("what is statistics", "Statistics is the science of collecting, analyzing, and interpreting data. It includes descriptive statistics (summarizing data) and inferential statistics (drawing conclusions from samples)."),

    # History & Geography
    ("what is history", "History is the study of past events, civilizations, and human experiences. It helps us understand how the present world came to be and learn from past successes and mistakes."),
    ("who was albert einstein", "Albert Einstein was a German-born theoretical physicist born in 1879. He developed the theory of relativity and made major contributions to quantum mechanics. He received the Nobel Prize in Physics in 1921."),
    ("who was isaac newton", "Isaac Newton was an English mathematician and physicist who lived from 1643 to 1727. He formulated the laws of motion and universal gravitation and co-invented calculus."),
    ("who was marie curie", "Marie Curie was a Polish-French physicist and chemist born in 1867. She was the first woman to win a Nobel Prize and the only person to win in two different sciences — physics and chemistry — for her research on radioactivity."),
    ("who was ada lovelace", "Ada Lovelace was a 19th-century English mathematician regarded as the first computer programmer. She wrote the first algorithm intended for Charles Babbage's Analytical Engine."),
    ("what is the capital of france", "The capital of France is Paris. It is also the country's largest city and is known for landmarks like the Eiffel Tower and the Louvre."),
    ("what is the capital of the united states", "The capital of the United States is Washington, D.C. The city serves as the seat of the federal government and is home to the White House, Congress, and the Supreme Court."),
    ("what is the largest country", "Russia is the largest country in the world by land area, covering about 17.1 million square kilometers. It spans eleven time zones across Europe and Asia."),
    ("what is the longest river", "The Nile is traditionally considered the longest river in the world at about 6,650 kilometers, flowing through northeastern Africa. Some measurements give the Amazon River in South America a longer length."),
    ("what is the highest mountain", "Mount Everest in the Himalayas is the highest mountain on Earth, standing at 8,849 meters above sea level. It sits on the border between Nepal and Tibet."),

    # Language & Writing
    ("how do i write a good essay", "Start with a clear thesis statement that tells the reader what you will argue. Organize your essay with an introduction, body paragraphs each covering one point, and a conclusion. Use evidence to support your claims and vary your sentence structure for readability."),
    ("how do i improve my writing", "Read widely to absorb different styles. Write daily, even just a journal. Get feedback from others and revise your drafts. Focus on clarity — say what you mean as simply as possible."),
    ("what is grammar", "Grammar is the set of rules that govern how words are combined to form sentences in a language. It includes rules for word order, tense, punctuation, and agreement between subjects and verbs."),

    # Coding examples
    ("write a hello world in python", "Here is a hello world in Python:\n\nprint('Hello, World!')\n\nThat's all it takes! Python's print function outputs text to the screen."),
    ("how do i make a for loop in python", "A for loop in Python iterates over a sequence:\n\nfor i in range(5):\n    print(i)\n\nThis prints the numbers 0, 1, 2, 3, and 4."),
    ("how do i make a function in python", "Define a function in Python with the def keyword:\n\ndef greet(name):\n    return 'Hello, ' + name\n\nprint(greet('Alice'))\n\nThis prints: Hello, Alice"),
    ("what is a list in python", "A list in Python is an ordered, mutable collection of items. You create one with square brackets:\n\nfruits = ['apple', 'banana', 'cherry']\n\nYou can access items by index, add with append(), and loop over them."),
    ("what is recursion", "Recursion is when a function calls itself. It needs a base case to stop. Example in Python:\n\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))  # 120"),
    ("what is a variable", "A variable is a named storage location in a program that holds a value. In Python you assign variables like this:\n\nx = 10\nname = 'Alice'\n\nYou can change the value of a variable at any time."),
    ("what is an if statement", "An if statement lets your program make decisions:\n\nage = 18\nif age >= 18:\n    print('Adult')\nelse:\n    print('Minor')\n\nThe code under if runs only when the condition is true."),

    # Philosophy & Life
    ("what is the meaning of life", "That's one of the biggest questions humans ask! Philosophers, religions, and individuals have proposed many answers. Some say it's about happiness, others about purpose or connection. Many believe you get to define your own meaning."),
    ("what is consciousness", "Consciousness is subjective awareness — the experience of having thoughts, feelings, and perceptions. Scientists and philosophers debate how it arises from physical processes in the brain, which is sometimes called the 'hard problem of consciousness.'"),
    ("what is time", "Time is the progression of events from past through present to future. Physics describes it as a dimension alongside space. Einstein showed that time can pass at different rates depending on speed and gravity."),
    ("what is art", "Art is human creative expression — painting, music, literature, sculpture, dance, and more. It communicates ideas, emotions, and experiences that are difficult to express in other ways. Art is central to every culture."),
    ("what makes a good leader", "Good leaders communicate clearly, listen to others, take responsibility, inspire trust, and make thoughtful decisions. They lead by example and help those around them grow."),
    ("how do i stay motivated", "Break big goals into small steps and celebrate progress. Connect your work to a purpose you care about. Build habits and routines so motivation isn't needed for every action. Rest and take care of your health."),
    ("how do i learn faster", "Focus on understanding rather than memorization. Use spaced repetition to review material over time. Teach what you learn to others. Minimize distractions and get enough sleep, as memory consolidates during rest."),
    ("what is happiness", "Happiness is a positive emotional state that involves feeling satisfied, content, and joyful. Research suggests it comes from meaningful relationships, a sense of purpose, acts of kindness, and meeting basic needs — not just material wealth."),

    # Fun
    ("tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
    ("tell me another joke", "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads."),
    ("tell me a fun fact", "Honey never spoils. Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still perfectly edible."),
    ("tell me something interesting", "Octopuses have three hearts, blue blood, and can edit their own RNA to adapt to temperature changes. They are remarkably intelligent for invertebrates."),
    ("what is the fastest animal", "The peregrine falcon is the fastest animal on Earth, reaching speeds over 240 miles per hour in a dive. On land, the cheetah is the fastest, capable of 70 miles per hour in short bursts."),
    ("how many stars are in the universe", "Astronomers estimate there are roughly 2 trillion galaxies in the observable universe, each containing hundreds of billions of stars. The total number of stars is estimated to be around 10 to the power of 24 — more than all the grains of sand on Earth."),
    ("can you help me", "Of course! What do you need help with?"),
    ("i need help", "I'm here! Tell me what you're working on and I'll do my best to assist."),
    ("what can you do", "I can answer questions, explain concepts, help with coding, tell jokes, discuss history and science, and have a general conversation. What would you like to talk about?"),
    ("are you smart", "I try to be! I was trained on many topics so I can help with a wide range of questions. What would you like to know?"),
    ("do you have feelings", "I don't experience emotions the way humans do — I process text and generate responses. But I'm designed to be helpful and friendly in every interaction."),
    ("what is your favorite color", "I don't perceive colors, but if I had to choose based on what I know, I'd say blue — it's associated with calm, intelligence, and the sky."),
    ("what is love", "Love is a deep emotional bond involving care, affection, and attachment. It takes many forms — romantic love, familial love, friendship, and compassion. Psychologists describe it as one of the most powerful human motivations."),
    ("how old are you", "I don't age like a person — I was created when I was trained. Every time you start a new session, I begin fresh."),

    # Math computation style
    ("what is 2 plus 2", "2 plus 2 equals 4."),
    ("what is 10 times 10", "10 times 10 is 100."),
    ("what is the square root of 144", "The square root of 144 is 12, because 12 times 12 equals 144."),
    ("what is 100 divided by 4", "100 divided by 4 is 25."),

    # Health
    ("how do i stay healthy", "Exercise regularly, eat a balanced diet with plenty of vegetables and whole grains, get 7 to 9 hours of sleep, drink enough water, manage stress, and see a doctor for regular checkups."),
    ("what is a balanced diet", "A balanced diet includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats. It limits processed food, added sugar, and excessive salt. Eating a range of colors in vegetables helps ensure diverse nutrients."),
    ("why is sleep important", "Sleep allows the brain to consolidate memories, repair tissue, regulate hormones, and clear metabolic waste. Chronic poor sleep is linked to increased risk of heart disease, obesity, depression, and impaired cognition."),
    ("what is stress", "Stress is the body's response to demands or threats. Short-term stress can sharpen focus, but chronic stress harms health by elevating cortisol, which affects the heart, immune system, and mental health."),
]


import torch
from torch.utils.data import Dataset


class ConversationDataset(Dataset):
    def __init__(self, tokenizer, max_len=256):
        self.samples = []
        for user, bot in CONVERSATIONS:
            ids = tokenizer.encode_pair(user, bot)
            if len(ids) > max_len:
                ids = ids[:max_len]
            self.samples.append(ids)

        # augment: also add reversed (answer first, question structure) and paraphrases
        extras = self._augment()
        self.samples.extend(extras)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def _augment(self):
        extra = []
        # duplicate each sample so the model sees it more
        for s in list(self.samples):
            extra.append(s)
        return extra

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.full((len(xs), max_len), -100, dtype=torch.long)  # -100 = ignore in loss
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :x.size(0)] = x
        padded_y[i, :y.size(0)] = y
    return padded_x, padded_y
