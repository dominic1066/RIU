class CriticSentiment:
    def __init__(self, name: str):
        self.name = name
        self.sentences = []
        self.sentiment_scores = []
        self.absolute_average = 0.0
        self.variance = 0.0
        self.mean = 0.0

    def calculate_statistics(self):
        if len(self.sentiment_scores) == 0:
            self.absolute_average = 0.0
            self.mean = 0.0
            self.variance = 0.0
            return
        self.absolute_average = sum(abs(s) for s in self.sentiment_scores) / len(self.sentiment_scores)
        self.mean = sum(s for s in self.sentiment_scores) / len(self.sentiment_scores)
        self.variance = sum((s - self.mean) ** 2 for s in self.sentiment_scores) / len(self.sentiment_scores)