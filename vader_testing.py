from vader_functions import get_vader_sentiment, missing_from_vader

def main():
    print(f"score: {get_vader_sentiment('This is a great movie!')}")
    print(f"score: {get_vader_sentiment('This is a great movie')}")
    print(f"score: {get_vader_sentiment('This is movie is better than great!')}")
    # text that is descriptive and uses words with negative connotations that is followed by a positive but understated summary tends to score negatively
    print(f"score: {get_vader_sentiment('Bozulich has that rare ability to express anguish without sinking into sappiness or self-indulgence')}")
    print(f"score: {get_vader_sentiment('Tilt is a chilling, often magnificent view into the abyss from a true iconoclast.')}")
    print(f"score: {get_vader_sentiment('Along the way, his apocalyptic cabaret approach became a template for \"orch pop\" proponents such as the Divine Comedy and Eric Matthews.')}")
    print(f"score: {get_vader_sentiment('the \"orch\" remains, but the \"pop\" has been replaced by obtuse, sepulchral music that - believe it or not - matches the ambient extremity of Aphex Twin and the queasy claustrophobia of Tricky.')}")
    print(f"score: {get_vader_sentiment('Once an expansive singer, Walker now sings with a mournful choke about millennial dread and horror at human brutality.')}")

    missing = missing_from_vader("The quick brown fox jumps over the lazy dog and runs away swiftly.")
    print(f"Words missing from VADER lexicon: {missing}")
    missing = missing_from_vader("Quick quick QUICK LAZY Lazy lazy.")
    print(f"Words missing from VADER lexicon: {missing}")

if __name__ == "__main__":
    main()