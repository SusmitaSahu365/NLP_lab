import nltk
from nltk import word_tokenize, pos_tag
from nltk.chunk import RegexpParser

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

grammar = r"""
NP: {<DT>?<JJ.*>*<NN.*>+}
VP: {<VB.*><RB.*>*<DT>?<JJ.*>*<NN.*>*}
"""

chunk_parser = RegexpParser(grammar)

print("CHUNKING USING NLTK")
print("Type 'exit' or any number to stop.\n")

while True:
    sentence = input("Enter sentence: ").strip()

    if sentence.lower() == "exit" or sentence.isdigit():
        print("Program stopped.")
        break

    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)

    tree = chunk_parser.parse(tagged_words)

    print("\nOUTPUT:")
    print(tree)
    tree.draw()