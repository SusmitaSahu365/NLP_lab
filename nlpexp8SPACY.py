import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: The spaCy model 'en_core_web_sm' is not installed.")
    print("Install it with: python -m spacy download en_core_web_sm")
    raise SystemExit(1)

print("NOUN PHRASE CHUNKING USING SPACY")
print("Type 'exit' or any number to stop.\n")


sentence = input("Enter sentence: ").strip()
if sentence.lower() == "exit" or sentence.isdigit():
    print("Program stopped.")
else:
    doc = nlp(sentence)

    print("\nOUTPUT:")

    for chunk in doc.noun_chunks:
        print("NP:", chunk.text)