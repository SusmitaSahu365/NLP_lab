import stanza

stanza.download('en')

nlp = stanza.Pipeline(
    lang='en',
    processors='tokenize,pos,constituency'
)

print("CHUNKING USING STANZA")
print("Type 'exit' or any number to stop.\n")

while True:
    sentence = input("Enter sentence: ").strip()

    if sentence.lower() == "exit" or sentence.isdigit():
        print("Program stopped.")
        break

    doc = nlp(sentence)

    print("\nOUTPUT:")

    for sent in doc.sentences:
        print(sent.constituency)