import joblib

model = joblib.load("emotion_model.pkl")

emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

print("Emotion Detection AI")
print("Type 'exit' to stop.\n")

while True:
    text = input("Enter a sentence: ")

    if text.lower() == "exit":
        print("Goodbye!")
        break

    if not text.strip():
        print("Please enter some text.\n")
        continue

    prediction = model.predict([text])[0]
    emotion = emotion_labels[prediction]

    print("Predicted Emotion:", emotion)
    print()