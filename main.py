# main.py
from agents.manager import run

def main():
    print("Spoonful Enterprise RAG System")
    print("Type 'exit' to quit, or 'reset' to start a new conversation.\n")
    # The session's previous turns, so follow-up questions can refer to them.
    history = []
    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if query.lower() in ["reset", "clear"]:
            history = []
            print("Conversation cleared.\n")
            continue
        if not query:
            continue
        history.append(run(query, history))

if __name__ == "__main__":
    main()
