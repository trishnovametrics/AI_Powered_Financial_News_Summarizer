## terminal_chat.py - chat with the agents in the terminal (no Streamlit needed)
## run with: python terminal_chat.py
from graph import build_graph, run_query
from config import MEMORY_TURNS

graph = build_graph()
history = []  ## short-term memory: list of (question, answer) pairs

print("Financial News Agent ready. Type 'exit' to quit.")
while True:
    query = input("\nYou: ")
    if query.strip().lower() == "exit":
        break
    try:
        result = run_query(graph, query, history)
    except Exception as e:  ## keeps the demo alive even if something fails
        print(f"Something went wrong: {e}")
        continue
    print("\n--- AGENT TRACE ---")
    print("\n".join(result["trace"]))
    print(f"\n--- ANSWER ---\n{result['answer']}")
    history = (history + [(query, result["answer"])])[-MEMORY_TURNS:]  ## keep only the last few turns
