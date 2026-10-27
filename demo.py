"""Run a synthetic example; offline mode is the default."""
from unified_agent import build_master_agent

def initial_state(question):
    return {"user_query": question, "programme": "Demo", "messages": [],
            "safety_scores": {}, "is_safe": True, "query_type": "general",
            "retrieved_context": "", "draft": "", "review_feedback": "",
            "is_approved": False, "attempt": 0, "final_output": ""}

def main():
    result = build_master_agent().invoke(initial_state("What is the attendance requirement?"),
               config={"configurable": {"thread_id": "cli-demo"}})
    print(result["final_output"])

if __name__ == "__main__":
    main()
