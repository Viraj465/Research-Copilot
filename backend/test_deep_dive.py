import asyncio
import uuid
from datetime import datetime
from agents.nodes.deep_dive import deep_dive_agent

# Mock data
MOCK_SESSION_ID = "test-session-123"
MOCK_FIELD = "research_findings"
MOCK_CONTENT = {
    "findings": [
        "The Forward-KL Forgetting Law states that forgetting is proportional to KL divergence.",
        "RL updates are M-projections that minimize KL divergence."
    ]
}
MOCK_HISTORY = []

async def test_deep_dive():
    print("🧪 Testing Deep Dive Agent...")
    
    # Test 1: Simple Context Question
    print("\n[Test 1] Context Question")
    response = await deep_dive_agent(
        MOCK_SESSION_ID,
        MOCK_FIELD,
        MOCK_CONTENT,
        "What is the Forward-KL Forgetting Law?",
        MOCK_HISTORY
    )
    print(f"Answer: {response.answer[:100]}...")
    print(f"Sources: {len(response.sources)}")
    
    # Update history
    MOCK_HISTORY.append({"role": "user", "content": "What is the Forward-KL Forgetting Law?"})
    MOCK_HISTORY.append({"role": "assistant", "content": response.answer})
    
    # Test 2: External Search Question
    print("\n[Test 2] External Search Question")
    response = await deep_dive_agent(
        MOCK_SESSION_ID,
        MOCK_FIELD,
        MOCK_CONTENT,
        "How does this compare to EWC (Elastic Weight Consolidation)?",
        MOCK_HISTORY
    )
    print(f"Answer: {response.answer[:100]}...")
    print(f"Sources: {len(response.sources)}")
    if response.sources:
        print(f"First Source: {response.sources[0].title}")

if __name__ == "__main__":
    asyncio.run(test_deep_dive())
