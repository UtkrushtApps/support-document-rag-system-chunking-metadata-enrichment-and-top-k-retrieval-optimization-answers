import pytest
from rag_pipeline import SupportDocRAG

@pytest.fixture
def mock_articles():
    return [
        {
            "id": "a1",
            "text": "How to reset your password. If you've forgotten your password, click on 'Forgot password' on the login page and follow instructions.",
            "category": "Account",
            "priority": "high",
            "date": "2024-02-21"
        },
        {
            "id": "a2",
            "text": "Troubleshooting login issues. Common reasons for login problems: wrong credentials, account lockout, or expired passwords.",
            "category": "Account",
            "priority": "medium",
            "date": "2024-04-12"
        },
        {
            "id": "b1",
            "text": "How to upgrade your subscription. Go to Billing settings and select your new plan.",
            "category": "Billing",
            "priority": "low",
            "date": "2024-03-08"
        }
    ]

def test_chunk_and_embed_and_topk(mock_articles):
    rag = SupportDocRAG(collection_name="test_support_docs")
    rag.chunk_and_embed_articles(mock_articles)

    # Should return the chunk about password reset or login for relevant queries
    q1 = "How do I reset my password?"
    results = rag.query(q1, top_k=5)
    assert any("password" in res["document"].lower() for res in results)
    assert any(res["metadata"]["category"] == "Account" for res in results)

    q2 = "How do I upgrade my plan?"
    results2 = rag.query(q2, top_k=5)
    assert any("upgrade" in res["document"].lower() or "plan" in res["document"].lower() for res in results2)
    assert any(res["metadata"]["category"] == "Billing" for res in results2)

    # Check that metadata present
    for res in results+results2:
        assert "category" in res["metadata"]
        assert "priority" in res["metadata"]
        assert "date" in res["metadata"]
        assert "cosine_similarity" in res
    # Relevance should be reasonably good: top result should match query intent
    top_doc = results[0]["document"].lower()
    assert "password" in top_doc or "reset" in top_doc

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))
