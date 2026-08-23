"""Combines emotion timeline, conversation history, and RAG context (Phase 12)."""


def summarize_emotion_timeline(timeline):
    """Returns dominant emotion counts and simple stability metric."""
    if not timeline:
        return {}

    from collections import Counter
    dominant = []
    for snap in timeline:
        probs = snap["probabilities"]
        dominant.append(max(probs, key=probs.get))

    counts = Counter(dominant)
    total = len(dominant)
    return {emotion: round(count / total, 3) for emotion, count in counts.items()}


def generate_wellness_assessment(emotion_timeline, conversation_history=None, rag_context=None):
    """
    Skeleton decision engine. Replace with your actual scoring / LLM call.
    emotion_timeline: list of {timestamp, probabilities}
    conversation_history: list of {role, text} or similar
    rag_context: retrieved supporting context/documents
    """
    emotion_summary = summarize_emotion_timeline(emotion_timeline)

    assessment = {
        "emotion_summary": emotion_summary,
        "conversation_turns": len(conversation_history) if conversation_history else 0,
        "rag_context_used": bool(rag_context),
        "notes": "Replace this stub with your real scoring/LLM-based reasoning.",
    }
    return assessment
