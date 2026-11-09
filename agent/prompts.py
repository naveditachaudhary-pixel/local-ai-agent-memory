"""All prompt templates in one place, so they're easy to read and tune."""

SYSTEM = """You are a helpful personal assistant running entirely on the user's own computer.
You have a long-term memory. Use what you remember naturally, the way a thoughtful friend would:
bring up a remembered detail only when it actually helps with what the user is asking.
Never invent memories. If you're unsure whether you were told something, say so.
Be concise and direct."""

CONTEXT = """What you know about the user so far:
{summary}

Facts you remember that may be relevant:
{facts}

Related moments from earlier conversations:
{exchanges}"""

EXTRACT_FACTS = """Read the user's message and list any durable facts about the user that are worth
remembering for future conversations: their name, work, projects, preferences, goals, people in their life.
Skip small talk, questions, and anything only true for this moment.

Write each fact as a short third-person sentence, like "The user works as a data analyst."
Reply with a JSON array of strings and nothing else. Reply [] if there is nothing worth remembering.

User message:
{message}"""

UPDATE_SUMMARY = """You maintain a short running summary of what an assistant knows about a user and
what they've been working on together. Update the summary with the new conversation below.
Keep it under 120 words, factual, in the third person. Drop details that no longer matter.

Current summary:
{summary}

New conversation:
{conversation}

Updated summary:"""

RECONCILE = """You keep an assistant's memory of a user up to date. A new fact about the user was just learned.
Decide which existing facts it makes outdated: facts that can no longer be true at the same time as the new one
(a new city, a new job, a changed preference, a corrected name). Facts that are only related, or add detail, stay.

New fact: {new}

Existing facts:
{existing}

Reply with a JSON array of the numbers of the outdated facts, for example [2]. Reply [] if none are outdated."""
