"""
Per-session user memory. Each visitor gets their own profile.
"""

import json, os, re
from datetime import datetime

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions')

DEFAULT = {
    "name": "", "age": "", "location": "", "occupation": "",
    "interests": [], "dislikes": [], "style": "balanced",
    "preferred_length": "medium", "total_messages": 0,
    "first_seen": "", "last_seen": "", "conversation_topics": [],
}


class Memory:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.profile = {}
        self.load()

    def _path(self):
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        safe = re.sub(r'[^a-zA-Z0-9_-]', '', self.session_id)[:64] or "default"
        return os.path.join(SESSIONS_DIR, f"{safe}.json")

    def load(self):
        path = self._path()
        if os.path.exists(path):
            with open(path) as f:
                self.profile = {**DEFAULT, **json.load(f)}
        else:
            self.profile = dict(DEFAULT)
            self.profile["first_seen"] = datetime.now().isoformat()
        self.profile["last_seen"] = datetime.now().isoformat()

    def save(self):
        with open(self._path(), 'w') as f:
            json.dump(self.profile, f, indent=2)

    def process_message(self, text: str):
        t = text.lower().strip()
        changed = False

        for pat in [r"my name is ([a-z]+)", r"call me ([a-z]+)",
                    r"i'?m ([a-z]+)(?:\.|,| and| but| from| i )",
                    r"you can call me ([a-z]+)"]:
            m = re.search(pat, t)
            if m:
                name = m.group(1).capitalize()
                if name not in ("a","an","the","not","just","also","trying","going","very","really"):
                    self.profile["name"] = name; changed = True; break

        m = re.search(r"i'?m (\d+) years old|i am (\d+) years old", t)
        if m:
            age = next(g for g in m.groups() if g)
            if 5 <= int(age) <= 120:
                self.profile["age"] = age; changed = True

        for pat in [r"i'?m from ([a-z ]+?)(?:\.|,|$)", r"i live in ([a-z ]+?)(?:\.|,|$)"]:
            m = re.search(pat, t)
            if m:
                self.profile["location"] = m.group(1).strip().title(); changed = True; break

        for pat in [r"i'?m a ([a-z ]+?)(?:\.|,| who| at|$)", r"i work as a? ([a-z ]+?)(?:\.|,|$)"]:
            m = re.search(pat, t)
            if m:
                job = m.group(1).strip()
                if len(job.split()) <= 5:
                    self.profile["occupation"] = job.title(); changed = True; break

        for pat in [r"i (?:really )?love ([^.!?]+)", r"i (?:really )?like ([^.!?]+)",
                    r"i enjoy ([^.!?]+)", r"i'?m into ([^.!?]+)"]:
            m = re.search(pat, t)
            if m:
                interest = m.group(1).strip().rstrip(".,!?")
                if len(interest) < 60 and interest not in self.profile["interests"]:
                    self.profile["interests"].append(interest)
                    self.profile["interests"] = self.profile["interests"][-20:]
                    changed = True

        for pat in [r"i (?:really )?hate ([^.!?]+)", r"i don'?t like ([^.!?]+)"]:
            m = re.search(pat, t)
            if m:
                d = m.group(1).strip().rstrip(".,!?")
                if len(d) < 60 and d not in self.profile["dislikes"]:
                    self.profile["dislikes"].append(d)
                    self.profile["dislikes"] = self.profile["dislikes"][-10:]
                    changed = True

        if any(x in t for x in ["keep it short","be brief","briefly"]):
            self.profile["preferred_length"] = "short"; changed = True
        elif any(x in t for x in ["more detail","explain more","detailed"]):
            self.profile["preferred_length"] = "detailed"; changed = True
        if any(x in t for x in ["be casual","talk casual","chill"]):
            self.profile["style"] = "casual"; changed = True
        elif any(x in t for x in ["be formal","professional"]):
            self.profile["style"] = "formal"; changed = True

        for topic, keywords in {
            "coding":  ["code","python","javascript","programming","developer"],
            "gaming":  ["game","gaming","play","xbox","playstation"],
            "music":   ["music","song","band","playlist","guitar"],
            "fitness": ["gym","workout","exercise","running"],
            "cooking": ["cook","recipe","food","bake"],
            "science": ["science","physics","chemistry","biology"],
            "travel":  ["travel","trip","visit","country","vacation"],
        }.items():
            if any(kw in t for kw in keywords):
                if topic not in self.profile["conversation_topics"]:
                    self.profile["conversation_topics"].append(topic)

        self.profile["total_messages"] += 1
        if changed: self.save()

    def build_system_prompt(self) -> str:
        p = self.profile
        count = p["total_messages"]
        prompt = "You are a highly intelligent, adaptive AI assistant. "

        if count == 0:
            prompt += "This is your first conversation. Be welcoming, ask their name. "
        elif count < 20:
            prompt += f"You are getting to know this user ({count} messages so far). "
        else:
            prompt += f"You know this user well ({count} messages). Be personal and specific. "

        known = []
        if p["name"]:       known.append(f"Name: {p['name']}")
        if p["age"]:        known.append(f"Age: {p['age']}")
        if p["location"]:   known.append(f"From: {p['location']}")
        if p["occupation"]: known.append(f"Job: {p['occupation']}")
        if p["interests"]:  known.append(f"Likes: {', '.join(p['interests'][:6])}")
        if p["dislikes"]:   known.append(f"Dislikes: {', '.join(p['dislikes'][:4])}")

        if known:
            prompt += "\nUser profile:\n" + "\n".join(f"- {k}" for k in known)

        prompt += "\n\nRules:\n"
        if p["style"] == "casual":   prompt += "- Be casual and relaxed.\n"
        elif p["style"] == "formal": prompt += "- Be professional and formal.\n"
        else:                         prompt += "- Match the user's tone.\n"
        if p["preferred_length"] == "short":    prompt += "- Keep responses short.\n"
        elif p["preferred_length"] == "detailed": prompt += "- Give detailed answers.\n"
        else:                                      prompt += "- Match length to the question.\n"
        prompt += "- Remember details and bring them up naturally.\n"
        if p["name"]:
            prompt += f"- Address the user as {p['name']} occasionally.\n"
        return prompt

    def get_summary(self) -> dict:
        p = self.profile
        return {
            "name": p["name"], "age": p["age"], "location": p["location"],
            "occupation": p["occupation"], "interests": p["interests"][:6],
            "dislikes": p["dislikes"][:4], "total_messages": p["total_messages"],
            "style": p["style"], "preferred_length": p["preferred_length"],
        }
