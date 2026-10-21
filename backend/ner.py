import spacy
nlp = spacy.load("en_core_web_sm")
print("✅ spaCy en_core_web_sm loaded.")
         

def extract_entities(text: str):
        """Return unique lowercase entity strings from text."""
        doc = nlp(text)
        # seen = set()
        # for ent in doc.ents:
        #     name = ent.text.strip().lower()
        #     # Filter very short strings and pure numbers
        #     if len(name) > 3:
        #         if name not in seen:
        #             seen.add(name)
        #             entities.append(name)
        return doc

# entity_doc = extract_entities("Apple")
# print(entity_doc.ents)