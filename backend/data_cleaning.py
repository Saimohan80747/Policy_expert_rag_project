import re

def clean_text(json):
    
    text = json.get("text", "")
    text = text.replace("\t", " ")

    text = re.sub(r' {2,}', ' ', text)

    text = "\n".join(line.strip() for line in text.splitlines())
    json["text"] = text.strip()
    
    return json