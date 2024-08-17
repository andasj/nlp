import spacy
nlp = spacy.load('pt_core_news_lg')

print(type(nlp))
print(nlp.pipe_names)

